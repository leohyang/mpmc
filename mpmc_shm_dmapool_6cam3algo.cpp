
/*   Example of 6-Camera DMA Pool with MPMC lock-free support 
 *   Date :  2026.01.26 v0.1  leohyang@gmail.com
 *   Build:   g++ -o ~/.bin/mpmc_campool mpmc_shm_dmapool_6cam3algo.cpp -std=c++17 -lpthread
 *            ~/.bin/mpmc_campool
 * 
    - control-plane: shared memory Vyukov rings
    - data-plane: DMA-BUF alloc from dma-heap via ioctl by child producer proc
    - each DMA buffer has its own fd, classic socket fd tx to consumer A/B/C procs
    - each consumer process: 1 socker rd-rx thread + 6/3/2 worker threads
    - each consumer proc has its own v_fd_table_[slot_id]：rx to WR，workers to RD
*/
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <array>
#include <iostream>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>

using std::cout;
using std::cerr;

static constexpr int    NumCams_6    = 6;
static constexpr int    NumGrps_3    = 3;
static constexpr int    FramesPerCam_30 = 30;
static constexpr int    TOTAL_FRM_180 = NumCams_6 * FramesPerCam_30;

static constexpr int A_workers = 6;
static constexpr int B_workers = 3;
static constexpr int C_workers = 2;

static constexpr size_t NslotsPow2 = 64;
static constexpr size_t BUF_BYTES_2MB  = 2 * 1024 * 1024;

static constexpr int TOTAL_SLOTS_64x6 = NumCams_6 * (int)NslotsPow2;

static_assert(std::atomic<size_t>::is_always_lock_free,
              "atomic<size_t> must be lock-free for this demo (Vyukov MPMC requirement)");

static inline uint32_t make_slot_id(uint32_t cam_id, uint32_t slot_id) {
    return cam_id * (uint32_t)NslotsPow2 + slot_id;
}
static inline uint32_t slot_cam(uint32_t slot_id) { return slot_id / (uint32_t)NslotsPow2; }
static inline uint32_t slot_idx(uint32_t slot_id) { return slot_id % (uint32_t)NslotsPow2; }

// ---------- dma-heap ioctl fallback ----------
#ifndef DMA_HEAP_IOC_MAGIC
    #define DMA_HEAP_IOC_MAGIC 'H'
#endif

struct dma_heap_allocation_data {  // fixed/predefined per include/uapi/linux/dma-heap.h
    uint64_t len;
    uint32_t fd;                // explicitly manually define the identical struct for ioctl at user space
    uint32_t fd_flags;
    uint64_t heap_flags;
};

#ifndef DMA_HEAP_IOCTL_ALLOC
    #include <linux/ioctl.h>
    #define DMA_HEAP_IOCTL_ALLOC _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct dma_heap_allocation_data)
#endif

/* ================= Vyukov style bounded MPMC (shared memory) ================= */
template <typename T, size_t Pow2N>
struct MPMCQueue {
    static_assert((Pow2N & (Pow2N - 1)) == 0, "Capacity must be power of 2");
    static constexpr size_t mask_ = Pow2N - 1;

    struct Cell { std::atomic<size_t> seq; T data; };

    alignas(64) std::atomic<size_t> head;
    alignas(64) std::atomic<size_t> tail;
    Cell cells[Pow2N];

    void init() {
        for (size_t i = 0; i < Pow2N; ++i) cells[i].seq.store(i, std::memory_order_relaxed);
        head.store(0, std::memory_order_relaxed);
        tail.store(0, std::memory_order_relaxed);
    }

    bool enqueue(const T& data) {
        Cell* cell;
        size_t hpos = head.load(std::memory_order_relaxed);
        for (;;) {
            cell = &cells[hpos & mask_];
            size_t seq = cell->seq.load(std::memory_order_acquire);
            intptr_t diff = (intptr_t)seq - (intptr_t)hpos;
            if (diff == 0) {
                if (head.compare_exchange_weak(hpos, hpos + 1,
                                              std::memory_order_relaxed, std::memory_order_relaxed))
                    break;
            } else if (diff < 0) {
                return false;
            } else {
                hpos = head.load(std::memory_order_relaxed);
            }
        }
        cell->data = data;
        cell->seq.store(hpos + 1, std::memory_order_release);
        return true;
    }

    bool dequeue(T& out) {
        Cell* cell;
        size_t tpos = tail.load(std::memory_order_relaxed);
        for (;;) {
            cell = &cells[tpos & mask_];
            size_t seq = cell->seq.load(std::memory_order_acquire);
            intptr_t diff = (intptr_t)seq - (intptr_t)(tpos + 1);
            if (diff == 0) {
                if (tail.compare_exchange_weak(tpos, tpos + 1,
                                              std::memory_order_release, std::memory_order_relaxed))
                    break;
            } else if (diff < 0) {
                return false;
            } else {
                tpos = tail.load(std::memory_order_relaxed);
            }
        }
        out = cell->data;
        cell->seq.store(tpos + Pow2N, std::memory_order_release);
        return true;
    }
};

// ctrl shm slot meta
struct FrameSlotMeta {
    uint32_t bytes;
    std::atomic<int> ref_count;
};

template <size_t N>
struct CameraPipeline {
    MPMCQueue<uint32_t, N> free_pool_q;
    MPMCQueue<uint32_t, N> qA, qB, qC;
    FrameSlotMeta slots[N];

    void init(uint32_t cam_id) {
        free_pool_q.init(); qA.init(); qB.init(); qC.init();
        for (uint32_t i = 0; i < (uint32_t)N; ++i) {
            slots[i].bytes = (uint32_t)BUF_BYTES_2MB;
            slots[i].ref_count.store(0, std::memory_order_relaxed);
            free_pool_q.enqueue(make_slot_id(cam_id, i));
        }
    }
};

struct SharedPipelinesCtrl {
    std::atomic<int> start_flag;
    std::atomic<int> producer_done;
    std::atomic<int> group_done[NumGrps_3];
    CameraPipeline<NslotsPow2> cams[NumCams_6];  // 64x slots/camera, ctor 6x cameras

    void init_all() {
        start_flag.store(0, std::memory_order_relaxed);
        producer_done.store(0, std::memory_order_relaxed);
        for (int g = 0; g < NumGrps_3; ++g) 
            group_done[g].store(0, std::memory_order_relaxed);
        for (uint32_t c = 0; c < (uint32_t)NumCams_6; ++c) 
            cams[c].init(c);
    }
};

    // ===== unix socket + SCM_RIGHTS (fd passing) :: classic std =====
    struct MsgHdrPayload {
        uint32_t slot_id;
    };
    static int make_unix_server(const std::string& path) {
        int s = socket(AF_UNIX, SOCK_STREAM, 0);
        if (s < 0) { perror("socket"); return -1; }
        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path.c_str());
        unlink(path.c_str());
        if (bind(s, (sockaddr*)&addr, sizeof(addr)) != 0) { perror("bind"); close(s); return -1; }
        if (listen(s, 8) != 0) { perror("listen"); close(s); return -1; }
        return s;
    }
    static int accept_one(int server_fd) {
        int c = accept(server_fd, nullptr, nullptr);
        if (c < 0) perror("accept");
        return c;
    }
    static int make_unix_client(const std::string& path) {
        int s = socket(AF_UNIX, SOCK_STREAM, 0);
        if (s < 0) { perror("socket"); return -1; }
        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path.c_str());
        // simple retry loop (no sleep)
        for (int i = 0; i < 100000; ++i) {
            if (connect(s, (sockaddr*)&addr, sizeof(addr)) == 0) return s;
            if (errno != ENOENT && errno != ECONNREFUSED) break;
        }
        perror("connect");
        close(s);
        return -1;
    }
    static bool send_fd_with_slot(int sock, uint32_t slot_id, int fd_to_send) {
        MsgHdrPayload payload{slot_id};
        iovec iov{};
        iov.iov_base = &payload;
        iov.iov_len  = sizeof(payload);
        char cmsgbuf[CMSG_SPACE(sizeof(int))];
        std::memset(cmsgbuf, 0, sizeof(cmsgbuf));
        msghdr msg{};
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        msg.msg_control = cmsgbuf;
        msg.msg_controllen = sizeof(cmsgbuf);
        cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type  = SCM_RIGHTS;
        cmsg->cmsg_len   = CMSG_LEN(sizeof(int));
        std::memcpy(CMSG_DATA(cmsg), &fd_to_send, sizeof(int));
        if (sendmsg(sock, &msg, 0) != (ssize_t)sizeof(payload)) {
            perror("sendmsg");
            return false;
        }
        return true;
    }
    static bool recv_fd_with_slot(int sock, uint32_t& slot_id_out, int& fd_out) {
        MsgHdrPayload payload{};
        iovec iov{};
        iov.iov_base = &payload;
        iov.iov_len  = sizeof(payload);
        char cmsgbuf[CMSG_SPACE(sizeof(int))];
        std::memset(cmsgbuf, 0, sizeof(cmsgbuf));
        msghdr msg{};
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        msg.msg_control = cmsgbuf;
        msg.msg_controllen = sizeof(cmsgbuf);
        ssize_t n = recvmsg(sock, &msg, 0);
        if (n == 0) return false; // peer closed
        if (n < 0) { perror("recvmsg"); return false; }
        if (n != (ssize_t)sizeof(payload)) { cerr << "recvmsg short\n"; return false; }
        int fd = -1;
        for (cmsghdr* cmsg = CMSG_FIRSTHDR(&msg); cmsg; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
            if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
                std::memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));
                break;
            }
        }
        if (fd < 0) { cerr << "no fd in SCM_RIGHTS\n"; return false; }
        slot_id_out = payload.slot_id;
        fd_out = fd;
        return true;
    }

// ===== DMA-BUF allocation (producer) 1x only by producer =====
struct DmaBufPool {
    int heap_fd = -1;
    int dma_fd[TOTAL_SLOTS_64x6];
    void* dma_mm_p[TOTAL_SLOTS_64x6];

    bool init_heap(const char* heap_path = "/dev/dma_heap/system") {
        heap_fd = open(heap_path, O_RDWR | O_CLOEXEC);
        if (heap_fd < 0) { perror("open dma_heap"); return false; }
        return true;
    }

    bool alloc_all() {
        for (int i = 0; i < TOTAL_SLOTS_64x6; ++i) {
            dma_fd[i] = -1;
            dma_mm_p[i] = nullptr;
        }

        for (int sid = 0; sid < TOTAL_SLOTS_64x6; ++sid) {
            dma_heap_allocation_data data{};
                data.len = BUF_BYTES_2MB;
                data.fd_flags = O_RDWR | O_CLOEXEC;
                data.heap_flags = 0;
            // 384x ioctl for 384 dma buffers one by one
            if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &data) != 0) { perror("DMA_HEAP_IOCTL_ALLOC"); return false; }

            int fd = (int)data.fd;
            void* p = mmap(nullptr, BUF_BYTES_2MB, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (p == MAP_FAILED) { perror("mmap dmabuf"); close(fd); return false; }
            dma_fd[sid] = fd;
            dma_mm_p[sid] = p;
        }
        return true;
    }

    void destroy_all() {
        for (int sid = 0; sid < TOTAL_SLOTS_64x6; ++sid) {
            if (dma_mm_p[sid] && dma_mm_p[sid] != MAP_FAILED) {
                munmap(dma_mm_p[sid], BUF_BYTES_2MB);
            }
            if (dma_fd[sid] >= 0) close(dma_fd[sid]);
            dma_fd[sid] = -1;
            dma_mm_p[sid] = nullptr;
        }
        if (heap_fd >= 0) close(heap_fd);
        heap_fd = -1;
    }

    inline void* map_ptr(uint32_t slot_id) { return dma_mm_p[(int)slot_id]; }
    inline int   fd_of(uint32_t slot_id)   { return dma_fd[(int)slot_id]; }
};

// ===== consumer process local fd table (process-local) =====
struct LocalFdTable {
    std::vector<std::atomic<int>> v_fd_table_;  // index via slot_id, inited fd=-1 as empty

    LocalFdTable() : v_fd_table_(TOTAL_SLOTS_64x6) {
        for (auto& a : v_fd_table_) a.store(-1, std::memory_order_relaxed);
    }

    void put(uint32_t slot_id, int fd) {
        v_fd_table_[slot_id].store(fd, std::memory_order_release);
    }

    int take_blocking(uint32_t slot_id) {
        for (;;) {
            int fd = v_fd_table_[slot_id].load(std::memory_order_acquire);
            if (fd >= 0) {
                // exchange to -1 immediately, consumed only once by one worker
                int fd_got = v_fd_table_[slot_id].exchange(-1, std::memory_order_acq_rel);
                if (fd_got >= 0) return fd_got;
            }
            std::this_thread::yield();
        }
    }
};

// ===== roles =====

// producer: needs sockets to A/B/C (already accepted), and dmabuf pool
static int run_producer(SharedPipelinesCtrl* shmp, int sfd_a, int sfd_b, int sfd_c, 
                        DmaBufPool& dmapool) 
{
    while (shmp->start_flag.load(std::memory_order_acquire) == 0) std::this_thread::yield();

    std::vector<std::thread> threads;
    threads.reserve(NumCams_6);   // 6x producers/cameras/threads

    for (int camid = 0; camid < NumCams_6; ++camid) {
        threads.emplace_back([shmp, camid, sfd_a, sfd_b, sfd_c, &dmapool] {
            auto& cam_pipe = shmp->cams[camid];

            for (int i = 0; i < FramesPerCam_30; ++i) {
                uint32_t sid = 0;
                while (!cam_pipe.free_pool_q.dequeue(sid)) std::this_thread::yield();

                uint32_t idx = slot_idx(sid);

                // write payload via producer master mapping
                void* p = dmapool.map_ptr(sid);
                std::memset(p, camid, BUF_BYTES_2MB);

                // metadata publish
                cam_pipe.slots[idx].ref_count.store(NumGrps_3, std::memory_order_release);

                // enqueue control-plane handles
                while (!cam_pipe.qA.enqueue(sid)) std::this_thread::yield();
                while (!cam_pipe.qB.enqueue(sid)) std::this_thread::yield();
                while (!cam_pipe.qC.enqueue(sid)) std::this_thread::yield();

                // send data-plane fd copies (dup) to A/B/C
                // each consumer to rx an independent fd (ref to same dmabuf object)
                int fd_master = dmapool.fd_of(sid);

                int fdA = dup(fd_master);
                int fdB = dup(fd_master);
                int fdC = dup(fd_master);
                if (fdA < 0 || fdB < 0 || fdC < 0) { perror("dup"); }

                if (!send_fd_with_slot(sfd_a, sid, fdA)) {}
                if (!send_fd_with_slot(sfd_b, sid, fdB)) {}
                if (!send_fd_with_slot(sfd_c, sid, fdC)) {}

                // after sendmsg, we can close the dup'ed fd in producer
                // (receiver process holds its own fd now)
                if (fdA >= 0) close(fdA);
                if (fdB >= 0) close(fdB);
                if (fdC >= 0) close(fdC);
            }
        });
    }

    for (auto& t : threads) t.join();
    shmp->producer_done.store(1, std::memory_order_release);
    return 0;
}

static int run_consumer(SharedPipelinesCtrl* shmp, int gid, int workers, int sock_in) {
    while (shmp->start_flag.load(std::memory_order_acquire) == 0) std::this_thread::yield();

    auto fdt = std::make_unique<LocalFdTable>(); // LocalFdTable fdt;

    // fd-socker receiver thread: receive TOTAL_FRM_180 fds for this group, store into v_fd_table_
    std::thread rx([&] {
        for (int i = 0; i < TOTAL_FRM_180; ++i) {
            uint32_t sid = 0;
            int fd = -1;
            if (!recv_fd_with_slot(sock_in, sid, fd)) { cerr << gid << " rxer failed\n"; return; }
            fdt->put(sid, fd); // fdt.put(sid, fd);
        }
    });

    std::vector<std::atomic<int>> workerCnt(workers);
    for (auto& x : workerCnt) x.store(0);

    std::vector<std::thread> threads;
    threads.reserve(workers);

    for (int w = 0; w < workers; ++w) {         //  6/3/2 threads of each A/B/C algorithms
        threads.emplace_back([&, w] {  // [C, gid, w, &fdt, &workerCnt]
            int next_cam = 0;

            while (shmp->group_done[gid].load(std::memory_order_relaxed) < TOTAL_FRM_180) {
                uint32_t sid = 0;
                bool got = false;

                // dequeue control-plane slot_id
                for (int k = 0; k < NumCams_6; ++k) {
                    int camid = (next_cam + k) % NumCams_6;
                    auto& cam_pipe = shmp->cams[camid];

                    if (gid == 0) got = cam_pipe.qA.dequeue(sid);
                    else if (gid == 1) got = cam_pipe.qB.dequeue(sid);
                    else got = cam_pipe.qC.dequeue(sid);

                    if (got) { next_cam = (camid + 1) % NumCams_6; break; }
                }

                if (!got) { std::this_thread::yield(); continue; }

                uint32_t camid = slot_cam(sid);
                uint32_t idx = slot_idx(sid);

                // wait for data-plane fd arrival
                int fd = fdt->take_blocking(sid);

                // mmap and touch payload
                void* p = mmap(nullptr, BUF_BYTES_2MB, PROT_READ, MAP_SHARED, fd, 0);
                if (p == MAP_FAILED) { perror("consumer mmap dmabuf"); close(fd); continue; }
                volatile uint8_t v = ((uint8_t*)p)[gid];
                (void)v;
                munmap(p, BUF_BYTES_2MB);
                close(fd);

                // last consumer returns to free_pool (ref_count--)
                if (shmp->cams[camid].slots[idx].ref_count.fetch_sub(
                                1, std::memory_order_acq_rel) == 1) {
                    while (!shmp->cams[camid].free_pool_q.enqueue(sid))
                                 std::this_thread::yield();
                }

                workerCnt[w].fetch_add(1, std::memory_order_relaxed);
                shmp->group_done[gid].fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& t : threads) t.join();
    rx.join();

    cout << "[Group " << gid << "] per-worker:\n";
    for (int i = 0; i < workers; ++i) cout << "  W[" << i << "]=" << workerCnt[i].load() << "\n";
    return 0;
}

// ===== shm helpers =====
static void shm_unmap_destroy(void* p, size_t bytes, int fd, const std::string& name) {
    munmap(p, bytes);
    close(fd);
    shm_unlink(name.c_str());
}

int main() {
    const std::string ctrl_name = "/ipc_ctrl_dmabuf";
    const size_t ctrl_bytes = sizeof(SharedPipelinesCtrl);
    int ctrl_fd = -1;  // Create ctrl shm in parent before fork

    ctrl_fd = shm_open(ctrl_name.c_str(), O_CREAT|O_RDWR, 0666);
    if (ctrl_fd < 0) { perror("shm open failed"); return -1; }
    if (ftruncate(ctrl_fd, (off_t)ctrl_bytes) != 0) { perror("ftruncate"); 
                                                close(ctrl_fd); return -2; }
    void * ctrl_p = mmap(nullptr, ctrl_bytes, PROT_READ|PROT_WRITE,
                        MAP_SHARED, ctrl_fd, 0);
    if (ctrl_p == MAP_FAILED) { perror("mmap"); close (ctrl_fd); return -3; }
    std::memset(ctrl_p, 0, ctrl_bytes);
    auto* shmctl_ptr = reinterpret_cast<SharedPipelinesCtrl*>(ctrl_p);
    shmctl_ptr->init_all();

    // Create 3 unix servers (A/B/C) in parent
    const std::string sockA_path = "/tmp/ipc_dmabuf_A.sock";
    const std::string sockB_path = "/tmp/ipc_dmabuf_B.sock";
    const std::string sockC_path = "/tmp/ipc_dmabuf_C.sock";

    int sfdListenA = make_unix_server(sockA_path);
    int sfdListenB = make_unix_server(sockB_path);
    int sfdListenC = make_unix_server(sockC_path);
    if (sfdListenA < 0 || sfdListenB < 0 || sfdListenC < 0) {
        cerr << "unix server setup failed\n";
        shm_unmap_destroy(ctrl_p, ctrl_bytes, ctrl_fd, ctrl_name);
        return 2;
    }

    auto fork_role = [&](auto fn) -> pid_t {   // lambda1
        pid_t pid = fork();
        if (pid < 0) { perror("fork"); return -1; }
        if (pid == 0) { int rc = fn(); _exit(rc); /* child process */ }
        // in parent process to return new child process's pid
        return pid;
    };

    // Spawn consumers
    pid_t a_pid = fork_role([&] {       // lambda1 to call lambda2a
        int sock_a = make_unix_client(sockA_path);
        if (sock_a < 0) return 10;
        int rc = run_consumer(shmctl_ptr, 0, A_workers, sock_a);
        close(sock_a);
        return rc;
    });

    pid_t b_pid = fork_role([&] {    // lambda1 to call lambda2b
        int sock_b = make_unix_client(sockB_path);
        if (sock_b < 0) return 11;
        int rc = run_consumer(shmctl_ptr, 1, B_workers, sock_b);
        close(sock_b);
        return rc;
    });

    pid_t c_pid = fork_role([&] {     // lambda1 to call lambda2c
        int sock_c = make_unix_client(sockC_path);
        if (sock_c < 0) return 12;
        int rc = run_consumer(shmctl_ptr, 2, C_workers, sock_c);
        close(sock_c);
        return rc;
    });

    // Accept connections in parent for producer 
    int connA = accept_one(sfdListenA);
    int connB = accept_one(sfdListenB);
    int connC = accept_one(sfdListenC);
    if (connA < 0 || connB < 0 || connC < 0) {
        cerr << "accept failed\n";
        shm_unmap_destroy(ctrl_p, ctrl_bytes, ctrl_fd, ctrl_name);
        close(sfdListenA); close(sfdListenB); close(sfdListenC);
        unlink(sockA_path.c_str()); unlink(sockB_path.c_str()); unlink(sockC_path.c_str());
        return 3;
    }

    // Now spawn producer: it will allocate dmabuf pool and send fd via connA/B/C
    pid_t prod_pid = fork_role([&] {    // lambda1 to call lambda2p
        close(sfdListenA); close(sfdListenB); close(sfdListenC);

        DmaBufPool dmapool;  //1x DMA pool only by producer
        if (!dmapool.init_heap("/dev/dma_heap/system")) return -10;
        if (!dmapool.alloc_all()) return -11;

        shmctl_ptr->start_flag.store(1, std::memory_order_release);
        int rc = run_producer(shmctl_ptr, connA, connB, connC, dmapool);

        dmapool.destroy_all();
        close(connA); close(connB); close(connC);
        return rc;
    });

    close(sfdListenA); close(sfdListenB); close(sfdListenC);

    int st = 0;
    waitpid(prod_pid, &st, 0);
    waitpid(a_pid, &st, 0);
    waitpid(b_pid, &st, 0);
    waitpid(c_pid, &st, 0);

    cout << "IPC (ctrl shm + dmabuf + SCM_RIGHTS) finished.\n";
    cout << "Group A done=" << shmctl_ptr->group_done[0].load() << " / " << TOTAL_FRM_180 << "\n";
    cout << "Group B done=" << shmctl_ptr->group_done[1].load() << " / " << TOTAL_FRM_180 << "\n";
    cout << "Group C done=" << shmctl_ptr->group_done[2].load() << " / " << TOTAL_FRM_180 << "\n";
    cout << "Producer done=" << shmctl_ptr->producer_done.load() << "\n";

    // cleanup
    shm_unmap_destroy(ctrl_p, ctrl_bytes, ctrl_fd, ctrl_name);

    close(connA); close(connB); close(connC);
    unlink(sockA_path.c_str());
    unlink(sockB_path.c_str());
    unlink(sockC_path.c_str());
    return 0;
}