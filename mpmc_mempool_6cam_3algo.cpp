/*   ThreadPool with MPMC lock-free support 
 *   Date :  2026.01.24 v0.1  leohyang@gmail.com
 *   Build:   g++ -o ~/.bin/mpmc_campool mpmc_mempool_6cam_3algo.cpp -std=c++17 -lpthread
 *            ~/.bin/mpmc_campool
 */
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <array>
#include <iostream>

using namespace std;

struct FrameSlot {
    void* buf = nullptr;          // 2MB payload
    uint32_t bytes = 0;
    std::atomic<int> ref_count{0}; // A/B/C consumer groups： return by the last one
};

template <typename T, size_t Pow2N>   //  T = "FrameSlot*" ptr
class VyukovMPMC {
    static_assert((Pow2N & (Pow2N - 1)) == 0, "Capacity must be power of 2");

 public:
    VyukovMPMC() {
        for (size_t i = 0; i < Pow2N; ++i) {
            cells_[i].seq.store(i, std::memory_order_relaxed);
        }
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
    }

    bool enqueue(const T& data) {
        Cell* cell;
        size_t hpos = head_.load(std::memory_order_relaxed);

        for (;;) {      // loop
            cell = &cells_[hpos & mask_];
            size_t seq = cell->seq.load(std::memory_order_acquire);  // prebuilt mark
            intptr_t diff = (intptr_t)seq - (intptr_t)hpos;
            if (diff == 0) {        // CAS to add
                if (head_.compare_exchange_weak(
                        hpos, hpos + 1,
                        std::memory_order_relaxed, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {  // full
                return false;       // may need to dequeue, then retry ?
            } else {                // other thread did it earlier, refresh
                hpos = head_.load(std::memory_order_relaxed);
            }
        }
        
        cell->data = data;    // vs. manual copy, std::move if needed/possible
        cell->seq.store(hpos + 1, std::memory_order_release);
        return true;
    }

    bool dequeue(T& out) {
        Cell* cell;
        size_t t = tail_.load(std::memory_order_relaxed);

        for (;;) {
            cell = &cells_[t & mask_];
            size_t seq = cell->seq.load(std::memory_order_acquire);
            long int diff = (long int)seq - (long int)(t + 1);
            if (diff == 0) {
                if (tail_.compare_exchange_weak(
                        t, t + 1,
                        std::memory_order_release, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false;
            } else {
                t = tail_.load(std::memory_order_relaxed);
            }
        }        
        out = cell->data;  //  as output/return

        // t + Pow2N， ready for next round
        cell->seq.store(t + Pow2N, std::memory_order_release);
        return true;
    }

 private:
    static constexpr size_t mask_ = Pow2N - 1;

    struct Cell {
        std::atomic<size_t> seq; // sequence number mark
        T data;         // T is normally a ptr type
    };

    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    Cell cells_[Pow2N];  // descriptor-ringbuf
};


/*  ======   Per-camera lock-free pipeline   ========  */
template <size_t N>
struct CameraPipeline {
    static_assert((N & (N - 1)) == 0, "N must be power of 2");
    
    VyukovMPMC<FrameSlot*, N> free_pool_q; // free pool: 管理 slot 的可复用集合（类似 reserve/free）
    VyukovMPMC<FrameSlot*, N> qA, qB, qC; // 三个 group 的工作队列：fan-out, instantiated Vyukov 3 rings

    std::array<FrameSlot, N> slots;  // 实际 slot 存储（固定 N 个）  ||||||...|||

    explicit CameraPipeline(size_t bytes) {
        for (size_t i = 0; i < N; ++i) {
            size_t sz = ((bytes + 4095) / 4096) * 4096;  // 2MB = 1024*1024*2 = 4096 x 512pages
            slots[i].buf = std::aligned_alloc(4096, sz);
            slots[i].bytes = static_cast<uint32_t>(bytes);
            slots[i].ref_count.store(0, std::memory_order_relaxed);

            free_pool_q.enqueue(&slots[i]);   //  64 slots, 1 slot has 2MB
        }
    }

    ~CameraPipeline() {
        for (auto& s : slots) std::free(s.buf);
    }
};

/* =========  main: 6 cameras + A/B/C groups =============== */
int main() {
    constexpr int NumCams_6 = 6, NumGrps_3 = 3, FramesPerCam = 30;
    constexpr int TOTAL_FRAMES = NumCams_6 * FramesPerCam;

    // Worker configuration
    constexpr int A_workers = 6, B_workers = 3, C_workers = 2;

    // each type/group/algorithm
    std::atomic<int> group_done[NumGrps_3];
    group_done[0].store(0);
    group_done[1].store(0);
    group_done[2].store(0);

    // 6/3/2 v of atomic<int> counters
    std::vector<std::atomic<int>> v_a_(A_workers), v_b_(B_workers), v_c_(C_workers);
    for (auto& a : v_a_) a.store(0);
    for (auto& b : v_b_) b.store(0);
    for (auto& c : v_c_) c.store(0);

    // 6 cameras => 6 pipelines
    std::array<CameraPipeline<64>*, NumCams_6> arr_cams_{};
    for (int i = 0; i < NumCams_6; ++i) {
        arr_cams_[i] = new CameraPipeline<64>(2 * 1024 * 1024);
    }

    std::vector<std::thread> threads;

    /* ---------------- Producers ---------------- */
    for (int id = 0; id < NumCams_6; ++id) {  // 6 cameras producing

        threads.emplace_back([&, id] {
            auto& P = *arr_cams_[id];   // P = Producer

            for (int i = 0; i < FramesPerCam; ++i) {
                FrameSlot* s = nullptr;

                // lock-free reserve：从 free pool 拿 slot
                while (!P.free_pool_q.dequeue(s)) { std::this_thread::yield(); }

                // full payload touch（camera DMA）, populating 2MB
                std::memset(s->buf, id, s->bytes);
                s->ref_count.store(NumGrps_3, std::memory_order_release);

                // multicast to MPSC
                while (!P.qA.enqueue(s)) std::this_thread::yield();
                while (!P.qB.enqueue(s)) std::this_thread::yield();
                while (!P.qC.enqueue(s)) std::this_thread::yield();
            }
        });
    }

    /* ---------------- Consumers ----------------
     * A/B/C have different number of workers：     
     *  ref_count--, the last one returning the buf to free pool 
     */
    auto launch_group = [&](int gid, int workers, std::vector<std::atomic<int>>& workerCnt) {
        for (int w = 0; w < workers; ++w) {

            threads.emplace_back([&, gid, w] {
                int next_cam = 0;

                // a group_done for all the group's workers
                while (group_done[gid].load(std::memory_order_relaxed) < TOTAL_FRAMES) {
                    FrameSlot* s = nullptr;
                    int picked_cam = -1;

                    // round-robin polling
                    for (int k = 0; k < NumCams_6; ++k) {
                        int id = (next_cam + k) % NumCams_6;
                        auto& P = *arr_cams_[id];

                        bool ok = false;
                        if (gid == 0) ok = P.qA.dequeue(s);
                        else if (gid == 1) ok = P.qB.dequeue(s);
                        else ok = P.qC.dequeue(s);

                        if (ok) {
                            picked_cam = id;
                            next_cam = (id + 1) % NumCams_6;
                            break;
                        }
                    }

                    if (!s) {
                        std::this_thread::yield();
                        continue;
                    }
                    
                    // emulating 2MB access or meta lines
                    volatile uint8_t v = static_cast<uint8_t*>(s->buf)[gid];
                    (void)v;

                    // ref_count--, return if this is the last one user
                    if (s->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        while (!arr_cams_[picked_cam]->free_pool_q.enqueue(s))
                            std::this_thread::yield();
                    }

                    workerCnt[w].fetch_add(1, std::memory_order_relaxed);
                    group_done[gid].fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
    };

    launch_group(0, A_workers, v_a_);
    launch_group(1, B_workers, v_b_);
    launch_group(2, C_workers, v_c_);

    for (auto& t : threads) t.join();

    std::cout << "Lock-free Vyukov fan-out finished.\n";
    std::cout << "Group A done=" << group_done[0].load() << " / " << TOTAL_FRAMES << "\n";
    std::cout << "Group B done=" << group_done[1].load() << " / " << TOTAL_FRAMES << "\n";
    std::cout << "Group C done=" << group_done[2].load() << " / " << TOTAL_FRAMES << "\n";

    std::cout << "\nA per-worker:\n";
    for (int i = 0; i < A_workers; ++i) std::cout << "  A[" << i << "]=" << v_a_[i].load() << "\n";
    std::cout << "\nB per-worker:\n";
    for (int i = 0; i < B_workers; ++i) std::cout << "  B[" << i << "]=" << v_b_[i].load() << "\n";
    std::cout << "\nC per-worker:\n";
    for (int i = 0; i < C_workers; ++i) std::cout << "  C[" << i << "]=" << v_c_[i].load() << "\n";

    for (int i = 0; i < NumCams_6; ++i) delete arr_cams_[i];
    return 0;
}