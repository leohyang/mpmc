#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <array>
#include <iostream>

struct FrameSlot {
    void* buf = nullptr;          // 2MB payload
    uint32_t bytes = 0;
    std::atomic<int> ref_count{0}; // A/B/C 三组计数：最后一个归还 slot
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

            // diff==0 表示 cell 处于 “空” 状态，可写
            intptr_t diff = (intptr_t)seq - (intptr_t)hpos;
            if (diff == 0) {
                // CAS 抢占一个 head slot
                if (head_.compare_exchange_weak(
                        hpos, hpos + 1,
                        std::memory_order_relaxed, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                // cell seq < h：队列满（full）
                return false;
            } else {
                // 其他线程推进了 head，更新 h 重试
                hpos = head_.load(std::memory_order_relaxed);
            }
        }

        // 写入 payload（这里是指针/对象本身）
        cell->data = data;    // if "data" has cv/mutext etc., this failed; need manual copy

        // 发布：把 seq 从 hpos 改成 hpos+1，表示 “populated”（可读）
        cell->seq.store(hpos + 1, std::memory_order_release);
        return true;
    }

    bool dequeue(T& out) {
        Cell* cell;
        size_t t = tail_.load(std::memory_order_relaxed);

        for (;;) {
            cell = &cells_[t & mask_];
            size_t seq = cell->seq.load(std::memory_order_acquire);

            // 对于 dequeue，期望 seq == t+1（表示已有数据）
            intptr_t diff = (intptr_t)seq - (intptr_t)(t + 1);
            if (diff == 0) {
                if (tail_.compare_exchange_weak(
                        t, t + 1,
                        std::memory_order_relaxed, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                // 队列空（empty）
                return false;
            } else {
                t = tail_.load(std::memory_order_relaxed);
            }
        }
        
        out = cell->data;  // 读出 data, as output/return

        // 释放 cell：seq 设置为 t + Pow2N，表示“空 taken-out”（下一轮可写）
        cell->seq.store(t + Pow2N, std::memory_order_release);
        return true;
    }

 private:
    static constexpr size_t mask_ = Pow2N - 1;

    struct Cell {
        std::atomic<size_t> seq; // sequence number
        T data;         // T is normally a ptr type
    };

    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    Cell cells_[Pow2N];  // descriptor-ringbuf
};


/* ============================================================
 * Per-camera lock-free pipeline
 * ============================================================
 */
template <size_t N>
struct CameraPipelineLF {
    static_assert((N & (N - 1)) == 0, "N must be power of 2");
    
    VyukovMPMC<FrameSlot*, N> free_pool_q; // free pool: 管理 slot 的可复用集合（类似 reserve/free）
    VyukovMPMC<FrameSlot*, N> qA, qB, qC; // 三个 group 的工作队列：fan-out, instantiated Vyukov 3 rings

    std::array<FrameSlot, N> slots;  // 实际 slot 存储（固定 N 个）  ||||||...|||

    explicit CameraPipelineLF(size_t bytes) {
        for (size_t i = 0; i < N; ++i) {
            size_t sz = ((bytes + 4095) / 4096) * 4096;  // 2MB = 1024*1024*2 = 4096 x 512pages
            slots[i].buf = std::aligned_alloc(4096, sz);
            slots[i].bytes = static_cast<uint32_t>(bytes);
            slots[i].ref_count.store(0, std::memory_order_relaxed);

            free_pool_q.enqueue(&slots[i]);   //  64 slots, 1 slot has 2MB
        }
    }

    ~CameraPipelineLF() {
        for (auto& s : slots) std::free(s.buf);
    }
};

/* ============================================================
 * main: 6 cameras + A/B/C groups
 * ============================================================
 */
int main() {
    constexpr int NumCams_6 = 6, NumGrps_3 = 3, FramesPerCam = 30;
    constexpr int TOTAL_FRAMES = NumCams_6 * FramesPerCam;

    // Worker configuration
    constexpr int A_workers = 6, B_workers = 3, C_workers = 2;

    // 每个 group 用于验证：总处理帧数（仅验证用，relaxed 足够）
    std::atomic<int> group_done[NumGrps_3];
    group_done[0].store(0);
    group_done[1].store(0);
    group_done[2].store(0);

    // 每个 worker 的处理计数（验证用） 6/3/2 v of atomic<int> counters
    std::vector<std::atomic<int>> v_a_(A_workers), v_b_(B_workers), v_c_(C_workers);
    for (auto& a : v_a_) a.store(0);
    for (auto& b : v_b_) b.store(0);
    for (auto& c : v_c_) c.store(0);

    // 6 cameras => 6 pipelines
    std::array<CameraPipelineLF<64>*, NumCams_6> arr_cams_{};
    for (int i = 0; i < NumCams_6; ++i) {
        arr_cams_[i] = new CameraPipelineLF<64>(2 * 1024 * 1024);
    }

    std::vector<std::thread> threads;

    /* ---------------- Producers ----------------
     * 每路 camera 一个 producer：
     * 1) 从 free_pool_q 取 slot
     * 2) 写 payload
     * 3) ref_count=3
     * 4) fan-out 到 A/B/C 三个队列
     */
    for (int id = 0; id < NumCams_6; ++id) {  // 6 cameras producing

        threads.emplace_back([&, id] {
            auto& P = *arr_cams_[id];   // P = Producer

            for (int i = 0; i < FramesPerCam; ++i) {
                FrameSlot* s = nullptr;

                // lock-free reserve：从 free pool 拿 slot
                while (!P.free_pool_q.dequeue(s)) { std::this_thread::yield(); }

                // 写 full payload（模拟 camera DMA 写）, populating 2MB
                std::memset(s->buf, id, s->bytes);

                // publish: 三个 group 都要处理一次
                s->ref_count.store(NumGrps_3, std::memory_order_release);

                // fan-out：同一个 slot* 推到 A/B/C
                // 注意：这三个 enqueue 必须最终成功，否则会丢帧      !!!
                while (!P.qA.enqueue(s)) std::this_thread::yield();
                while (!P.qB.enqueue(s)) std::this_thread::yield();
                while (!P.qC.enqueue(s)) std::this_thread::yield();
            }
        });
    }

    /* ---------------- Consumers ----------------
     * 每个 group 多 workers：
     * - 从所有 arr_cams_ 的对应队列中 try dequeue
     * - 成功则处理并 ref_count--
     * - 最后一个 group 把 slot 还回对应 camera 的 free_pool_q
     *
     * 轮询策略：round-robin 扫 6 个 cams（公平）
     */
    auto launch_group = [&](int gid, int workers, std::vector<std::atomic<int>>& workerCnt) {
        for (int w = 0; w < workers; ++w) {

            threads.emplace_back([&, gid, w] {
                int next_cam = 0;

                // group_done 是 group 总完成数；多线程共享
                while (group_done[gid].load(std::memory_order_relaxed) < TOTAL_FRAMES) {
                    FrameSlot* s = nullptr;
                    int picked_cam = -1;

                    // round-robin 扫描：每次从 next_cam 开始
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
                        // 没拿到活：yield 避免空转太猛
                        std::this_thread::yield();
                        continue;
                    }

                    // 模拟算法访问：
                    // A/B 真实情况会读 full 2MB；C 只读 header lines
                    // baseline 这里只 touch 一个 byte
                    volatile uint8_t v = static_cast<uint8_t*>(s->buf)[gid];
                    (void)v;

                    // ref_count--：最后一个负责归还 slot 到 free_pool
                    if (s->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        // 最后一个 group 完成：归还 slot 到对应 camera 的 pool
                        while (!arr_cams_[picked_cam]->free_pool_q.enqueue(s))
                            std::this_thread::yield();
                    }

                    // 验证计数：relaxed 足够（不参与同步决策）
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
