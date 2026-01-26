/*   ThreadPool with MPMC lock-free support 
 *   Date :  2026.01.25 v0.1  leohyang@gmail.com
 *   Build:   g++ -o ~/.bin/mpmc_threadpool mpmc_threadpool_drain-vs-discard.cpp -std=c++20 -lpthread
 *            ~/.bin/mpmc_threadpool
 */
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <chrono>
#include <syncstream>

using namespace std;

/* ==========  1) Vyukov bounded MPMC queue (pointer payload recommended)
    Correct signed dif (intptr_t).  ========== */
template <typename T>           //  expecting T as ptr style
class MPMCQueue {
 private:
    struct Cell { //  alignas (64) reduce cache false sharing vs. more cache hit
        std::atomic<size_t> seq{0};
        T value;
        Cell() = default; 
        Cell(const Cell &) = delete;
        Cell & operator =(const Cell &) = delete;
        Cell (const Cell &&) = delete;
        Cell && operator =(const Cell &&) = delete;
    };

    const size_t cap_;
    const size_t mask_;
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    std::unique_ptr<Cell[]> buffer_;  // smart pointer, old raw pointer, both worked
    //Cell* buffer_;        // #DND   --->[][][][]...[]    vs. std::vector<Cell> buffer_;

 public:
    explicit MPMCQueue(size_t len)          // long unsigned int for size_t
        : cap_(npower2(len)), mask_(cap_ - 1),
          head_(0), tail_(0) {

        // buffer_ = static_cast<Cell*>(::operator new[](cap_ * sizeof(Cell))); // #DND  raw alloc
        buffer_ = std::make_unique<Cell[]>(cap_); // easier to implement/understand, yet cache hit is BAD !!!
        for (size_t i = 0; i < cap_; ++i) {
            // new (&buffer_[i]) Cell();    // #DND    ctor for each item
            buffer_[i].seq.store(i, std::memory_order_relaxed);
        }
    }

    // ~MPMCQueue() {       // #DND
    //     for (size_t i = 0; i < cap_; ++i) buffer_[i].~Cell(); // dtor for each obj
    //     ::operator delete[](buffer_);            // free the entire raw memory
    // }
    ~MPMCQueue() = default; // unique_ptr auto deletes

    MPMCQueue(const MPMCQueue&) = delete;
    MPMCQueue& operator=(const MPMCQueue&) = delete;

    bool enqueue(T&& v) {
        size_t pos = tail_.load(std::memory_order_relaxed);
        for (;;) {
            // Cell* c = &buffer_[pos & mask_]; // works both unique_ptr<Cell[]>, Cell *
            // size_t seq = c->seq.load(std::memory_order_acquire);
            Cell & c = buffer_[pos & mask_];
            size_t seq = c.seq.load(std::memory_order_acquire);
            long int dif = (long int)seq - (long int)pos;   // unsigned long 2 long int, any loss?

            if (dif == 0) {
                if (tail_.compare_exchange_weak(
                        pos, pos + 1,
                        std::memory_order_relaxed,
                        std::memory_order_relaxed)) {
                    // c->value = std::move(v);
                    // c->seq.store(pos + 1, std::memory_order_release);
                    c.value = std::move(v);
                    c.seq.store(pos + 1, std::memory_order_release);
                    return true;
                }
            } else if (dif < 0) {
                return false; // full
            } else {
                pos = tail_.load(std::memory_order_relaxed);
            }
        }
    }

    bool dequeue(T& out) {
        size_t pos = head_.load(std::memory_order_relaxed);
        for (;;) {
            Cell* c = &buffer_[pos & mask_]; // works both unique_ptr<Cell[]>, Cell *
            size_t seq = c->seq.load(std::memory_order_acquire);
            long int dif = (long int)seq - (long int)(pos + 1);

            if (dif == 0) {
                if (head_.compare_exchange_weak(
                        pos, pos + 1,
                        std::memory_order_relaxed,
                        std::memory_order_relaxed)) {
                    out = std::move(c->value);
                    c->seq.store(pos + cap_, std::memory_order_release);
                    return true;
                }
            } else if (dif < 0) {
                return false; // empty
            } else {
                pos = head_.load(std::memory_order_relaxed);
            }
        }
    }

 private:
    static size_t npower2(size_t x) {
        if (x < 2) return 2;
        size_t r = 1;
        while (r < x) r <<= 1;
        return r;
    }
};

/* ========== 2) ThreadPool with shutdown choices ======== */
class ThreadPool {
 public: 
    enum class ShutdownMode {
        Drain,   // A: execute all successfully accepted tasks
        StopNow  // B: stop ASAP, cancel/discard queued-but-not-run tasks
    };

    struct Configs {
        size_t num_threads = std::thread::hardware_concurrency();
        size_t queue_capacity = 1024;
        uint32_t spin_before_sleep = 8; // hybrid spin-then-sleep
    };

 private:    
    // ------------ Internal types ------------
    enum class State : uint8_t { Running,  StoppingDrain, StoppingNow,  Stopped };

    struct TaskNode {
        std::function<void()> run;
        std::function<void()> cancel;
    };

    Configs cfg_;
    MPMCQueue<TaskNode*> q_taskptr_;    
    std::vector<std::thread> v_workers_;

    std::atomic<State> atm_sta_;        //  atomic<uint8_t> 
    std::atomic<size_t> atm_pending_;

    std::condition_variable cv_;    
    std::mutex mtx_;                // only for the cv_

 public:
    explicit ThreadPool(const Configs& cfg) : cfg_(cfg), q_taskptr_(cfg.queue_capacity), 
               atm_sta_(State::Running),    atm_pending_(0) {

        if (cfg_.num_threads == 0) cfg_.num_threads = 1;
        v_workers_.reserve(cfg_.num_threads);
        for (size_t i = 0; i < cfg_.num_threads; ++i) {
            v_workers_.emplace_back([this, i] { worker_loop(i); });
        }
    }

    ~ThreadPool() {
        shutdown(ShutdownMode::Drain);
        join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void shutdown(ShutdownMode mode) {
        State target = (mode == ShutdownMode::Drain) ? 
                            State::StoppingDrain : State::StoppingNow;
        State s = atm_sta_.load(std::memory_order_acquire);

        for (;;) {
            if (s == State::Stopped || s == State::StoppingDrain || s == State::StoppingNow) break;
            if (atm_sta_.compare_exchange_weak(s, target,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
                break;
            }
        }
        
        cv_.notify_all();  // wake up all workers to drain/exit
    }

    void join() {
        for (auto& t : v_workers_) {
            if (t.joinable()) t.join();
        }
        v_workers_.clear();

        if (atm_sta_.load(std::memory_order_acquire) == State::StoppingNow) {
            // cancel remaining queued_tasks
            TaskNode* node = nullptr;
            while (q_taskptr_.dequeue(node)) {                
                node->cancel();
                delete node;
                atm_pending_.fetch_sub(1, std::memory_order_acq_rel);
            }          // the discarded task shall decrease pending-counter
            cv_.notify_all();
        }

        atm_sta_.store(State::Stopped, std::memory_order_release);
    }

    template<class F, class... Args>
    bool post(F&& f, Args&&... args) {
        auto fn = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
        TaskNode* node = new TaskNode;
            node->run = [fn = std::move(fn)]() mutable {  fn();  };
            node->cancel = []() {}; // no-op
        return enqueue_node(node);
    }

 private:
    // ------------ Internal helpers ------------

    bool enqueue_node(TaskNode* node) {
        if (atm_sta_.load(std::memory_order_acquire) != State::Running) {
            return false;   // reject all new tasks if not Running
        }

        atm_pending_.fetch_add(1, std::memory_order_acq_rel);

        TaskNode* p = node;
        for (;;) {
            State s = atm_sta_.load(std::memory_order_acquire);
            if (s != State::Running) {
                // 还没成功 publish，就允许失败并 rollback
                atm_pending_.fetch_sub(1, std::memory_order_acq_rel);
                cv_.notify_all();
                return false;
            }

            if (q_taskptr_.enqueue(std::move(p))) {
                cv_.notify_one();
                return true;
            }

            std::this_thread::yield();
        }
    }

    void worker_loop(size_t /*idx*/) {
        TaskNode* node = nullptr;
        uint32_t spins = 0;

        for (;;) {
            if (q_taskptr_.dequeue(node)) {
                spins = 0;

                try {
                    node->run();
                } catch (...) {
                    // exception caught/ignore;  this worker won't terminate
                }
                delete node;

                auto prev = atm_pending_.fetch_sub(1, std::memory_order_acq_rel);
                (void)prev;

                if (atm_sta_.load(std::memory_order_acquire) == State::StoppingDrain &&
                    atm_pending_.load(std::memory_order_acquire) == 0) {
                    cv_.notify_all();
                }
                continue;
            }

            // queue empty
            State s = atm_sta_.load(std::memory_order_acquire);

            if (s == State::StoppingNow) {                
                return; // StopNow：empty task-queue, done. already running tasks by join() cancel
            }
            if (s == State::StoppingDrain) {  // Drain： pending==0 all finished
                if (atm_pending_.load(std::memory_order_acquire) == 0) {
                    std::osyncstream(std::cout) << "all accepted tasks before close-time are served\n";
                    return;
                }
            }

            if (++spins <= cfg_.spin_before_sleep) {
                std::this_thread::yield();  // running-or-draining tasks, spin n rounds first
                continue;
            }
            spins = 0;
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [&] {              // + then sleep  = hybrid combo
                State ss = atm_sta_.load(std::memory_order_acquire);
                return ss != State::Running ||
                       atm_pending_.load(std::memory_order_acquire) > 0;
            });
        }
    }

};

// === 3) test app ===
int main() {
    ThreadPool::Configs cfg;
        cfg.num_threads = 16;
        cfg.queue_capacity = 1024;
        cfg.spin_before_sleep = 3;
    auto pool = std::make_unique<ThreadPool>(cfg);  // ThreadPool pool(cfg); ok mostly

    std::atomic<uint64_t> cnt{0};
    constexpr size_t kProd = 32, kJobs = 1000;

    std::vector<std::thread> v_producers;
    v_producers.reserve(kProd);

    for (size_t i = 0; i < kProd; ++i) {        // xx Producers, nestled lambdas
        v_producers.emplace_back([&, i] {
            for (size_t j = 0; j < kJobs; ++j) {    // yy Tasks each producer
                while (!pool->post([&cnt, i, j] { 
                        if ((j%100) == 99) std::osyncstream(std::cout) << " producer " << i << "'s #"<< j << " job done\n";
                        cnt.fetch_add(1, std::memory_order_relaxed);
                        })
                ) {
                    std::this_thread::yield();
                }
            }
        });
    }

    for (auto& t : v_producers) t.join();

    uint64_t expected = kProd * kJobs;
    while (cnt.load(std::memory_order_relaxed) < expected) {
        std::this_thread::yield();
    }

    // drain+join : default dtor will auto-do, explicit manual invoking to check
    pool->shutdown(ThreadPool::ShutdownMode::Drain); // Drain test1; StopNow test2
    pool->join();

    std::osyncstream(std::cout) << "post=" << expected << " executed=" << cnt.load() << "\n";
    std::osyncstream(std::cout) << "PASSED\n";
    return 0;
}