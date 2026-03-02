// SPDX-License-Identifier: GPL-2.0
/*
 *  SPSC to find out the kernel capacity on this platform 
 *
 *  Pure SPSC benchmark (no pool, no CAS): producers push encoded tokens (id+seq).
 *  Multi-consumer merge: mergers drain rings by shard (i % mergers == cid).
 *
 *
 * Build:
 *   obj-m += spsc_multi_ipc.o
 *   make -C /lib/modules/$(uname -r)/build M=$PWD modules
 *
 * Run:
 *   sudo insmod spsc_multi_ipc.ko rings=6 cap=8192 run_seconds=10 blocking=0 bind_threads=1 batch=32 mergers=1
     sudo rmmod spsc_multi_ipc
     sudo insmod spsc_multi_ipc.ko rings=6 cap=8192 run_seconds=10 blocking=0 bind_threads=1 batch=32 mergers=6
 *   
 */

#include <linux/cache.h>
#include <linux/compiler.h>
#include <linux/delay.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/module.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/types.h>
#include <linux/wait.h>
#include <linux/workqueue.h>
#include <linux/ktime.h>
#include <linux/atomic.h>

#define DRV_NAME "spsc_multi_bench"

/* -----------------------------
 * Params
 * ----------------------------- */

static unsigned int rings = 6;
module_param(rings, uint, 0644);
MODULE_PARM_DESC(rings, "number of SPSC rings (== number of producer threads)");

static unsigned int cap = 8192; /* must be power-of-two */
module_param(cap, uint, 0644);
MODULE_PARM_DESC(cap, "ring capacity (power-of-two)");

static unsigned int run_seconds = 10;
module_param(run_seconds, uint, 0644);
MODULE_PARM_DESC(run_seconds, "benchmark run time in seconds");

static bool blocking = false;
module_param(blocking, bool, 0644);
MODULE_PARM_DESC(blocking, "blocking mode (mergers sleep when all assigned rings empty)");

static bool bind_threads = true;
module_param(bind_threads, bool, 0644);
MODULE_PARM_DESC(bind_threads, "bind threads round-robin to online CPUs");

static bool yield_enable = true;
module_param(yield_enable, bool, 0644);
MODULE_PARM_DESC(yield_enable, "enable periodic cond_resched()");

static unsigned int batch = 32;
module_param(batch, uint, 0644);
MODULE_PARM_DESC(batch, "batch pop per ring scan (suggest 16/32/64)");

static unsigned int mergers = 1;
module_param(mergers, uint, 0644);
MODULE_PARM_DESC(mergers, "number of merger consumer threads (C step)");

/* Validation knobs */
static bool validate_seq = true;
module_param(validate_seq, bool, 0644);
MODULE_PARM_DESC(validate_seq, "validate per-ring seq increments (expect last+1); disable for pure throughput");

static bool validate_pid = true;
module_param(validate_pid, bool, 0644);
MODULE_PARM_DESC(validate_pid, "validate decoded pid matches ring index");

static bool clear_on_pop = false;
module_param(clear_on_pop, bool, 0644);
MODULE_PARM_DESC(clear_on_pop, "DEBUG: clear ring slot to NULL on pop (slower, helps detect stale reads)");

/* backoff knobs */
static unsigned int backoff_spin_loops = 64;
module_param(backoff_spin_loops, uint, 0644);

static unsigned int backoff_yield_every = 256;
module_param(backoff_yield_every, uint, 0644);

static unsigned int backoff_sleep_us_min = 0;
module_param(backoff_sleep_us_min, uint, 0644);
static unsigned int backoff_sleep_us_max = 0;
module_param(backoff_sleep_us_max, uint, 0644);

/* -----------------------------
 * Helpers
 * ----------------------------- */

static inline bool is_pow2_u32(unsigned int x) {	return x && ((x & (x - 1)) == 0); }

static inline u64 ns_to_ops_per_s(u64 ops, s64 dt_ns)
{
	if (dt_ns <= 0)		return 0;
	return div64_u64(ops * (u64)NSEC_PER_SEC, (u64)dt_ns);
}

struct backoff_state {	u32 fails;   };

static inline void backoff_fail(struct backoff_state *b)
{
	u32 i;
	b->fails++;

	for (i = 0; i < backoff_spin_loops; i++)		cpu_relax();

	if (yield_enable && backoff_yield_every &&	unlikely((b->fails % backoff_yield_every) == 0))
		cond_resched();

	if (backoff_sleep_us_min && backoff_sleep_us_max &&
	    backoff_sleep_us_max >= backoff_sleep_us_min &&
	    unlikely((b->fails % (backoff_yield_every ? backoff_yield_every : 512)) == 0))
		usleep_range(backoff_sleep_us_min, backoff_sleep_us_max);
}

static inline void backoff_reset(struct backoff_state *b) {	b->fails = 0; }

static inline void maybe_yield(u32 *iters)
{
	if (!yield_enable)		return;
	if (unlikely(((++(*iters)) & 0x3FF) == 0))		cond_resched();
}

/* -----------------------------
 * Token encoding (NO pool, NO pointer deref)
 * Keep token within low 48-bit so it looks like a canonical low VA.
 * Layout:             never  *deref token
 *   bits [47:32] = producer id (0..65535)  高16位：producer ID (0-65535)
 *   bits [31:0]  = seq (wraps at 2^32)     低32位：序列号
 *
 * Store token as an opaque pointer, never dereference.
 * ----------------------------- */

static __always_inline unsigned long make_token(u32 prod_id, u32 seq)
{
	u64 t = (((u64)prod_id & 0xFFFFULL) << 32) | ((u64)seq);
	/* ensure non-NULL */
	if (t == 0)		t = 1;
	return (unsigned long)t;
}

static __always_inline u32 token_prod_id(unsigned long t) {	return (u32)((t >> 32) & 0xFFFFUL); }

static __always_inline u32 token_seq(unsigned long t) {	return (u32)(t & 0xFFFFFFFFUL); }

/* -----------------------------
 * SPSC ring (void* payload) 无CAS操作：利用SPSC特性，不需要复杂的原子操作
 * ----------------------------- */

struct spsc_ring {  // SPSC环（无锁队列） 防止head和tail在同一个cache line，避免false sharing
    void **buf;        // ring buf's ptr-array
    u32 cap;           // 容量（2的幂）
    u32 mask;          // cap - 1，用于快速取模
	u32 head ____cacheline_aligned_in_smp; /* producer writes */
	u32 tail ____cacheline_aligned_in_smp; /* consumer writes */
};

static int spsc_init(struct spsc_ring *r, u32 cap0, gfp_t gfp)
{
	u32 i;
	if (!r || !is_pow2_u32(cap0))		return -EINVAL;
		r->buf = kmalloc_array(cap0, sizeof(r->buf[0]), gfp);	if (!r->buf)	return -ENOMEM;
		r->cap = cap0;
		r->mask = cap0 - 1;
		r->head = 0;
		r->tail = 0;
		for (i = 0; i < cap0; i++)		r->buf[i] = NULL;
	return 0;
}

static void spsc_destroy(struct spsc_ring *r)
{
	if (!r)		return;
	kfree(r->buf);
	r->buf = NULL;
	r->cap = 0;
	r->mask = 0;
	r->head = r->tail = 0;
}

static __always_inline bool spsc_try_enqueue(struct spsc_ring *r, void *p)
{
	u32 head, next, tail;

	head = READ_ONCE(r->head);
	next = head + 1;

	tail = smp_load_acquire((u32 *)&r->tail);
	if (unlikely((next - tail) > r->cap))		return false;

	WRITE_ONCE(r->buf[head & r->mask], p);
	smp_store_release((u32 *)&r->head, next);
	return true;
}

static __always_inline bool spsc_try_dequeue(struct spsc_ring *r, void **out)
{
	u32 tail, head, idx;
	void *p;

	tail = READ_ONCE(r->tail);
	head = smp_load_acquire((u32 *)&r->head);

	if (unlikely(tail == head))		return false;

	idx = tail & r->mask;
	p = READ_ONCE(r->buf[idx]);

	if (clear_on_pop)		WRITE_ONCE(r->buf[idx], NULL);

	*out = p;
	smp_store_release((u32 *)&r->tail, tail + 1);
	return true;
}

static __always_inline u32 spsc_dequeue_batch(struct spsc_ring *r, void **arr, u32 n)
{
	u32 popped = 0;
	while (popped < n) {
		if (!spsc_try_dequeue(r, &arr[popped]))
			break;
		popped++;
	}
	return popped;
}

/* -----------------------------
 * Stats / contexts
 * ----------------------------- */

struct prod_stats {
	u64 enq_ok;
	u64 enq_full;
};

struct cons_stats {
	u64 deq_ok;
	u64 empty_loops;
	u64 bad_token;
	u64 drained_after_stop;
};

struct producer_ctx {
	u32 id;
	struct task_struct *task;
	struct spsc_ring *ring;

	u32 seq;
	struct backoff_state bo;
	struct prod_stats st;
};

struct consumer_ctx {
	u32 id;
	struct task_struct *task;

	struct spsc_ring *rings;
	u32 nrings;

	u32 batch;
	struct backoff_state bo;
	struct cons_stats st;

	/* Per-ring last seq (only for rings assigned to this consumer); size = nrings */
	u32 *last_seq_by_ring;
};

static struct spsc_ring *g_rings;
static struct producer_ctx *g_prod;
static struct consumer_ctx *g_cons; /* array of mergers */

static DECLARE_WAIT_QUEUE_HEAD(g_not_empty_wq);

static atomic_t g_stopping = ATOMIC_INIT(0);
static ktime_t g_t0;
static struct delayed_work g_stop_work;

/* -----------------------------
 * CPU binding
 * ----------------------------- */

static void bind_to_cpu(struct task_struct *t, int idx)
{
	int ncpus;
	if (!bind_threads || !t)		return;

	ncpus = num_online_cpus();	if (ncpus <= 0)		return;

	kthread_bind(t, idx % ncpus);
}

/* -----------------------------
 * Thread bodies
 * ----------------------------- */

static int producer_fn(void *arg)
{
	struct producer_ctx *p = arg;
	u32 iters = 0;

	while (!kthread_should_stop() && !atomic_read(&g_stopping)) {
		// unsigned long tok = make_token(p->id, p->seq++);
        unsigned long tok = make_token(p->id, p->seq); // 把 seq++ 改成 “成功 push 才递增”

		if (spsc_try_enqueue(p->ring, (void *)tok)) {
			p->st.enq_ok++;    	// enqueued successfully
            p->seq++;   		//  counting only if really enqueued
			if (blocking)
				wake_up_interruptible(&g_not_empty_wq);
			backoff_reset(&p->bo);
		} else {
			p->st.enq_full++;		// cnt for the full case
			backoff_fail(&p->bo);	// wait a bit
		}

		maybe_yield(&iters);
	}

	return 0;
}

/* Check if all rings assigned to this consumer are empty */
static bool assigned_rings_empty(struct consumer_ctx *c)
{
	u32 i;
	for (i = c->id; i < c->nrings; i += mergers) {
		u32 tail = READ_ONCE(c->rings[i].tail);
		u32 head = smp_load_acquire((u32 *)&c->rings[i].head);
		if (tail != head)	return false;
	}
	return true;
}

static int consumer_fn(void *arg)
{
	struct consumer_ctx *c = arg;
	void *local[256];		// hard limit, batch <= 256
	u32 maxbatch = min_t(u32, c->batch, (u32)ARRAY_SIZE(local));
	u32 iters = 0;

	while (!kthread_should_stop() && !atomic_read(&g_stopping)) {
		u32 i;
		u32 total_popped = 0;

		for (i = c->id; i < c->nrings; i += mergers) {
			u32 n, j;

			n = spsc_dequeue_batch(&c->rings[i], local, maxbatch);
			total_popped += n;

			for (j = 0; j < n; j++) {
				unsigned long tok = (unsigned long)local[j];

				if (validate_pid) {
					u32 pid = token_prod_id(tok);
					if (unlikely(pid != i))		c->st.bad_token++;
				}

				if (validate_seq) {
					u32 seq = token_seq(tok);
					u32 last = READ_ONCE(c->last_seq_by_ring[i]);
					/* first observation: last == 0xFFFFFFFF */
					if (likely(last == 0xFFFFFFFFU || seq == last + 1)) {
						WRITE_ONCE(c->last_seq_by_ring[i], seq);
					} else {
						c->st.bad_token++;
						WRITE_ONCE(c->last_seq_by_ring[i], seq);
					}
				}

				c->st.deq_ok++;
			}
		}

		if (total_popped == 0) {
			c->st.empty_loops++;

			if (blocking) {
				wait_event_interruptible_timeout(
					g_not_empty_wq,
					kthread_should_stop() ||	atomic_read(&g_stopping) ||	!assigned_rings_empty(c),
					msecs_to_jiffies(10));
			} else {
				backoff_fail(&c->bo);
			}
		} else {
			backoff_reset(&c->bo);
		}

		maybe_yield(&iters);
	}

	return 0;
}

/* -----------------------------
 * Stop/drain/report
 * ----------------------------- */

static void stop_all(void)
{
	u32 i;

	if (atomic_xchg(&g_stopping, 1))		return;

	if (g_prod) {
		for (i = 0; i < rings; i++)
			if (g_prod[i].task)
				kthread_stop(g_prod[i].task);
	}

	if (g_cons) {
		for (i = 0; i < mergers; i++)
			if (g_cons[i].task)
				kthread_stop(g_cons[i].task);
	}
}

static void drain_all(void)
{
	u32 i;
	void *tmp;

	if (!g_cons)		return;

	for (i = 0; i < rings; i++) {
		while (spsc_try_dequeue(&g_rings[i], &tmp)) {
			g_cons[0].st.drained_after_stop++;
		}
	}
}

static void print_report(const char *tag, s64 dt_ns)
{
	u64 enq_ok = 0, enq_full = 0;
	u64 deq_ok = 0, empty_loops = 0, bad = 0, drained = 0;
	u32 i;

	for (i = 0; i < rings; i++) {
		enq_ok += g_prod[i].st.enq_ok;
		enq_full += g_prod[i].st.enq_full;
	}

	drain_all();

	for (i = 0; i < mergers; i++) {
		deq_ok += g_cons[i].st.deq_ok;
		empty_loops += g_cons[i].st.empty_loops;
		bad += g_cons[i].st.bad_token;
		drained += g_cons[i].st.drained_after_stop;
	}

	pr_info(DRV_NAME "[%s]: run=%lld ns (~%llu s), blocking=%d bind=%d rings=%u cap=%u batch=%u mergers=%u validate_pid=%d validate_seq=%d clear_on_pop=%d\n",
		tag, dt_ns, div64_u64((u64)dt_ns, (u64)NSEC_PER_SEC),
		blocking, bind_threads, rings, cap, batch, mergers, validate_pid, validate_seq, clear_on_pop);

	pr_info("  enq_ok=%llu (ops/s=%llu) enq_full=%llu\n",
		enq_ok, ns_to_ops_per_s(enq_ok, dt_ns), enq_full);

	pr_info("  deq_ok=%llu (ops/s=%llu) empty_loops=%llu bad_token=%llu\n",
		deq_ok, ns_to_ops_per_s(deq_ok, dt_ns), empty_loops, bad);

	if (drained)
		pr_info("  drained_after_stop=%llu\n", drained);
}

static void stop_work_fn(struct work_struct *wk)
{
	ktime_t t1;
	s64 dt_ns;

	stop_all();

	t1 = ktime_get();
	dt_ns = ktime_to_ns(ktime_sub(t1, g_t0));

	print_report("final", dt_ns);
}

/* -----------------------------
 * Module init/exit
 * ----------------------------- */

static int __init spsc_multi_init(void)
{
	int ret = 0;
	u32 i;

	if (!is_pow2_u32(cap)) { pr_err(DRV_NAME ": cap must be power-of-two, got %u\n", cap);	return -EINVAL;	}
	if (rings == 0) {	pr_err(DRV_NAME ": rings must be >0\n");	return -EINVAL;	}
	if (batch == 0) {	pr_err(DRV_NAME ": batch must be >0\n");	return -EINVAL;	}
	if (mergers == 0) {	pr_err(DRV_NAME ": mergers must be >0\n");  return -EINVAL;	}
	if (mergers > rings)		mergers = rings;

	atomic_set(&g_stopping, 0);

	g_rings = kcalloc(rings, sizeof(*g_rings), GFP_KERNEL);	if (!g_rings) return -ENOMEM;

	for (i = 0; i < rings; i++) {
		ret = spsc_init(&g_rings[i], cap, GFP_KERNEL);
		if (ret) {pr_err(DRV_NAME ": spsc_init ring %u failed: %d\n", i, ret);	goto out_free_rings;	}
	}

	g_prod = kcalloc(rings, sizeof(*g_prod), GFP_KERNEL);	if (!g_prod) {	ret = -ENOMEM;	goto out_free_rings;	}

	g_cons = kcalloc(mergers, sizeof(*g_cons), GFP_KERNEL);	if (!g_cons) {	ret = -ENOMEM;	goto out_free_prod;	}

	/* Allocate per-consumer per-ring last_seq array (full size for simplicity) */
	for (i = 0; i < mergers; i++) {
		g_cons[i].id = i;
		g_cons[i].rings = g_rings;
		g_cons[i].nrings = rings;
		g_cons[i].batch = batch;

		g_cons[i].last_seq_by_ring = kmalloc_array(rings, sizeof(u32), GFP_KERNEL);
		if (!g_cons[i].last_seq_by_ring) {		ret = -ENOMEM;	goto out_free_cons_state;	}
		memset(g_cons[i].last_seq_by_ring, 0xFF, rings * sizeof(u32));
	}

	/* Create consumer threads first */
	for (i = 0; i < mergers; i++) {
		g_cons[i].task = kthread_create(consumer_fn, &g_cons[i], DRV_NAME "_cons/%u", i);
		if (IS_ERR(g_cons[i].task)) {	ret = PTR_ERR(g_cons[i].task);	g_cons[i].task = NULL; goto out_stop; }
		bind_to_cpu(g_cons[i].task, (int)(rings + i));
		wake_up_process(g_cons[i].task);
	}

	/* Create producer threads */
	for (i = 0; i < rings; i++) {
		g_prod[i].id = i;
		g_prod[i].ring = &g_rings[i];
		g_prod[i].seq = 0;

		g_prod[i].task = kthread_create(producer_fn, &g_prod[i], DRV_NAME "_prod/%u", i);
		if (IS_ERR(g_prod[i].task)) {	ret = PTR_ERR(g_prod[i].task);	g_prod[i].task = NULL;	goto out_stop;	}
		bind_to_cpu(g_prod[i].task, (int)i);
		wake_up_process(g_prod[i].task);
	}

	g_t0 = ktime_get();
	INIT_DELAYED_WORK(&g_stop_work, stop_work_fn);
	schedule_delayed_work(&g_stop_work, msecs_to_jiffies(run_seconds * 1000));

	pr_info(DRV_NAME ": started. run_seconds=%u blocking=%d bind=%d rings=%u cap=%u batch=%u mergers=%u validate_pid=%d validate_seq=%d clear_on_pop=%d\n",
		run_seconds, blocking, bind_threads, rings, cap, batch, mergers, validate_pid, validate_seq, clear_on_pop);

	return 0;

out_stop:
	stop_all();
	cancel_delayed_work_sync(&g_stop_work);

out_free_cons_state:
	if (g_cons) {
		for (i = 0; i < mergers; i++) {
			kfree(g_cons[i].last_seq_by_ring);
			g_cons[i].last_seq_by_ring = NULL;
		}
		kfree(g_cons);
		g_cons = NULL;
	}

out_free_prod:
	kfree(g_prod);
	g_prod = NULL;

out_free_rings:
	if (g_rings) {
		for (i = 0; i < rings; i++)
			spsc_destroy(&g_rings[i]);
		kfree(g_rings);
		g_rings = NULL;
	}
	return ret;
}

static void __exit spsc_multi_exit(void)
{
	ktime_t t1;
	s64 dt_ns;
	u32 i;

	cancel_delayed_work_sync(&g_stop_work);

	stop_all();

	t1 = ktime_get();
	dt_ns = ktime_to_ns(ktime_sub(t1, g_t0));
	print_report("exit", dt_ns);

	if (g_cons) {
		for (i = 0; i < mergers; i++) {
			kfree(g_cons[i].last_seq_by_ring);
			g_cons[i].last_seq_by_ring = NULL;
		}
		kfree(g_cons);
		g_cons = NULL;
	}

	kfree(g_prod);
	g_prod = NULL;

	if (g_rings) {
		for (i = 0; i < rings; i++)
			spsc_destroy(&g_rings[i]);
		kfree(g_rings);
		g_rings = NULL;
	}

	pr_info(DRV_NAME ": exit\n");
}
module_init(spsc_multi_init);
module_exit(spsc_multi_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Lianghua Yang");
MODULE_DESCRIPTION("multi-ring SPSC benchmark (no pool/CAS) + multi-merger consumers; per-ring validation");

/* ---------- on Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz based Ubuntu22.04 benchmark testing, 12 cores ----------------229M/s
make
sudo insmod spsc_multi_ipc.ko rings=6 cap=8192 run_seconds=10 blocking=0 bind_threads=1 batch=32 mergers=6
sudo dmesg | tail -n 10
[396640.693739] spsc_multi_ipc: loading out-of-tree module taints kernel.
[396640.693749] spsc_multi_ipc: module verification failed: signature and/or required key missing - tainting kernel
[396640.780702] spsc_multi_bench: started. run_seconds=10 blocking=0 bind=1 rings=6 cap=8192 batch=32 mergers=6 validate_pid=1 validate_seq=1 clear_on_pop=0
[396650.872441] spsc_multi_bench[final]: run=10091820954 ns (~10 s), blocking=0 bind=1 rings=6 cap=8192 batch=32 mergers=6 validate_pid=1 validate_seq=1 clear_on_pop=0
[396650.872447]   enq_ok=2317789572 (ops/s=229670104) enq_full=1104973
[396650.872449]   deq_ok=2317788921 (ops/s=229670039) empty_loops=14921561 bad_token=1
[396650.872450]   drained_after_stop=651


-------------- on Orin AGX  Cortex-A78AE,  12 cores  -------------------25.9M/s much slower than x86
	[602853.794515] spsc_multi_bench: started. run_seconds=10 blocking=0 bind=1 rings=6 cap=8192 batch=32 mergers=6 validate_pid=1 validate_seq=1 clear_on_pop=0
	[602863.866785] spsc_multi_bench[final]: run=10072414171 ns (~10 s), blocking=0 bind=1 rings=6 cap=8192 batch=32 mergers=6 validate_pid=1 validate_seq=1 clear_on_pop=0
	[602863.866798]   enq_ok=261334502 (ops/s=25945567) enq_full=1051953
	[602863.866800]   deq_ok=261331328 (ops/s=25945252) empty_loops=20381386 bad_token=1
	[602863.866802]   drained_after_stop=3174
*/
