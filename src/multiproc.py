import torch 
from tqdm import tqdm
from queue import Empty
from multiprocessing import get_context

class MultiProcessDispatch():
    def __init__(self, worker_func):
        self.worker_func = worker_func

    def initial_assertions(self):
        pass

    def worker_init_args(self):
        return ()
    
    def get_nbatch(self):
        return -1
    
    def process_result(self, res):
        pass
    
    def generate_work(self, idx):
        return None
    
    def postprocess_results(self):
        pass

    def num_processes(self):
        return 0
    
    def fast_work_gen(self):
        return False

    def fastest_gpu_idx(self):
        # If only have one work slice, assign it to the fastest GPU
        if torch.cuda.device_count() == 1:
            return 0
        return 1


    def work_q_len(self):
        if self.fast_work_gen():
            return 0
        else:
            return 50 * self.num_processes()

    def _kickoff_workers(self, ctx, work_q, res_q, work_done, offset, init_args, daemon, n_proc):
        # t = time()
        proc = [ctx.Process(target=self.worker_func, args=(work_q, res_q, work_done, i+offset, init_args), daemon=daemon) for i in range(n_proc)]
        for p in proc:
            p.start()
        # print(f"Time to start workers: {time() - t}", flush=True)
        # t = time()
        return proc

    def n_work_gen_check(self):
        return 10
    
    def n_work_gen_advance(self):
        return 10

    def run(self, daemon=False):
        self.initial_assertions()

        # Create work queue, kick off worker processes (one per GPU)
        n_proc = self.num_processes()
        assert n_proc > 0, "Invalid number of worker processes."
        offset = 0    
        if self.get_nbatch == 1:        # if only one batch, use the fastest GPU
            offset = self.fastest_gpu_idx()
            n_proc = 1
            
        ctx = get_context("spawn")
        res_q = ctx.Queue()
        work_q = ctx.Queue(maxsize=self.work_q_len())      # not unbounded, otherwise main process may load entire a matrix into memmory (in chunks)
        work_done = ctx.Event()
        init_args = self.worker_init_args()
        proc = self._kickoff_workers(ctx, work_q, res_q, work_done, offset, init_args, daemon, n_proc)

        # Parcel out work to workers, one batch at a time
        if not self.fast_work_gen():
            pbar = tqdm(total=self.get_nbatch())    
        n_work_left = 0
        for idx in tqdm(range(self.get_nbatch())):
            work = self.generate_work(idx)
            if isinstance(work, (list)):
                for w in work:
                    work_q.put(w, block=True, timeout=None)    
                n_work_left += len(work)
            else:
                work_q.put(work, block=True, timeout=None)
                n_work_left += 1

            # Read a limited number of items out of the result queue
            if idx % self.n_work_gen_check() == 0 and not self.fast_work_gen():
                idx = 0
                while not res_q.empty() and idx < self.n_work_gen_advance():
                    res = res_q.get(block=False)
                    self.process_result(res)
                    n_work_left -= 1
                    pbar.update(1)
                    idx += 1
        if self.fast_work_gen():
            pbar = tqdm(total=n_work_left)


        # Wait for remaining work to complete
        while n_work_left > 0:
            try:
                res = res_q.get(block=True, timeout=0.5)        # TODO(as) this shouldn't be needed, but sometimes it hangs at the end...
            except Empty:
                continue
            self.process_result(res)
            n_work_left -= 1 
            pbar.update(1)
        work_done.set()     # notify workers that no more work is coming
        pbar.close()

        # Clean up worker processes
        print("\tWaiting for workers to terminate...", flush=True)
        for p in proc:
            p.join()
            assert p.exitcode == 0, f"Worker process failed with error code: {p.exitcode}"

        self.postprocess_results()

