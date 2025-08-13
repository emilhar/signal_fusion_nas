from Globals import *
from datahelpers.data import Data
import multiprocessing
import os

class EA_Controller:
    def __init__(self):
        self.batch_size = 32
        self.max_workers = max(1, os.cpu_count() - 1)
        ctx = multiprocessing.get_context('spawn')
        self.evaluation_lock = ctx.Lock()

    def run_ea(self, target_to_update):
        if target_to_update.given_name not in Data.get_all_target_names():
            raise ValueError(f"Target does not exist: {target_to_update}")

        da = Data()
        signals = da.signal_objects
        
        processes = []
        queues = []
        results = [None] * len(signals)
        
        # Start all processes
        for i, signal in enumerate(signals):
            ctx = multiprocessing.get_context('spawn')
            queue = ctx.Queue()
            p = ctx.Process(
                target=self._run_single_ea_worker,
                args=(queue, signal, target_to_update, self.batch_size, self.evaluation_lock)
            )
            p.start()
            processes.append((i, p))
            queues.append((i, queue))
        
        # Collect results from queues
        for i, queue in queues:
            results[i] = queue.get()
        
        # Join all processes
        for i, p in processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"EA process failed for {signals[i].name}/{target_to_update.given_name}")
        
        return results

    @staticmethod
    def _run_single_ea_worker(queue, signal, target, batch_size, evaluation_lock):
        from ea_controller.optimizer import KernelSizeEvolutionaryOptimizer
        
        optimizer = KernelSizeEvolutionaryOptimizer(
            signal_type=signal.name,
            n_samples=signal.n_samples,
            classification_class=target,
            batch_size=batch_size,
        )

        optimizer.evaluation_lock = evaluation_lock # IMPORTANT! :)

        result = optimizer.run_evolution(part_of_bigger_run=True)
        
        queue.put((result, signal, target))
        
        del optimizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()