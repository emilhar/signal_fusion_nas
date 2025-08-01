from Globals import *
from ea_controller.optimizer import KernelSizeEvolutionaryOptimizer
from datahelpers.data import Data
import multiprocessing

class EA_Controller:
    def __init__(self):
        self.batch_size = 32

    def run_ea(self, target_to_update):
        if target_to_update.given_name not in Data.get_all_target_names():
            raise ValueError(f"Target does not exist: {target_to_update}")

        da = Data()
        for signal in da.signal_objects:
            self._run_single_ea_in_process(signal, target_to_update)

    def _run_single_ea_in_process(self, signal, target):
        """Run EA optimization in a separate process to isolate memory"""
        ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()
        p = ctx.Process(
            target=self._run_single_ea_worker,
            args=(queue, signal, target, self.batch_size)
        )
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"EA process failed for {signal.name}/{target.given_name}")

    @staticmethod
    def _run_single_ea_worker(queue, signal, target, batch_size):
        """Worker function that runs in a separate process"""
        from ea_controller.optimizer import KernelSizeEvolutionaryOptimizer
        
        optimizer = KernelSizeEvolutionaryOptimizer(
            signal_type=signal.name,
            n_samples=signal.n_samples,
            classification_class=target,
            batch_size=batch_size,
        )
        optimizer.run_evolution(part_of_bigger_run=True)
        
        # Signal completion
        queue.put(True)
        
        # Explicit cleanup before process exit
        del optimizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()