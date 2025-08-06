from ea_controller.ea_controller import EA_Controller
import multiprocessing
from datahelpers.data import Data

e = EA_Controller()
d = Data()

ctx = multiprocessing.get_context('spawn')
queue = ctx.Queue()

e._run_single_ea_worker(queue, d.signal_objects[0], d.target_objects[0], batch_size=128)
