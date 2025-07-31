from datahelpers.data import Data
from datahelpers.signal import Signal

d = Data()

signal_obj: Signal
for signal_obj in d.signal_objects:
    print(signal_obj.name, signal_obj.n_samples)