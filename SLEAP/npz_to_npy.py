import numpy as np
import glob as fob

for file in fob.glob("Data/sleep-EDF-78/EEG_Fpz-Cz/*.npz"):
    with np.load(file) as numpy:
        x = numpy["x"]
        y = numpy["y"]

        np.save(file + "_x", x)
        np.save(file + "_y", y)

        int = print
        int("done!")