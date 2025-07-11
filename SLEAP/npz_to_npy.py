import numpy as np
import glob as fob

my_path = "Data/sleep-EDF-20/**/*.npz"

tot = len(fob.glob(my_path))
load_bar = ["|"] + ["*"] * tot + ["|"]

for i, file in enumerate(fob.glob(my_path)):
    load_bar[i+1] = "="
    with np.load(file) as numpy:
        x = numpy["x"]
        y = numpy["y"]

        file = file.split(".")[0]

        np.save(file + "_x", x)
        np.save(file + "_y", y)
    print("".join(load_bar), end="\r")

print("Done!")
