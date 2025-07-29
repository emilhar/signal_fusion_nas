import numpy as np

with np.load("Data/sleep-EDF-78/EEG_Fpz-Cz/SC4001E0.npz", allow_pickle=True) as f:
    a = []
    for key_ in f.files:
        a.append(str(f[key_][0]))

    print(", ".join(a))