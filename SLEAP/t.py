from Globals import Signal, Sleepstage, DataManager
from GridSearch.KRNL import QKernel_GridSearch, GridSearch

choices = [
    "train_loss",
    "test_loss",
    "precision",
    "recall",
    "accuracy",
    "branches",
    "best_f1",
    "best_auc",
    "time"
]


# for _ in range(9999999):
#     for signal in Signal.ALL_SIGNALS:
#         n_samples = 3000 if signal != Signal.EMG.SUBMENTAL else 30
#         qgrid = QKernel_GridSearch(signal, Sleepstage.N2, DataManager.DatasetNames.EDF_78, GridSearch._RunType.no_k0, 0.20, 1, n_samples)
#         qgrid.plot_qkernel_3d_vstime(metric="best_auc")
#         qgrid.plot_qkernel_slice_vstime(metric="best_auc", grid_steps=1000)


qgrid = QKernel_GridSearch(Signal.EEG.Pz_Oz, Sleepstage.N2, DataManager.DatasetNames.EDF_78, GridSearch._RunType.no_k0_full, 1.00, 10, 3000)
# qgrid.plot_qkernel_3d_vstime(metric="best_auc")
indi1 = qgrid.grid[6, 0, 0][0]["branches"][0]
indi2 = qgrid.grid[3, 3, 3][0]["branches"][0]
qgrid.head2head(indi1, indi1)


class EnSomniaC:
    def __init__(self):
        pass

fart = "EnSomnia-C"
fart = "Ensemble - Somna - Categorizer"
ensemble = EnSomniaC()

