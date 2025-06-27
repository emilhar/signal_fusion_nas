from ModelController.TrainedModelMaker import TrainedModelMaker
from Globals import Sleepstage, Signal
from EAController.SleepDataLoader import SleepDataLoader
import datetime

def train_full(branches):

    SDL = SleepDataLoader(verbose=True, signal_type=Signal.EEG.Fpz_Cz, sleepstage=Sleepstage.WAKE, batch_size=128)
    individual_training_set, individual_test_set, n_samples, pos_weight = SDL.get_full_dataset()
    a = datetime.datetime.now()
    print(a)

    # Things marked with # come from the SDL
    new_model = TrainedModelMaker(
        branches = branches,
        name=f"{branches}, sleepstage: {Sleepstage.WAKE}, {128}batch, {20}epochs",
        sleepstage = Sleepstage.WAKE,
        signal_type=Signal.EEG.Fpz_Cz,
        batch_size= 128,
        train_loader = individual_training_set,
        test_loader = individual_test_set,
        epochs= 20,
        learning_rate=8e-5,
        verbose= True,
        N_SAMPLES= n_samples, #
        pos_weight= pos_weight,
        champion=True) #

    b = datetime.datetime.now()
    print(b)

    return new_model.model_performance

branches = []
options = [[1, 1, 1]]
for i in range(100, 1501, 100):
    options.append([i, i//2, i//4])

print(options)
for item in options:
    print("Training", item)
    train_full([item])