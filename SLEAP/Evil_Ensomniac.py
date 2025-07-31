from ensemble_controller.ensemble_controller import EnsembleController
from utils.trained_model_maker import TrainedModelMaker
from data.data_loader import SDataLoader
from Globals import Targets, Signal, LoggingTemplate, EvolutionManager, DataManager
import random
import os
import torch

training_epochs_for_models = 20
model_batch_size = 128
lr = 5e-5

EvolutionManager.VERY_VERBOSE = True
EvolutionManager.VERBOSE = True
DataManager.DATASET = DataManager.DatasetNames.EDF_78
def main():
    fog = EnsembleController()
    for x in [1, 2, 4, 8, 16, 32]:
        fog.NUM___FILTERS = x
        print(f"Creating models with {x} filters")
        # 👿 Evil Models Section 👿
        print("🔥🔥 MUHAHAH EVIL MODELS 🔥🔥")
        print("👹 Summoning the dark forces... 👹")
        get_random_indis(x)
        
        print("\n")
        
        # 😇 Good Models Section 😇
        print("✨✨ GOOD MODELS ✨✨")
        print("🕊️ Calling upon the righteous... 🕊️")
        get_good_indis(x)
        fog.create_ensemble(given_folder="EvilEnsomniacModels/EvilModels", model_marker="Evil")
        fog.create_ensemble(given_folder="EvilEnsomniacModels/GoodModels", model_marker="Good")



def save_model(tmm, signal_type, classification_class, prefix):
    os.makedirs(f"EvilEnsomniacModels/{prefix}Models", exist_ok=True)
    
    model_path = f"_misc/{prefix}Models/{classification_class}_{signal_type}_model.pt"
    torch.save({
        'state_dict': tmm.model_performance[LoggingTemplate.state_dict],
        'model_args': tmm.model_args
    }, model_path)

def model_exists(classification_class, signal_type, prefix):

    data_dir = f"_misc/{prefix}Models"
    if os.path.exists(data_dir):
        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
        for file_name in all_files:
            if classification_class in file_name and signal_type in file_name:
                return True
        
    return False

def get_random_indis(filters):
    for sig in Signal.ALL_SIGNALS:
        for st in Targets.All_CLASSES:
            if model_exists(st, sig, "Evil"):
                print(f"Model for {st} in {sig} already exists in EvilModels. Skipping...")
                print("Just kidding")
                
            print("\nTRAINING", st, "IN", sig)
            sdl = SDataLoader(sig, st, batch_size=model_batch_size)
            indi = []
            for _ in range(random.randint(1, 3)):
                indi.append([random.randint(2, 750) for _ in range(random.randint(2, 3))])
            for i, x in enumerate(indi):
                x = [x//(i*4) for i, x in enumerate(x, start=1)]
                x = [max(1, y) for y in x]
                indi[i] = x
            print("Branches: ", indi)
            m = TrainedModelMaker(
                branches=indi,
                N_SAMPLES=30 if sig==Signal.EMG.SUBMENTAL else 3000,
                pos_weight=sdl.pos_weight,
                train_loader=sdl.train_loader,
                test_loader=sdl.test_loader,
                epochs=training_epochs_for_models,
                batch_size=model_batch_size,
                filters=filters
            )
            save_model(m, sig, st, "Evil")

def get_good_indis(filters):
    for sig in Signal.ALL_SIGNALS:
        for st in Targets.All_CLASSES:
            if model_exists(st, sig, "Good"):
                print(f"Model for {st} in {sig} already exists in GoodModels. Skipping...")
                print("Just kidding")
                
            print("TRAINING", st, "IN", sig)
            sdl = SDataLoader(sig, st, batch_size=model_batch_size)
            if sig == Signal.EMG.SUBMENTAL:
                branch = [[20, 8, 8]]
            else:
                branch = [[400, 22, 22]]
            m = TrainedModelMaker(
                branches=branch,
                N_SAMPLES=30 if sig==Signal.EMG.SUBMENTAL else 3000,
                pos_weight=sdl.pos_weight,
                train_loader=sdl.train_loader,
                test_loader=sdl.test_loader,
                epochs=training_epochs_for_models,
                batch_size=model_batch_size,
                filters=filters
            )
            save_model(m, sig, st, "Good")

if __name__ == "__main__":
    main()