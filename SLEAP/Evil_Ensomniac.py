from ModelController.EnSomniaC.EnSomniaC_Controller import superMain
from ea_controller.trained_model_maker import TrainedModelMaker
from EAController.data_loader import SDataLoader
from Globals import Classes, Signal, ModelManager, LoggingTemplate
import random
import os
import torch

def main():
    # 👿 Evil Models Section 👿
    print("🔥🔥 MUHAHAH EVIL MODELS 🔥🔥")
    print("👹 Summoning the dark forces... 👹")
    get_random_indis()
    superMain(given_folder="SavedModels/EvilModels")
    
    print("\n")
    
    # 😇 Good Models Section 😇
    print("✨✨ GOOD MODELS ✨✨")
    print("🕊️ Calling upon the righteous... 🕊️")
    get_good_indis()
    superMain(given_folder="SavedModels/GoodModels")

def save_model(tmm, signal_type, classification_class, prefix):
    os.makedirs(f"SavedModels/{prefix}Models", exist_ok=True)
    
    model_path = f"SavedModels/{prefix}Models/{classification_class}_{signal_type}_model.pt"
    torch.save({
        'state_dict': tmm.model_performance[LoggingTemplate.state_dict],
        'model_args': tmm.model_args
    }, model_path)

def model_exists(classification_class, signal_type, prefix):

    data_dir = f"SavedModels/{prefix}Models"
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
    for file_name in all_files:
        if classification_class in file_name and signal_type in file_name:
            return True
        
    return False

def get_random_indis():
    for sig in Signal.ALL_SIGNALS:
        for st in Classes.All_CLASSES:
            if model_exists(st, sig, "Evil"):
                print(f"Model for {st} in {sig} already exists in EvilModels. Skipping...")
                continue
                
            print("\nTRAINING", st, "IN", sig)
            sdl = SDataLoader(sig, st)
            indi = []
            for _ in range(random.randint(*ModelManager.NUMBER_OF_BRANCHES_RANGE)):
                indi.append([random.randint(0, 750) for _ in range(random.randint(*ModelManager.NUMBER_OF_KERNELS_RANGE))])
            print("Branches: ", indi)
            m = TrainedModelMaker(
                branches=indi,
                N_SAMPLES=30 if sig==Signal.EMG.SUBMENTAL else 3000,
                pos_weight=sdl.pos_weight,
                train_loader=sdl.train_loader,
                test_loader=sdl.test_loader,
            )
            save_model(m, sig, st, "Evil")

def get_good_indis():
    for sig in Signal.ALL_SIGNALS:
        for st in Classes.All_CLASSES:
            if model_exists(st, sig, "Good"):
                print(f"Model for {st} in {sig} already exists in GoodModels. Skipping...")
                continue
                
            print("TRAINING", st, "IN", sig)
            sdl = SDataLoader(sig, st)
            m = TrainedModelMaker(
                branches=[[400,22,22]],
                N_SAMPLES=30 if sig==Signal.EMG.SUBMENTAL else 3000,
                pos_weight=sdl.pos_weight,
                train_loader=sdl.train_loader,
                test_loader=sdl.test_loader,
            )
            save_model(m, sig, st, "Good")

if __name__ == "__main__":
    main()