from models.cnn_binary_classifier import CNN_BinaryClassifier

from dataloaders.data_loader import SDataLoader

from models.ensemble_model import EnsembleModel

from utils.trained_model_maker import TrainedModelMaker

from datahelpers.data import Data
from datahelpers.signal import Signal

from Globals import LoggingHelper, device

from ensemble_controller.ensemble_plotter import analyze_predictions

import os
import multiprocessing
import torch
import warnings
import numpy as np

from sklearn.metrics import confusion_matrix

class SmartBranchGenerator():
    def get_branch(self):
        return [[3, 3, 3]]


class EnsembleController:
    def __init__(self, targets, signals: list[Signal], debug=False):
        self.targets = targets
        self.signals = signals
        self.debug = debug
        self.branch_generator = SmartBranchGenerator()

    def create_ensemble(self, weights, use_temp=False):
        ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()
        p = ctx.Process(
            target=self._create_ensemble_in_process,
            args=(queue, weights, use_temp, self.targets, self.signals, self.debug)
        )
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError("Subprocess for ensemble creation failed")
        return queue.get()

    def _create_ensemble_in_process(self, queue, weights, use_temp, targets, signals, debug):
        # Reinitialize necessary components in subprocess
        from dataloaders.multimodal_dataset import get_dataloaders_with_multimodal_datasets
        from models.ensemble_model import EnsembleModel
        from Globals import device
        
        print("📦 Loading Data...")
        train_loader, test_loader = get_dataloaders_with_multimodal_datasets(targets, signals)
        
        print("🧠 Loading Models...")
        models_dict = self.load_each_model(use_temp)

        print("🚀 Training Model...")
        model = EnsembleModel(models_dict)
        model.to(device)
        trained_state = EnsembleModel.train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            weights=weights,
            epochs=10,
        )
        model.load_state_dict(trained_state)

        cm = self.get_confusion_matrix(model, test_loader)
        target_ranking = []
        for i, target in enumerate(targets):
            target_ranking.append((target, cm[i][i]))

        # Aggressive cleanup
        train_loader.dataset.clear_all()
        test_loader.dataset.clear_all()
        del train_loader
        del test_loader
        del model
        for signal_models in models_dict.values():
            for m in signal_models:
                del m
        models_dict.clear()
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        queue.put(sorted(target_ranking, key=lambda x: x[1]))

    def load_each_model(self, use_temp):
        models_dict = {}
        
        # First handle temp models if needed
        if use_temp and os.path.exists("temp_models"):
            temp_model_files = [f for f in os.listdir("temp_models") if f.endswith('.pt')]
            temp_prefixes = {f.split('_')[0] for f in temp_model_files}
            
            for signal in self.signals:
                signal_name = signal.name
                models_dict[signal_name] = []
                
                for model_file in temp_model_files:
                    if signal_name in model_file:
                        full_path = os.path.join("temp_models", model_file)
                        models_dict[signal_name].append(self.load_model(full_path))
        else:
            temp_prefixes = set()
        
        # Then handle saved models, excluding any with prefixes found in temp
        if os.path.exists("saved_models"):
            saved_model_files = [f for f in os.listdir("saved_models") 
                            if f.endswith('.pt') and 
                            (not use_temp or f.split('_')[0] not in temp_prefixes)]
            
            for signal in self.signals:
                signal_name = signal.name
                if signal_name not in models_dict:
                    models_dict[signal_name] = []
                    
                for model_file in saved_model_files:
                    if signal_name in model_file:
                        full_path = os.path.join("saved_models", model_file)
                        models_dict[signal_name].append(self.load_model(full_path))
        
        # Verify we have all expected models
        assert len(models_dict.keys()) == len(Data.get_all_signal_names()), "Not enough models"
        for k in models_dict.keys():
            assert len(models_dict[k]) == len(Data.get_all_target_names()), f"Not all signals have all targets (signal {k} has {len(models_dict[k])})"
        
        return models_dict


    def load_model(self, model_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            checkpoint = torch.load(model_path, map_location=device, weights_only=True, mmap=True)

        model_args = checkpoint["model_args"]

        model = CNN_BinaryClassifier(
            n_samples=model_args["n_samples"],
            branch_configs=model_args["branch_configs"],
            batch_size=model_args["batch_size"]
        )

        model.load_state_dict(checkpoint["state_dict"])
        return model
    

    def load_model_config(self, model_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            checkpoint = torch.load(model_path, map_location=device)

        return checkpoint["model_args"]

    
    def get_confusion_matrix(self, model, test_loader):
        all_true = []
        all_preds = []

        model.eval()
        model.to(device)  # Ensure model is on the right device

        with torch.inference_mode():
            for (datas, labels) in test_loader:
                # Move data to the same device as model
                datas = {k: v.to(device) for k, v in datas.items()}
                labels = labels.to(device)
                
                output = model(datas)
                pred_label = torch.argmax(output, dim=1).cpu()
                all_true.extend(labels.cpu().numpy())
                all_preds.extend(pred_label)

        all_true = np.array(all_true)
        all_preds = np.array(all_preds)

        analyze_predictions(all_true, all_preds, self.targets)

        return confusion_matrix(all_true, all_preds, labels=range(len(Data.get_all_target_names())), normalize="true")



    def save_ensemble(self, ensemble_model, path):
        save_data = {
            "mlp_state_dict": ensemble_model.mlp.state_dict(),
            "binary_classifier_configs": {
                signal: [model.model_args for model in ffe_list] 
                for signal, ffe_list in ensemble_model.feature_extractors.items()
            }
        }
        torch.save(save_data, path)


    def load_ensemble(self, path, device="cuda"):
        data = torch.load(path, map_location=device)
        
        models = {}
        for signal, configs in data["binary_classifier_configs"].items():
            models[signal] = [CNN_BinaryClassifier(**config) for config in configs]
        
        ensemble = EnsembleModel(models)
        ensemble.mlp.load_state_dict(data["mlp_state_dict"])
        return ensemble
    

    def get_initial_models(self):
        ctx = multiprocessing.get_context('spawn')
        processes = []
        
        for signal in self.signals:
            for target in self.targets:
                p = ctx.Process(
                    target=self._train_and_save_model_in_process,
                    args=(signal, target)
                )
                p.start()
                processes.append(p)
                p.join()
                if p.exitcode != 0:
                    raise RuntimeError("Subprocess for model training failed")

    def _train_and_save_model_in_process(self, signal, target):
        # Reinitialize necessary components in subprocess
        from dataloaders.data_loader import SDataLoader
        from utils.trained_model_maker import TrainedModelMaker
        from datahelpers.data import Data
        
        sdl = SDataLoader(signal, target, batch_size=Data.batch_size)
        indi = self.branch_generator.get_branch()
        m = TrainedModelMaker(
            branches=indi,
            N_SAMPLES=signal.n_samples,
            pos_weight=sdl.pos_weight,
            train_loader=sdl.train_loader,
            test_loader=sdl.test_loader,
            epochs=30,
            batch_size=Data.batch_size,
        )
        
        # Save the model
        os.makedirs("saved_models", exist_ok=True)
        model_path = f"saved_models/{target}_{signal}_model.pt"
        torch.save({
            'state_dict': m.model_performance["state_dict"],
            'model_args': m.model_args
        }, model_path)
        
        # Cleanup
        del sdl
        del m
        import gc
        gc.collect()
        torch.cuda.empty_cache()


    def update_filters_for_binary_models(self):
        for (target, signal), model_config in self.load_each_model_config().items():
            sdl = SDataLoader(signal, target, batch_size=Data.batch_size)

            indi = [
                model_config["branch_configs"][f"branch_{i}"]["kernel_sizes"]
                for i in range(len(model_config["branch_configs"]))
            ]
            m = TrainedModelMaker(
                branches=indi,
                N_SAMPLES=signal.n_samples,
                pos_weight=sdl.pos_weight,
                train_loader=sdl.train_loader,
                test_loader=sdl.test_loader,
                epochs=30,
                batch_size=Data.batch_size,
            )
            self.save_binary_model(m, signal, target)


    def load_each_model_config(self):
        model_configs = {}
        for signal in self.signals:
            signal_name = signal.name
            
            saved_model_files = [
                f for f in os.listdir("saved_models") 
                if f.endswith('.pt')
            ]
            
            for model_path in saved_model_files:
                if signal_name in model_path:
                    full_path = os.path.join("saved_models", model_path)
                    model_configs.append(self.load_model_config(full_path))

        return model_configs
        
    def load_each_model_config(self):
        model_configs = {}

        saved_model_files = [
                f for f in os.listdir("saved_models") 
                if f.endswith('.pt')
            ]
            
        for model_path in saved_model_files:
            model_target = [t for t in self.targets if t.given_name in model_path][0]
            model_signal = [s for s in self.signals if s.name in model_path][0]
            full_path = os.path.join("saved_models", model_path)
            model_configs[(model_target, model_signal)] = self.load_model_config(full_path)

        return model_configs
    
    def save_binary_model(self, tmm, signal, target):
        os.makedirs(f"saved_models", exist_ok=True)
        
        model_path = f"saved_models/{target}_{signal}_model.pt"
        torch.save({
            'state_dict': tmm.model_performance["state_dict"],
            'model_args': tmm.model_args
        }, model_path)