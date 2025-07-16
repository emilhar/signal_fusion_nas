import glob
import os
import shutil

import logging


edf_20_holdout = [6, 12]
edf_78_holdout = [6, 12, 14, 19, 25, 38, 49, 81]
edfx_holdout = [6, 12, 14, 19, 25, 38, 49, 64, 81, 87]


def flob():
    # sleep-EDF-20
    for signal in ["EEG_Fpz-Cz", "EEG_Pz-Oz", "EMG_submental", "EOG_horizontal"]:
        agg = []
        for x in edf_20_holdout:
            agg += glob.glob(f"Data/sleep-EDF-20/{signal}/SC4{x:02d}*")
        for src_path in agg:
            # Create holdout directory if it doesn't exist
            dst_dir = os.path.join(os.path.dirname(src_path).replace("sleep-EDF-20", "holdout_sleep-EDF-20"))
            os.makedirs(dst_dir, exist_ok=True)
            
            # Move the file
            dst_path = os.path.join(dst_dir, os.path.basename(src_path))
            shutil.move(src_path, dst_path)
            print(f"Moved: {src_path} -> {dst_path}")

    # sleep-EDF-78
    for signal in ["EEG_Fpz-Cz", "EEG_Pz-Oz", "EMG_submental", "EOG_horizontal"]:
        agg = []
        for x in edf_78_holdout:
            agg += glob.glob(f"Data/sleep-EDF-78/{signal}/SC4{x:02d}*")
        for src_path in agg:
            dst_dir = os.path.join(os.path.dirname(src_path).replace("sleep-EDF-78", "holdout_sleep-EDF-78"))
            os.makedirs(dst_dir, exist_ok=True)
            
            dst_path = os.path.join(dst_dir, os.path.basename(src_path))
            shutil.move(src_path, dst_path)
            print(f"Moved: {src_path} -> {dst_path}")

    # sleep-EDFx
    for signal in ["EEG_Fpz-Cz", "EEG_Pz-Oz", "EMG_submental", "EOG_horizontal"]:
        agg = []
        for x in edfx_holdout:
            agg += glob.glob(f"Data/sleep-EDFx/{signal}/SC4{x:02d}*")
        for src_path in agg:
            dst_dir = os.path.join(os.path.dirname(src_path).replace("sleep-EDFx", "holdout_sleep-EDFx"))
            os.makedirs(dst_dir, exist_ok=True)
            
            dst_path = os.path.join(dst_dir, os.path.basename(src_path))
            shutil.move(src_path, dst_path)
            print(f"Moved: {src_path} -> {dst_path}")

if __name__ == "__main__":
    flob()