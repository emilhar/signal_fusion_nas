import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    SC = pd.read_excel("./Data/SC-subjects.xls")
    ST = pd.read_excel("./Data/ST-subjects.xls")
    combined = pd.concat([ST, SC], ignore_index=True)
    
    statistics(combined)

    used_subjects = set()
    
    edf20_train, edf20_holdout = stratify_sleep_edf_20(SC, used_subjects)
    edf78_train, edf78_holdout = stratify_sleep_edf_78(SC, used_subjects)
    edfx_train, edfx_holdout = stratify_sleep_edfx(combined, used_subjects)

    used_subjects = set(int(x) for x in used_subjects)

    print_subject_ids("SleepEDF-20", edf20_train, edf20_holdout)
    print_subject_ids("SleepEDF-78", edf78_train, edf78_holdout)
    print_subject_ids("SleepEDFx", edfx_train, edfx_holdout)

    sanity_check_holdout(
        np.array(edf20_train), np.array(edf20_holdout), 
        np.array(edf78_train), np.array(edf78_holdout),
        np.array(edfx_train), np.array(edfx_holdout)
    )

def statistics(df: pd.DataFrame):
    print(df)
    AGE = "age"
    SEX = "sex"




def stratify_sleep_edf_20(df: pd.DataFrame, used_subjects: set) -> tuple:
    EDF_20_SUBJECTS = list(range(20))
    df = df[df["subject"].isin(EDF_20_SUBJECTS)]
    
    available_subjects = [s for s in df['subject'].unique() if s not in used_subjects]    
    train, holdout = train_test_split(available_subjects, test_size=0.1, random_state=42)
    
    used_subjects.update(holdout)
    
    return train, holdout

def stratify_sleep_edf_78(df: pd.DataFrame, used_subjects: set) -> tuple:
    EDF_78_SUBJECTS = list(range(79))
    df = df[df["subject"].isin(EDF_78_SUBJECTS)]
    
    forced_holdout = [s for s in used_subjects if s in set(df['subject'])]
    available_subjects = [s for s in df['subject'].unique() if s not in forced_holdout]
    
    total_subjects = 78
    total_holdout_needed = int(round(0.1 * total_subjects))
    additional_holdout_needed = max(0, total_holdout_needed - len(forced_holdout))
    
    if additional_holdout_needed > 0 and available_subjects:
        train, additional_holdout = train_test_split(
            available_subjects,
            test_size=additional_holdout_needed,
            random_state=42
        )
    else:
        train = available_subjects
        additional_holdout = []
    
    holdout = forced_holdout + additional_holdout
    used_subjects.update(additional_holdout)
    
    return train, holdout

def stratify_sleep_edfx(df: pd.DataFrame, used_subjects: set) -> tuple:
    forced_holdout = [s for s in used_subjects if s in set(df['subject'])]
    available_subjects = [s for s in df['subject'].unique() if s not in forced_holdout]
    
    total_subjects = len(df['subject'].unique())
    total_holdout_needed = int(round(0.1 * total_subjects))
    additional_holdout_needed = max(0, total_holdout_needed - len(forced_holdout))
    
    if additional_holdout_needed > 0 and available_subjects:
        train, additional_holdout = train_test_split(
            available_subjects,
            test_size=additional_holdout_needed,
            random_state=42
        )
    else:
        train = available_subjects
        additional_holdout = []
    
    holdout = forced_holdout + additional_holdout
    used_subjects.update(additional_holdout)
    
    return train, holdout

def print_subject_ids(dataset, train, holdout):
    print(dataset)
    print("Train:", ' '.join(map(str, sorted(train))))
    print("Holdout:", ' '.join(map(str, sorted(holdout))))
    print()



def sanity_check_holdout(edf20_train, edf20_holdout, edf78_train, edf78_holdout, edfx_train, edfx_holdout):
    if len(edf20_holdout) != len(np.unique(edf20_holdout)):
        raise ValueError("EDF20 holdout contains duplicates")
    if len(edf78_holdout) != len(np.unique(edf78_holdout)):
        raise ValueError("EDF78 holdout contains duplicates")
    if len(edfx_holdout) != len(np.unique(edfx_holdout)):
        raise ValueError("EDFx holdout contains duplicates")
    
    if len(edf20_train) != len(np.unique(edf20_train)):
        raise ValueError("EDF20 holdout contains duplicates")
    if len(edf78_train) != len(np.unique(edf78_train)):
        raise ValueError("EDF78 holdout contains duplicates")
    if len(edfx_train) != len(np.unique(edfx_train)):
        raise ValueError("EDFx holdout contains duplicates")
    

    s_edf20_holdout = set(edf20_holdout)
    s_edf20_train = set(edf20_train)
    s_edf78_holdout = set(edf78_holdout)
    s_edf78_train = set(edf78_train)
    s_edfx_holdout = set(edfx_holdout)
    s_edfx_train = set(edfx_train)

    if len(
            (s_edf20_holdout & s_edf78_holdout & s_edfx_holdout)
            ^ 
            (s_edf20_train & s_edf78_train & s_edfx_train)
    ) == 0:
        raise ValueError("training contains holdout data")
    
    if not s_edf20_holdout.issubset(edf78_holdout):
        raise ValueError()
    if not s_edf20_holdout.issubset(edfx_holdout):
        raise ValueError()
    if not s_edf78_holdout.issubset(edfx_holdout):
        raise ValueError()
    

if __name__ == "__main__":
    main()