import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    SC = pd.read_excel("./Data/SC-subjects.xls")
    ST = pd.read_excel("./Data/ST-subjects.xls")
    combined = pd.concat([ST, SC], ignore_index=True)
    print(len(combined.drop_duplicates("subject")))
    
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
        np.array(edfx_train), np.array(edfx_holdout),
        combined
    )

def create_strata(df):
    subject_df = df[["subject", "age", "sex (F=1)"]].drop_duplicates(subset="subject")
    subject_df["stratum"] = subject_df["sex (F=1)"]
    return dict(zip(subject_df["subject"], subject_df["stratum"]))


def stratify_sleep_edf_20(df: pd.DataFrame, used_subjects: set) -> tuple:
    EDF_20_SUBJECTS = list(range(20))
    df = df[df["subject"].isin(EDF_20_SUBJECTS)]
    
    subject_stratum = create_strata(df)
    available_subjects = [s for s in df["subject"].unique() if s not in used_subjects]
    
    strata = [subject_stratum[s] for s in available_subjects]
    
    train, holdout = train_test_split(
        available_subjects,
        test_size=0.1,
        random_state=42,
        stratify=strata
    )
    
    used_subjects.update(holdout)
    return train, holdout


def stratify_sleep_edf_78(df: pd.DataFrame, used_subjects: set) -> tuple:
    df = df[df["subject"] > -1] # Everyone from SC is in sleepedf78s
    
    subject_stratum = create_strata(df)
    forced_holdout = [s for s in used_subjects if s in set(df["subject"])]
    available_subjects = [s for s in df["subject"].unique() if s not in forced_holdout]
    
    total_subjects = 78
    total_holdout_needed = int(round(0.1 * total_subjects))
    additional_holdout_needed = max(0, total_holdout_needed - len(forced_holdout))
    
    strata = [subject_stratum[s] for s in available_subjects]
    
    train, additional_holdout = train_test_split(
        available_subjects,
        test_size=additional_holdout_needed,
        train_size=len(available_subjects) - additional_holdout_needed,
        random_state=42,
        stratify=strata
    )
    
    holdout = forced_holdout + additional_holdout
    used_subjects.update(additional_holdout)
    
    return train, holdout


def stratify_sleep_edfx(df: pd.DataFrame, used_subjects: set) -> tuple:
    subject_stratum = create_strata(df)
    forced_holdout = [s for s in used_subjects if s in set(df["subject"])]
    available_subjects = [s for s in df["subject"].unique() if s not in forced_holdout]
    
    total_subjects = len(df["subject"].unique())
    total_holdout_needed = int(round(0.1 * total_subjects))
    additional_holdout_needed = max(0, total_holdout_needed - len(forced_holdout))
    
    strata = [subject_stratum[s] for s in available_subjects]
    
    train, additional_holdout = train_test_split(
        available_subjects,
        test_size=additional_holdout_needed,
        random_state=42,
        stratify=strata
    )
    
    holdout = forced_holdout + additional_holdout
    used_subjects.update(additional_holdout)
    
    return train, holdout


def print_subject_ids(dataset, train, holdout):
    print(dataset)
    print("Train:", ' '.join(map(str, sorted(train))))
    print("Holdout:", ' '.join(map(str, sorted(holdout))))
    print()


def sanity_check_holdout(edf20_train, edf20_holdout, edf78_train, edf78_holdout, edfx_train, edfx_holdout, dataset: pd.DataFrame):
    def get_sex_distribution(subjects, df):
        subject_sex = df[df['subject'].isin(subjects)]
        subject_sex = subject_sex.drop_duplicates('subject')
        
        female_count = (subject_sex['sex (F=1)'] == 1).sum()
        total_subjects = len(subjects)
        
        if total_subjects == 0:
            return 0.0
        
        return female_count / total_subjects
    

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

    if (x := len(s_edf20_holdout | s_edf20_train)) != 20:
        raise ValueError(f"EDF20 not fully used: {x} of 20")
    
    if (x := len(s_edf78_holdout | s_edf78_train)) != 78:
        print("Combination:")
        print(" ".join(map(str, sorted(s_edf78_holdout | s_edf78_train))))
        raise ValueError(f"EDF78 not fully used: {x} of 78")

    if len(
            (s_edf20_holdout | s_edf78_holdout | s_edfx_holdout)
            &
            (s_edf20_train | s_edf78_train | s_edfx_train)
    ) == 0:
        raise ValueError("training contains holdout data")
    
    if not s_edf20_holdout.issubset(edf78_holdout):
        raise ValueError()
    if not s_edf20_holdout.issubset(edfx_holdout):
        raise ValueError()
    if not s_edf78_holdout.issubset(edfx_holdout):
        raise ValueError()
    

    print("\nSex Distribution:")
    print("--------------------------")
    
    # SleepEDF-20 analysis
    edf20_total = len(edf20_train) + len(edf20_holdout)
    print(f"SleepEDF-20 (n={edf20_total}):")
    train_female = get_sex_distribution(edf20_train, dataset)
    holdout_female = get_sex_distribution(edf20_holdout, dataset)
    print(f"  Train: {len(edf20_train)} subjects, {train_female:.1%} female")
    print(f"  Holdout: {len(edf20_holdout)} subjects, {holdout_female:.1%} female")
    
    # SleepEDF-78 analysis
    edf78_total = len(edf78_train) + len(edf78_holdout)
    print(f"\nSleepEDF-78 (n={edf78_total}):")
    train_female = get_sex_distribution(edf78_train, dataset)
    holdout_female = get_sex_distribution(edf78_holdout, dataset)
    print(f"  Train: {len(edf78_train)} subjects, {train_female:.1%} female")
    print(f"  Holdout: {len(edf78_holdout)} subjects, {holdout_female:.1%} female")
    
    # SleepEDFx analysis
    edfx_total = len(edfx_train) + len(edfx_holdout)
    print(f"\nSleepEDFx (n={edfx_total}):")
    train_female = get_sex_distribution(edfx_train, dataset)
    holdout_female = get_sex_distribution(edfx_holdout, dataset)
    print(f"  Train: {len(edfx_train)} subjects, {train_female:.1%} female")
    print(f"  Holdout: {len(edfx_holdout)} subjects, {holdout_female:.1%} female")


    dataset = dataset.drop_duplicates("subject")
    total_female = (dataset["sex (F=1)"] == 1).sum()
    total_subjects = len(dataset["subject"].unique())
    female_percentage = total_female / total_subjects if total_subjects > 0 else 0.0

    print(f"\nOverall dataset: {female_percentage:.1%} female")


if __name__ == "__main__":
    main()