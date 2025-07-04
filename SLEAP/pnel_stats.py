import pandas as pd

NAME = "name"
F1 = "F1"
LOSS = "loss"
SIGNAL = "signal"
SLEEP_STAGE = "sleep_stage"
TRAIN_TIME = "train_time"

def main():
    signal, sleep_stage = get_input()
    df = pd.read_csv("./Logs/T_fully_trained_models.csv")
    df = filter_data(
        df, 
        sleep_stage=sleep_stage, 
        signal=signal,
    )
    overview(df, signal, sleep_stage)
    

def overview(df: pd.DataFrame, signal, sleep_stage):
    branches = []
    for branch_str in df[NAME]:
        branches.append(eval(branch_str))

    title = f"    {signal} Models predicting {sleep_stage}    "
    divider = "=" * len(title)
    print(f"\n{divider}\n{title}\n{divider}")

    print(f"Entries: {len(df)}")
    print(f"Average branch number: {sum(len(branch) for branch in branches) / len(branches):.2f}")
    
    conv_counts = []
    for branch in branches:
        conv_counts.append(len(branch[0]))
    print(f"Average conv layers: {sum(conv_counts) / len(conv_counts):.2f}")

    print(f"\nF1 Score - Min: {df[F1].min():.4f}, Max: {df[F1].max():.4f}, Mean: {df[F1].mean():.4f}")
    print(f"Loss - Min: {df[LOSS].min():.4f}, Max: {df[LOSS].max():.4f}, Mean: {df[LOSS].mean():.4f}")

    print("\n\n===== Performance Analysis =====")
    
    print("\n>>>>> F1 Score Quartiles <<<<<")
    print("-"*100)
    best = quartile_stats(df, crit=F1, q=0.25, which='best')
    print("-"*100)
    worst = quartile_stats(df, crit=F1, q=0.25, which='worst')
    print("-"*100)

    print("\n\n\n>>>>> Loss Quartiles <<<<<")
    print("-"*100)
    quartile_stats(df, crit=LOSS, q=0.25, which='best')
    print("-"*100)
    quartile_stats(df, crit=LOSS, q=0.25, which='worst')
    print("-"*100)

def quartile_stats(df: pd.DataFrame, crit: str, q: float, which: str = 'best'):
    ascending = True
    if crit == F1:
        ascending = False if which == "best" else True
    if crit == LOSS:
        ascending = True if which == "best" else False

    
    total_models = len(df)
    num_models = max(1, int(total_models * q))
    
    if which == 'best':
        sorted_df = df.sort_values(crit, ascending=ascending).head(num_models)
    else:
        sorted_df = df.sort_values(crit, ascending=ascending).head(num_models)
    
    metric_name = "F1 Score" if crit == F1 else "Loss"
    print(f"\n{which.capitalize()} {q*100}% {metric_name}:")
    print(f"Models: {len(sorted_df)}")
    
    if len(sorted_df) > 0:
        print(f"Range: {sorted_df[crit].min():.4f} - {sorted_df[crit].max():.4f}")
        print(f"Average: {sorted_df[crit].mean():.4f}")
        
        print("Entries:")
        print(sorted_df[[NAME, F1, LOSS, TRAIN_TIME]].to_string(index=False))
    else:
        print("No models found in this quartile")


    return sorted_df


def get_input():
    signals = ("EEG_Fpz-Cz", "EEG_Pz-Oz", "EMG_submental", "EOG_horizontal")
    stages = ("W", "N1", "N2", "N3", "REM")
    
    for i, signal in enumerate(signals, 1):
        print(f"{i}. {signal}")
    while True:
        try:
            a = int(input("Select signal: ")) - 1
            if 0 <= a < len(signals):
                break
        except ValueError:
            pass
        print(f"Invalid input. Enter 1-{len(signals)}")
    
    for i, stage in enumerate(stages, 1):
        print(f"{i}. {stage}")
    while True:
        try:
            b = int(input("Select stage: ")) - 1
            if 0 <= b < len(stages):
                break
        except ValueError:
            pass
        print(f"Invalid input. Enter 1-{len(stages)}")
    
    return signals[a], stages[b]


def filter_data(df: pd.DataFrame, **filters) -> pd.DataFrame:
    for column, value in filters.items():
        if column in df.columns:
            df = df[df[column] == value]
    return df


if __name__ == "__main__":
    main()