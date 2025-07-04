import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    signals = ("EEG_Fpz-Cz", "EEG_Pz-Oz", "EMG_submental", "EOG_horizontal")
    sleep_stages = ("W", "N1", "N2", "N3", "REM")

    random = "random_" if (i := input("Random? [y/n] ")) == "" or i == "y" else ""
    epochs = "1_" if (i:= input("Epochs? [1-10] ")) == "" else i + "_"
    for i, signal in enumerate(signals, start=1):
        print(i, signal)
    signal = signals[int(input()) - 1]
    print("\n")
    for i, sleep_stage in enumerate(sleep_stages, start=1):
        print(i, sleep_stage)
    sleep_stage = sleep_stages[int(input()) - 1]

    img = mpimg.imread(f"./Logs/penelope_plots/{random}{epochs}{sleep_stage}_{signal}_f1.png")
    plt.title(signal + " for sleep stage " + sleep_stage, fontsize=30)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    while True:
        main()