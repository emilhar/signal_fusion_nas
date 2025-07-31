from grid_search_controller.grid_search_controller import GridSearchController, GridSearchTimeoutError
from datahelpers.data import Data
from datahelpers.signal import Signal
from datahelpers.target import Target

def main():
    data = Data()
    signal = data.signal_objects[0]
    target = data.target_objects[0]

    grid_controller = GridSearchController(signal, target, dataset_percentage=1.0, epochs=10)
    grid_controller.compute_grid()


if __name__ == "__main__":
    main()