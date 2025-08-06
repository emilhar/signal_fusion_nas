from datahelpers.data import Data
from datahelpers.datasplitter import split_data

def prepare_data(mb_per_part):
    helper = Data()
    #split_data(mb_per_part)

    return helper.target_objects, helper.signal_objects
