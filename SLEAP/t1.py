import numpy as np

data = np.load('Data/telemetry/TestingData/telemetry_EEG_Fpz-Cz_test.npz')
y = data['y']

print(len(y))
