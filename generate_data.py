import os
import numpy as np
from sos.data_generation import generate
from config.schedule import test_schedule

if __name__ == "__main__":
    for system, _, _, N, std, _, _ in test_schedule:
        files = os.listdir('./data/experts')
        filename = system.name + str(N) + '.npz'
        if any(filename in s for s in files):
            print('File - ' + filename + ' - already exists.')
            pass
        else:
            data = generate(system.states, system.K @ system.Z, N, std=std)
            np.savez('data/experts/' + filename, **data)