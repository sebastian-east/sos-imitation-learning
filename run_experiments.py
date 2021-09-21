import numpy as np
from sos.sos import Learning
from config.schedule import test_schedule

if __name__ == "__main__":

    test = 0
    for system, algorithm, step_length, N, std, seed, iterations \
            in test_schedule:

        test += 1

        sos = Learning(system.states, system.Z, system.A, system.B,
                       oP=system.oP, oF=system.oF, epsilon=0.1)
        is_feasable = sos.feasability_check(verbose=False)

        if is_feasable:
            filename = system.name + str(N) + '.npz'
            datafile = np.load('./data/experts/' + filename)
            data = {key : datafile[key] for key in datafile.keys()}
            data['N'] = N
            sos.import_data(data)

            try:
                F, P = sos.imitate(algorithm, iterations=iterations,
                                   verbose=False, seed=seed,
                                   step_length=step_length)

                if algorithm == 'pgd':
                    save_data = {'F': F, 'P': P, 'obj': sos.objective,
                                 'Completed': True}
                elif algorithm == 'admm':
                    save_data = {'F': F, 'P': P, 'res_pri': sos.primal_residual,
                                 'obj': sos.objective1, 'Complete': True}

                np.savez('./data/results/' + system.name + '_' + algorithm
                         + '_' + str(seed) + '_' + str(N) + '.npz',
                         **save_data)

            except:
                pass