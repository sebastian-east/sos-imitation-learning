import numpy as np
from sympy.utilities.lambdify import lambdify

def control(states, controller, x):
    ''' Generates control inputs given states x and controller.

    Parameters:
    states (list[sympy.symbol]) : the states used in controller
    controller (m x 1 sympy.Matrix) : a polynomial matrix in terms of 'states'.
    x (n x N numpy.array) : a matrix of N samples of states vector (of dimension n)

    Returns:
    _ (m x N numpy.array) : a matrix of N samples of control actions.
    '''
    cont = lambdify(states, controller)
    return cont(*x).squeeze(1)

def generate(states, controller, N, min_state=-20, max_state=20, seed=0, std=1):
    ''' Generates training data for a given system.

    Parameters:
    states (list[sympy.symbol]) : the system's state variables.
    controller (m x 1 sympy.Matrix) : the `expert' controller.
    N (int) : the number of samples to be generated.
    min_state, max_state (int) : the minimum and maximum values for the states to be sampled from.
    seed (int) : the random seed used for data generation.
    std (float) : the standard deviation of the noise added to the training data.

    Returns:
    _ (dict) : a dictionary of training data with key-value pairs:
        'x' (n x N numpy.array) : state samples
        'u' (m x N numpy.array) : control samples
        'N' (int) : number of samples
    '''
    np.random.seed(seed)
    x = np.random.uniform(min_state, max_state, size=(len(states), N))
    u = control(states, controller, x)
    u += np.random.normal(scale=std, size=u.shape)
    return {'x': x, 'u': u, 'N': N}