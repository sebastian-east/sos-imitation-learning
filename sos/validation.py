import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.integrate import odeint
from sympy import expand, collect
from .sos import extract_monoms_and_coeffs

def vector(x, y, system, states):
    ''' returns a vector of derivaties (given by system) for a 2d ODE, `states'
    are the states of the syste, x is the value of state[0], and y is the value
    of state[1].

    ONLY WORKS WITH A 2DOF SYSTEM

    Parameters:
    x (float) : value to evaluate for state[0]
    y (float) : value to evaluate for state[1]
    system (2x1 sympy.Matrix) : ODE of system
    states (list[sympy.symbol]) : the symbolic variables used for the two
        states in `system'
    '''
    xdot = system.evalf(subs={states[0]:x, states[1]:y})
    return float(xdot[0]), float(xdot[1])

def lyap(x, y, function, states):
    ''' Evaluates a *2D* lyapunov function at point (x, y)

    ONLY WORKS WITH A 2DOF SYSTEM

    Parameters:
    x (float) : value to evaluate for state[0]
    y (float) : value to evaluate for state[1]
    function (2x1 sympy.Matrix) : Lyapunov function
    states (list[sympy.symbol]) : the symbolic variables used for the two
        states in `system'
    '''
    return np.log10(float(function.evalf(subs={states[0]:x, states[1]:y})))

def print_check(LHS, RHS, states, string):
    ''' Checks that the monomials and coefficients of two polynomials are
     equal, and prints the values on `both sides'.

    Parameters:
    TODO: check these
    '''
    print('-'*85)
    print('        Checking ' + string)
    print('-'*85)
    monL, coeffL = extract_monoms_and_coeffs(LHS, states)
    monR, coeffR = extract_monoms_and_coeffs(RHS, states)
    mL = {m : c for m, c in zip(monL, coeffL)}
    mR = {m : c for m, c in zip(monR, coeffR)}
    print('\n Monomials present on both sides: \n')
    for mon in set(monL).intersection(set(monR)):
        print('Absolute difference : %.5f | LHS value : %.5f '
              '| RHS value : %.5f | monomial : '
              % (np.abs(mL[mon] - mR[mon]), mL[mon], mR[mon]), mon)
    print('\n Monomials only present on LHS: \n')
    for mon in set(monL).difference(set(monR)):
        print(mon, mL[mon])
    print('\n Monomials only present on RHS: \n')
    for mon in set(monR).difference(set(monL)):
        print(mon, mR[mon])

def polynomial_check(sos, threshold=1E-6):
    ''' Checks that the constraints of a sos 'Learning' object are satisfied,
    and prints them.

    Parameters:
    sos (sos.Learning) : The learning object to validate.
    threshold (float) : coefficients below this threshold are ignored.
    '''

    print('-'*85)
    print('        ' +
          'Checking that polynomials on both sides of constraints are equal')
    print('-'*85)
    K, P, Q1, Q2 = sos.return_variables(threshold=threshold)
    LHS = P
    for monomial in sos.c1.monomials:
        LHS += monomial * sos.c1.variables[monomial]
    LHS = expand((sos.V.T @ LHS @ sos.V)[0])
    RHS = expand((sos.Kron.T @ Q1 @ sos.Kron)[0])
    print_check(LHS, RHS, sos.x + sos.v, 'First Constraint')

    e, _ = eigsh(Q1, 1, which='SA')
    print(' Minimum eigenvalue of Q1 = %.5f \n' % e)

    LHS = - sos.M @ sos.A @ P - (sos.M @ sos.A @ P).T \
          - sos.M @ sos.B @ K - (sos.M @ sos.B @ K).T
    for i in sos.zero_rows:
        LHS += sp.diff(P, sos.x[i]) * (sos.A[i, :] @ sos.Z)[0]
    for monomial in sos.c2.monomials:
        #LHS += monomial * sos.c2.variables[monomial]
        pass
    LHS = expand((sos.V.T @ LHS @ sos.V)[0])
    RHS = expand((sos.Kron2.T @ Q2 @ sos.Kron2)[0])
    print_check(LHS, RHS, sos.x + sos.v, 'Second Constraint')
    e, _ = eigsh(Q2, 1, which='SA')
    print(' Minimum eigenvalue of Q2 = %.5f \n' % e)

def lyapunov_check(sos, threshold=1E-6):
    ''' Checks that the Lyapunov function of a sos 'Learning' object and its
    derivate are valid by printing them.

    Parameters:
    sos (sos.Learning) : The learning object to validate.
    threshold (float) : coefficients below this threshold are ignored.
    '''
    print('-'*85)
    print('        Calculating Lypunov Function and Time Derivative')
    print('-'*85)
    K, P, _, _ = sos.return_variables(threshold=threshold)
    Pinv = P.inv()
    A_hat = (sos.A + sos.B @ K @ Pinv)
    system = A_hat @ sos.Z
    lyapunov = sp.expand((sos.Z.T @ Pinv @ sos.Z)[0])
    print(' Lyapunov function : ', lyapunov)

    lyap_deriv = Pinv @ sos.M @ A_hat
    lyap_deriv = lyap_deriv + lyap_deriv.T
    for i in sos.zero_rows:
        lyap_deriv += sp.diff(Pinv, sos.x[i]) * (sos.A[i, :] @ sos.Z)[0]
    lyap_deriv = expand((sos.Z.T @ lyap_deriv @ sos.Z)[0])
    print(' Lyapunov time derivative : ', lyap_deriv, '\n')

    return system, lyapunov, lyap_deriv

def visualise(system, lyapunov, states, xlims=[-5, 5], ylims=[-5, 5], nx=50,
              ny=50):
    ''' Visualises the Lyapunov function and generates a streamline plot for a
    *2D* system.

    Parameters:
    system (2x1 sympy.Matrix) : system dynamics.
    lyapunov (TODO: check this) : the lyapunov function
    states (list[sympy.sybols]) : state variables in the system dynamics and lyapunov function.
    xlims, ylims (list[float]) : limits of figure.
    nx, ny (int) : discretization in x and y directions.
    '''

    if len(system) != 2:
        print('error: system not 2 dimensional')
        return

    x = np.linspace(xlims[0], xlims[1], nx)
    y = np.linspace(ylims[0], ylims[1], ny)
    X, Y = np.meshgrid(x, y)
    Vector = np.vectorize(vector, excluded=[2, 3])
    Lyapunov = np.vectorize(lyap, excluded=[2, 3])
    vx, vy = Vector(X, Y, system, states)
    Z = Lyapunov(X, Y, lyapunov, states)
    plt.contourf(X, Y, Z)
    plt.streamplot(X, Y, vx, vy, color='k')
    plt.grid()
    plt.show()

def controller_gen(sos):
    ''' Generate the controller from the learned F(x) and P(x) polynomial 
    matrices in a sos.Learning object.

    Parameters:
    sos (sos.Learning) : the imitation learning object
    '''
    F, P, _, _ = sos.return_variables(threshold=1E-3)
    controller = expand(F @ P.inv() @ sos.Z, sos.x)
    print('Controller ', controller[0])

'''
def simulate(sos, states, x0):
    K, P, _, _ = sos.return_variables(threshold=1E-3)
    system = (sos.A + sos.B @ K @ P.inv()) @ sos.Z
    sol = odeint(dynamics, x0, t=np.logspace(-3, 2, 100),
                 args=(system, states))
    plt.plot(sol[:, 0], sol[:, 1], 'ro')
    return sol[:, 0], sol[:, 1]

def dynamics(x, t, system, states):
    dyn = sp.lambdify(states, np.squeeze(system), "numpy")
    return np.array(dyn(*x))
'''