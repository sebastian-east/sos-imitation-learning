import sympy as sp

x1, x2 = sp.symbols('x1, x2')

class NonlinearController:
    states = [x1, x2]
    Z = sp.Matrix([x1, x2])
    A = sp.Matrix([[0, 1], [-1, 0]])
    B = sp.Matrix([[0], [1]])
    K = sp.Matrix([[-0.1 -0.1*x1**2, -0.1-0.1*x2**2]])
    system = (A + B @ K) @ Z
    oP = 0
    oF = 2
    oV = 4
    name = 'nonlinear_controller'

class NonlinearSystem:
    states = [x1, x2]
    Z = sp.Matrix([x1, x2])
    A = sp.Matrix([[-1 + x1 - 3 / 2 * x1**2 - 3 / 4 * x2**2,
                        1 / 4 - x1**2 - 1 / 2 * x2**2], [0, 0]])
    B = sp.Matrix([[0], [1]])
    K = sp.Matrix([[-2, -10]])
    system = (A + B @ K) @ Z
    oP = 0
    oF = 0
    oV = 2
    name = 'nonlinear_system'
