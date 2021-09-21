import numpy as np
import sympy as sp
import cvxpy as cv
import itertools
from sympy.polys.orderings import monomial_key
from sympy.utilities.lambdify import lambdify
from sympy import S, expand
from scipy.special import comb
from scipy.sparse import dok_matrix
import jax
jax.config.update('jax_platform_name', 'cpu')
from jax import grad
import jax.numpy as jnp
import jax.random as jr

def n_monomials(number_of_states, max_degree):
    ''' Returns the number of unique monomials in `number_of_states' of degree up to
    and including 'max_degree'.

    Parameters:
    number_of_states (int) : number of unique variables that can be included in
    each monomial.
    max_degree (int) : maximum degree of the monomials to be considered.

    Returns:
    total (int) : The number of unique monomials.

    '''

    total = 0
    for i in range(max_degree + 1):
        total += comb(i + number_of_states - 1, number_of_states - 1)
    return int(total)

def monomial_generator(states, max_degree, min_degree):
    ''' Generates all monomials with variables in 'states' with maximum degree
    'max_degree' and minimum degree 'min_degree'. Returns monomials as a list,
    sorted in graded lexicographical order.

    Parameters:
    states (list[sympy.symbol]) : A list of variables used to generate the
    monomials.
    max_degree (int) : maximum degree of the monomials.
    min_degree (int) : minimum degree of the monomials.

    Returns:
    _ (list(sympy.symbol)) : a list of monomials.
    '''

    return sorted(sp.itermonomials(states, max_degree, min_degree),
                  key=monomial_key('grlex', states[::-1]))

def zero_rows_B(B):
    ''' Determines which rows of a matrix `B` contain only zeros.

    Parameters:
    B (m x n numpy array) : The matrix that the zero rows are to be found in.

    Returns:
    zero_rows (list(int)) : A list of row indexes of zero rows.
    '''

    zero_rows = []
    n, m = B.shape
    for i in range(n):
        zero_rows.append(i)
        for j in range(m):
            if B[i, j] != 0:
                zero_rows.pop()
                break

    return zero_rows

def extract_monomials(matrix, states):
    ''' Determines which monomials of `states' are present in `matrix'.

    Parameters:
    matrix (sympy.Matrix) : A polynomial matrix.
    states (list(sympy.symbol)) : A list of symbolic variables.

    Returns:
    monomials (set(sympy.symbol)) : a set of monomials present in matrix.
    '''

    monomials = set()
    m, n = matrix.shape
    polys = [sp.Poly(matrix[row, column], states)
             for row, column in itertools.product(range(m), range(n))]
    for poly in polys:
        for power in poly.monoms(): # monoms returns monomials in lexigraphical order
            term = 1
            for state, p in zip(states, power):
                term = term * state**p
            monomials.add(term)
    return monomials

def extract_monoms_and_coeffs(polynomial, states):
    ''' Determines the monomials of `states' present in 'polynomial' and their
    coefficients.

    Parameters:
    polynomial (??) : A polynomial.
    states (list(sympy.symbol)) : A list of symbolic variables.

    Returns:
    monomials (list(sympy.symbol)) : A list of monomials.
    coefss (list(??)) : A list of coefficients.
    '''

    coeffs = []
    monomials = list(extract_monomials(sp.Matrix([polynomial]), states))
    for monomial in monomials:
        coeff = sp.Poly(polynomial, states).coeff_monomial(monomial)
        coeffs.append(coeff)
    return monomials, coeffs

def einsum1(A, B):
    ''' Matrix operation for PGD loss function calculation

    Parameters:
    A : a dictionary of nxm numpy arrrays
    B : a dictionary of numpy arrays of length p

    Returns:
    C : a nxmxp numpy array such that
        C[:, :, i] = sum A[key] * B[key][i]
    for key in A.keys()
    '''
    return sum([jnp.einsum('ij,k->ijk', A[mon], B[mon])
                for mon in A.keys()])

def einsum2(A, B, C):
    ''' Another matrix operation for PGD loss function calculation

    Parameters:
    A (numpy.array ?? ) : A n x m x p numpy array
    B ...
    C ...

    returns
    _ : a numpy array such that...
    '''
    return jnp.einsum('ijk,jl,lk->ik', A, B, C)

def round_expr(expr, num_digits):
    ''' Rounds the coefficients of symbolc 'expr' to 'num_digits' significant
    figures.

    Parameters:
    expr (sympy.??) : A symbolic expression.
    num_digits (int) : number of significant figures to round to.

    Returns:
    _ (sympy.??) : A symbolic expression.
    '''
    return expr.xreplace({n : round(n, num_digits)
                          for n in expr.atoms(sp.Number)})

class Analysis:
    '''A class for sum-of-squares stability analysis of polynomial systems.
    '''

    def __init__(self, states, system, oV=2, epsilon=0.0001, verbose=False):
        ''' Immediately runs the entire stability analysis on initialization.

        Parameters:
        states (list[sympy.symbol]): list of the symbols use in the system dynamics
        system (nx1 sympy.matrix): the dynamics of the system
        oV (int): desired order of the lyapunov function (must be even)
        epsilon (float): parameter for enforcing degree of asymptotic stability
        verbose (bool): switches on verbose printing of the optimization software
        '''

        self.x = states
        self.f = system
        self.oV = oV
        self.epsilon = epsilon

        self.generate_variables()
        self.generate_sdp()
        self.solve(verbose)

    def generate_variables(self):
        ''' Creates variables for polynomial coefficients used in sos analysis.
        '''

        self.constant = PolyCon(self.x,
                                sum([self.epsilon * x**2 for x in self.x]))
        self.z1 = sp.Matrix(monomial_generator(self.x, self.oV // 2, 0))
        self.M = self.z1.jacobian(self.x)
        self.P = MatrixVar(len(self.z1), len(self.z1), states=[S.One],
                           symmetric=True)
        self.poly1 = Polynomial(self.P, self.x, self.z1, self.z1)

        self.poly2 = Polynomial(self.P, self.x, 2 * self.z1, self.M @ self.f)
        if self.poly2.max_degree % 2 != 0:
            print(self.poly2.max_degree / 2)

        d2 = self.poly2.max_degree // 2
        self.z2 = sp.Matrix(monomial_generator(self.x, d2, 0))
        self.Q = MatrixVar(len(self.z2), len(self.z2), symmetric=True)
        self.sos = Polynomial(self.Q, self.x, self.z2, self.z2)

    def generate_sdp(self):
        ''' Creates the semidefinite program for sos stability analysis.
        '''

        self.constraints = []
        for monomial in self.constant.monomials:
            self.constraints += [self.poly1.coefficient(monomial)
                                 >= self.constant.coefficient(monomial)]
        for mon in self.P.monomials:
            self.constraints += [self.P.variables[mon] >> 0]
        for mon in self.Q.monomials:
            self.constraints += [self.Q.variables[mon] << 0]

        for monomial in self.sos.monomials:
            self.constraints += [self.poly2.coefficient(monomial)
                                 + self.constant.coefficient(monomial)
                                 == self.sos.coefficient(monomial)]

    def solve(self, verbose):
        ''' Solves the sos feasability problem. Feasability implies that the
        system is stable.
        '''

        obj = 0
        for mon in self.P.monomials:
            obj += cv.norm(self.P.variables[mon], 'fro')
        self.prob = cv.Problem(cv.Minimize(obj), self.constraints)
        self.prob.solve(verbose=verbose, solver=cv.SCS)
        #self.prob.solve(verbose=verbose, solver=cv.CVXOPT)
        #self.prob.solve(verbose=verbose, solver=cv.MOSEK)
        print(self.prob.status)

    def lyapunov(self, printout=False, dec=7):
        ''' Returns the Lypunov function generated from the sos feasability
        problem.

        Parameters:
        printout (bool): Determines whether to print the Lyapunov function and
        its derivative.
        dec (int): number of decimal places to print the Lypunov function and
        its deriative to.
        '''

        Vtemp = expand((self.z1.T @ self.P.variables[1].value @ self.z1)[0])
        monoms, coeffs = extract_monoms_and_coeffs(Vtemp, self.x)
        V = sum([m * round(c, dec) for m, c in zip(monoms, coeffs)
                if np.abs(c) >= 10**-dec])
        if printout:
            print('Lyapunov function: ', V)
            Vdottemp = expand((2 * self.z1.T @ self.P.variables[1].value
                               @ self.M @ self.f)[0])
            monoms, coeffs = extract_monoms_and_coeffs(Vdottemp, self.x)
            Vdot = sum([m * round(c, dec) for m, c in zip(monoms, coeffs)
                        if np.abs(c) >= 10**-dec])
            print('Lyapunov time derivative: ', Vdot)
        return V

class Learning:
    '''A class for imitation learning with sum-of-squares stability guarantees.
    '''

    def __init__(self, states, Z, A, B, oP=0, oF=0, epsilon=0.001,
                 verbose=False):
        ''' Imports the system parameters for learning and immediately
        generates constraints for semdefinite feasability program.

        The variables are more fully explained in # TODO:

        Parameters:
        states (list[sympy.symbol]): list of the symbolic variables used in the system dynamics
        Z (px1 sympy.matrix) : An array of monomials of 'states'
        A (nxp sympy.matrix) : A polynomial matrix of `states'
        B (nxm sympy.matrix) : A polynomial matrix of 'states'
        oP (int): The chosen degree of the decision matrix polynomial
        P(\tilde{x}) - CURRENTLY CAN ONLY BE ZERO (i.e. P is constant)
        oF (int): The chosen degree of the decision matrix polynomial F(x)
        epsilon (float): parameter for enforcing degree of asymptotic stability
        verbose (bool): switches on verbose printing of the optimization software
        '''

        if oP != 0:
            raise Exception('Imitation learning currently only works' \
                            + ' for constant P(x) (i.e. oP=0).')
            #TODO: implement polynomial P(x)

        self.n, self.m = B.shape
        self.p = Z.shape[0]
        self.x = states.copy()
        self.Z = Z.copy()
        self.v = [sp.Symbol('v%i' % int(i + 1)) for i in range(len(Z))]
        self.V = sp.Matrix(self.v)
        self.A = A.copy()
        self.B = B.copy()
        self.oP = oP
        self.oF = oF
        self.zero_rows = zero_rows_B(B)
        if len(self.zero_rows) > 0:
            self.xTilde = [self.x[i] for i in self.zero_rows]
        else:
            self.xTilde = [S.One]
        self.nTilde = len(self.xTilde)

        self.EPS = epsilon

        print('Generating variables...')
        self.generate_variables()
        print('...done!')
        print('Generating SDP...')
        self.generate_sdp()
        print('...done!')

    def generate_variables(self):
        ''' Creates variables for polynomial coefficients used in sos analysis.
        '''

        M = sp.Matrix(self.Z).jacobian(self.x)
        #sos_constant = np.sum([- self.EPS * state**2 for state in self.x])
        sos_constant = - self.EPS * S.One

        self.P = MatrixVar(self.p, self.p, states=self.xTilde, order=self.oP,
                           symmetric=True)
        self.F = MatrixVar(self.m, len(self.Z), states=self.x, order=self.oF)
        self.c1 = MatrixCon(len(self.V), self.x, sos_constant)
        self.c2 = MatrixCon(len(self.V), self.x, sos_constant)

        self.poly1 = Polynomial(self.P, self.x + self.v, self.V, self.V)
        self.poly2 = Polynomial(self.c1, self.x + self.v, self.V, self.V)

        d1 = self.oP // 2
        z1 = sp.Matrix(monomial_generator(self.x, d1, 0))
        self.kron, self.Kron = self.kronecker(self.v, z1)

        self.Q1 = MatrixVar(len(self.kron), len(self.kron), symmetric=True)
        self.sos1 = Polynomial(self.Q1, self.x + self.v, self.Kron, self.Kron)

        self.poly3 = Polynomial(self.F, self.x + self.v,
                                - 2 * self.V.T @ M @ self.B, self.V)
        self.poly4 = Polynomial(self.P, self.x + self.v,
                                - 2 * self.V.T @ M @ self.A, self.V)
        self.poly_deriv = [Polynomial(self.P, self.x + self.v, self.V,
                                (self.A[i,:] @ self.Z)[0] * self.V, deriv=True,
                                deriv_index=i) for i in self.zero_rows]
        self.poly5 = Polynomial(self.c2, self.x + self.v, self.V, self.V)

        d2 = np.max([i.monomial_degree(self.x) for i in self.poly_deriv]
                     + [self.poly3.monomial_degree(self.x),
                     self.poly4.monomial_degree(self.x)]) // 2
        z2 = sp.Matrix(monomial_generator(self.x, d2, 0))
        self.kron2, self.Kron2 = self.kronecker(self.v, z2)

        self.Q2 = MatrixVar(len(self.kron2), len(self.kron2), symmetric=True)
        self.sos2 = Polynomial(self.Q2, self.x + self.v, self.Kron2,
                               self.Kron2)
        #these are needed for post-processing
        self.M = M

    def kronecker(self, V, X):
        ''' Returns the Kronecker product of V and X

        Parameters:
        V (list[sympy.symbol]) : List of symbolic variables.
        X (list[sympy.symbol]) : List of symbolic variables.

        Returns:
        kron (list[sympy.symbol]) : Kronecker product of V and X as list.
        _ (sympy.Matrix) : Kronecker product as sympy matrix.
        '''

        kron = []
        for v in V:
            for x in X:
                kron.append(v * x)
        return kron, sp.Matrix(kron)

    def generate_sdp(self):
        ''' Creates the semidefinite program for sos stability analysis.
        '''

        self.constraints = []

        for mon in self.Q1.monomials:
            self.constraints += [self.Q1.variables[mon] >> 0]

        for mon in self.Q2.monomials:
            self.constraints += [self.Q2.variables[mon] >> 0]

        for monomial in self.sos1.monomials:
            term = self.poly1.coefficient(monomial)
            term += self.poly2.coefficient(monomial)
            self.constraints += [term == self.sos1.coefficient(monomial)]

        for monomial in self.sos2.monomials:
            term = self.poly3.coefficient(monomial) \
                   + self.poly4.coefficient(monomial) \
                   + self.poly5.coefficient(monomial)

            for poly in self.poly_deriv:
                term += poly.coefficient(monomial)

            self.constraints += [term == self.sos2.coefficient(monomial)]

    def feasability_check(self, verbose=False):
        ''' Determines whether the learning process is feasable
        
        Parameters:
        verbose (bool) : Enable verbose optimizer output

        Returns:
        _ (bool) : Feasability of learning process
        '''

        obj = 0
        for mon in self.P.monomials:
            obj += cv.norm(self.P.variables[mon], 'fro')
        for mon in self.F.monomials:
            obj += cv.norm(self.F.variables[mon], 'fro')
        self.prob = cv.Problem(cv.Minimize(obj), self.constraints)
        self.prob.solve(verbose=verbose, solver=cv.SCS)
        #self.prob.solve(verbose=verbose, solver=cv.CVXOPT)
        #self.prob.solve(verbose=verbose, solver=cv.MOSEK)
        print(self.prob.status)
        if self.prob.status == 'optimal':
            return True
        else:
            return False

    def return_variables(self, threshold=1E-5):
        ''' Returns the optimization variables F(x), P(x), Q_1, and Q_2.

        Parameters:
        threshold (float) : coefficients below this value will be set to zero.

        Returns:
        F (sympy.matrix)
        P (sympy.matrix)
        Q1 (numpy.array)
        Q2 (numpy.array)
        '''

        F = sp.zeros(*self.F.dimension)
        for mon in self.F.monomials:
            coeffs = self.F.variables[mon].value.copy()
            coeffs[np.abs(coeffs) < threshold] = 0
            F += mon * coeffs

        P = sp.zeros(*self.P.dimension)
        for mon in self.P.monomials:
            coeffs = self.P.variables[mon].value.copy()
            coeffs[np.abs(coeffs) < threshold] = 0
            P += mon * coeffs

        Q1 = self.Q1.variables[1].value
        Q1[np.abs(Q1) < threshold] = 0
        Q2 = self.Q2.variables[1].value
        Q2[np.abs(Q2) < threshold] = 0

        return F, P, Q1, Q2

    def import_data(self, data):
        ''' Imports the training data into the imitation learning object

        Parameters:
        data (dict) : A dictionary containing the key-value pairs
            'x' (numpy.array) : state samples
            'u' (numpy.array) : control input samples
            'N' (int) : number of state and control input samples
        '''

        self.data = data.copy()
        fZ = lambdify(self.x, self.Z, 'numpy')
        self.data['Z'] = fZ(*self.data['x']).squeeze(1)
        for monomial in self.F.variables.keys():
            if monomial == S.One:
                self.data[str(monomial)] = np.ones(self.data['N'])
            else:
                f_mon = lambdify(self.x, monomial, 'numpy')
                self.data[str(monomial)] = f_mon(*self.data['x'])

    def imitate(self, algorithm, iterations=100, verbose=False, seed='zero',
                step_length=1):
        ''' Run the imitation learning algorithm

        Parameters:
        algorithm (str) : choose from either 'admm' or 'pgd'
        iterations (int) : number of iterations
        verbose (bool) : Turn verbose output on
        seed (int) : random seed for initialization ('zero' sets all arrays to
        zero instead)
        step length
            [for admm] (float) : value of `rho' parameter
            [for pgd] (str) : dcitionary for schedule of 'alpha' parameter,
            where key is first value that alpha value is used
        '''

        if algorithm == 'admm':
            self.admm_initialize(rho=step_length, seed=seed)
            for iteration in range(iterations):
                self.admm_step_1(verbose=False)
                self.admm_step_2(verbose=False)
                self.admm_step_3()
                if verbose:
                    self.admm_print()
                if iteration % 25 == 0:
                    print('ADMM iterations completed : ', iteration)
                self.primal_residual.append(self.admm_problem2.value)
                self.objective1.append(self.admm_control_eval())

            F = 0
            for mon in self.F.monomials:
                F += mon * np.array(self.F.variables[mon].value.copy())
            P = 0
            for mon in self.P.monomials:
                P += mon * np.array(self.P.variables[mon].value.copy())
            return F, P

            '''
        elif algorithm == 'pf':
            self.admm_initialize()
            self.policy_fitting()
            K = 0
            for mon in self.K.monomials:
                K += self.K.variables[mon].value * mon
            print(expand((K @ self.Z)[0]))
            print(self.K.variables[1].value)
            '''

        elif algorithm == 'pgd':
            self.pgd_initialise(seed=seed)
            self.pgd_projection()
            imitation_loss = self.pgd_objective(self.Fp, self.Pp)
            self.objective.append(imitation_loss.item())
            print(0, imitation_loss)
            for iteration in range(iterations):
                if iteration in step_length.keys():
                    print('Alpha Update')
                    alpha = step_length[iteration]
                self.pgd_grad_step(alpha=alpha)
                self.pgd_projection()
                imitation_loss = self.pgd_objective(self.Fp, self.Pp)
                self.objective.append(imitation_loss.item())
                print(iteration + 1, imitation_loss)

            F = 0
            for mon in self.F.monomials:
                F += mon * np.array(self.Fp[str(mon)])
            P = 0
            for mon in self.P.monomials:
                P += mon * np.array(self.Pp[str(mon)])
            return F, P

        else:
            raise Exception('Please choose a valid optimization algorithm.')


    ######## Projected Gradient Descent

    def pgd_initialise(self, seed):
        ''' Initialize the projected gradient descent algorithm

        Parameters:
        seed (int) : random seed for initialization
        '''
        self.grad = grad(self.pgd_objective, (0, 1))

        if seed == 'zero':
            pass
        else:
            self.key = jr.PRNGKey(seed)
            self.key, *subkeys = jr.split(self.key, len(self.F.monomials) + 1)
            self.Fp = {str(m) : jr.uniform(k, self.F.dimension, minval=-5,
                maxval=5) for m, k in zip(self.F.variables.keys(), subkeys)}
            self.key, *subkeys = jr.split(self.key, len(self.P.monomials) + 1)
            self.Pp = {str(m) : jr.uniform(k, self.P.dimension, minval=-3,
                maxval=5) for m, k in zip(self.P.variables.keys(), subkeys)}

        self.Ftilde = {str(m) : cv.Parameter(self.F.dimension)
            for m in self.F.variables.keys()}
        self.Ptilde = {str(m) : cv.Parameter(self.P.dimension)
            for m in self.P.variables.keys()}

        obj = 0
        for mon in self.F.monomials:
            obj += cv.norm(self.F.variables[mon] - self.Ftilde[str(mon)],
                           'fro')
        for mon in self.P.monomials:
            obj += cv.norm(self.P.variables[mon] - self.Ptilde[str(mon)],
                           'fro')

        self.projection = cv.Problem(cv.Minimize(obj), self.constraints)
        self.objective = []

    def pgd_control_eval(self, F, P):
        ''' Evaluate the control inputs for the state training data, given F
        and P (implemented in Jax for autodiff)

        # THIS ONLY WORKS FOR CONSTANT P - NEEDS TO BE CHANGED FOR HIGHER
        # DEGREE P MATRICES

        Parameters:
        F (dict) : F matrix with key-value pairs
        # TODO: Check this
            monomial : jax.array
        P (dict) : P matrix with key-value pairs
            monomial : jax.array

        Returns:
        _ (jax.array) : control inputs
        '''
        Fsum = einsum1(F, self.data)
        return einsum2(Fsum, jnp.linalg.inv(P['1']), self.data['Z'])

    def pgd_objective(self, F, P):
        ''' Evaluate the imitation learning cost function, given F and P

        Parameters:
        F (dict) : F matrix with key-value pairs
        # TODO: Check this
            monomial : jax.array
        P (dict) : P matrix with key-value pairs
            monomial : jax.array

        Returns:
        _ (float) : Imitation loss
        '''
        u = self.pgd_control_eval(F, P)
        return jnp.sum((u - self.data['u']) ** 2) / self.data['N']

    def pgd_grad_step(self, alpha=1E-3):
        ''' Take projected gradient step.

        Parameters:
        alpha (float) : step length
        '''
        Fgrad, Pgrad = self.grad(self.Fp, self.Pp)
        for mon in Fgrad.keys():
            self.Fp[mon] += - alpha * Fgrad[mon].copy()
        for mon in Pgrad.keys():
            self.Pp[mon] += - alpha * Pgrad[mon].copy()
            self.Pp[mon] = 0.5 * (self.Pp[mon] + self.Pp[mon].T)

    def pgd_projection(self):
        ''' Do projection step of pgd algorithm.
        '''
        for mon in self.Fp.keys():
            self.Ftilde[mon].value = np.array(self.Fp[mon].copy())
        for mon in self.Pp.keys():
            self.Ptilde[mon].value = np.array(self.Pp[mon].copy())

        self.projection.solve(verbose=False, solver=cv.SCS)

        for mon in self.F.monomials:
            self.Fp[str(mon)] = jnp.array(self.F.variables[mon].value.copy())
        for mon in self.P.monomials:
            self.Pp[str(mon)] = jnp.array(self.P.variables[mon].value.copy())

    ######## ADMM

    def admm_initialize(self, rho=1, seed='zero'):
        ''' Initialize the ADMM algorithm.

        Parameters:
        rho (float) : value of rho
        seed (int) : random seed for initialization
        '''
        self.rho = rho
        self.primal_residual = []
        self.objective1 = []

        order_K = self.F.order - self.P.order
        self.K = MatrixVar(self.m, self.p, states=self.x, order=order_K)

        self.Ftilde = {str(m) : cv.Parameter(self.F.dimension)
            for m in self.F.variables.keys()}
        self.Ptilde = {str(m) : cv.Parameter(self.P.dimension)
            for m in self.P.variables.keys()}
        self.Ktilde = {str(m) : cv.Parameter(self.K.dimension)
            for m in self.K.variables.keys()}

        self.Y = {m : cv.Parameter(self.F.dimension)
            for m in self.F.variables.keys()}

        if seed == 'zero':
            for m in self.F.variables.keys():
                self.Ftilde[str(m)].value = np.zeros(self.F.dimension)
                self.Y[m].value = np.zeros(self.F.dimension)
            for m in self.K.variables.keys():
                self.Ktilde[str(m)].value = np.zeros(self.K.dimension)
            for m in self.P.variables.keys():
                self.Ptilde[str(m)].value = np.zeros(self.P.dimension)
        else:
            np.random.seed(seed)
            for m in self.F.variables.keys():
                self.Ftilde[str(m)].value = np.random.uniform(
                    5, 5, size=self.F.dimension)
                self.Y[m].value = np.random.uniform(
                    -5, 5, size=self.F.dimension)
            for m in self.K.variables.keys():
                self.Ktilde[str(m)].value = np.random.uniform(
                    -5, 5, size=self.K.dimension)
            for m in self.P.variables.keys():
                self.Ptilde[str(m)].value = np.random.uniform(
                    -5, 5, size=self.P.dimension)

        upred = cv.sum([cv.multiply(np.expand_dims(self.data[str(mon)], 0),
            self.K.variables[mon] @ self.data['Z'])
            for mon in self.K.monomials])
        loss = cv.norm(self.data['u'] - upred, 'fro') ** 2 / self.data['N']

        con1 = {mon : self.Ftilde[str(mon)] + self.Y[mon]
                for mon in self.F.monomials}
        for mon1 in self.K.monomials:
            for mon2 in self.P.monomials:
                mon = mon1 * mon2
                con1[mon] += - self.K.variables[mon1] @ self.Ptilde[str(mon2)]

        self.loss_function = cv.Problem(cv.Minimize(loss))

        aug1 = 1 / 2 * cv.sum([rho * cv.norm(con1[mon], 'fro')**2
                               for mon in con1.keys()])

        self.admm_problem1 = cv.Problem(cv.Minimize(loss + aug1))
        # TODO: attempt another way of parametrizing this loss function.
        # CVX seems to take a lot of iterations to minimize it (even though
        # it's unconstrained) at times.

        con2 = {mon : self.F.variables[mon] + self.Y[mon]
                for mon in self.F.monomials}
        for mon1 in self.K.monomials:
            for mon2 in self.P.monomials:
                mon = mon1 * mon2
                con2[mon] += - self.Ktilde[str(mon1)] @ self.P.variables[mon2]

        aug2 = rho / 2 * cv.sum([cv.norm(con2[mon], 'fro')**2
                                 for mon in con2.keys()])

        self.admm_problem2 = cv.Problem(cv.Minimize(aug2),
                                        constraints=self.constraints)

    def admm_control_eval(self):
        ''' Evaluate the cost function with the current values of F(x) and P(x)
        '''
        # TODO: this only works for P with degree 1
        Pinv = np.linalg.inv(self.P.variables[1].value.copy())
        upred = sum([np.expand_dims(self.data[str(mon)], 0) * (self.F.variables[mon].value.copy() @ Pinv @ self.data['Z']) for mon in self.F.monomials])
        return np.linalg.norm(self.data['u'] - upred, 'fro') ** 2 / self.data['N']

    def admm_step_1(self, verbose=True):
        ''' First step of the ADMM algorithm

        Parameters:
        verbose (bool) : enables verbose output of the optimization software
        '''

        for mon in self.F.monomials:
            self.Ftilde[str(mon)].value = self.F.variables[mon].value.copy()

        for mon in self.P.monomials:
            self.Ptilde[str(mon)].value = self.P.variables[mon].value.copy()

        self.admm_problem1.solve(
            verbose=verbose, solver=cv.SCS, warm_start=True, max_iters=10000)

    def admm_step_2(self, verbose=True):
        ''' Second step of the ADMM algorithm

        Parameters:
        verbose (bool) : enables verbose output of the optimization software
        '''

        for mon in self.K.monomials:
            self.Ktilde[str(mon)].value = self.K.variables[mon].value.copy()

        self.admm_problem2.solve(verbose=verbose, solver=cv.SCS)

    def admm_step_3(self):
        ''' Third step of the ADMM algorithm
        '''

        Ydiff = {}

        for mon in self.F.monomials:
            Ydiff[mon] = self.F.variables[mon].value.copy()

        for mon1 in self.K.monomials:
            for mon2 in self.P.monomials:
                mon = mon1 * mon2
                Ydiff[mon] += (- self.K.variables[mon1].value.copy()
                               @ self.P.variables[mon2].value.copy())

        for mon in self.F.monomials:
            self.Y[mon].value += Ydiff[mon].copy()

    def admm_print(self):
        ''' Print relevant values during the ADMM algorithm
        '''

        K = 0
        P = 0
        F = 0

        for mon in self.K.monomials:
            K += self.K.variables[mon].value * mon
        for mon in self.P.monomials:
            P += self.P.variables[mon].value * mon
        for mon in self.F.monomials:
            F += self.F.variables[mon].value * mon

        #print(*[self.Y[mon].value for mon in self.Y.keys()])
        print('Solver iterations for first ADMM step : ',
              self.admm_problem1.solver_stats.num_iters)
        print('Solver iterations for second ADMM step : ',
              self.admm_problem2.solver_stats.num_iters)
        print('K(x) . Z(x)           : ',
              round_expr(expand((K @ self.Z)[0]), 4))
        print('F(x) . P^-1(x) . Z(x) : ',
              round_expr(expand((F
                                 @ np.linalg.inv(P.astype(float))
                                 @ self.Z)[0]), 4))
    '''
    def policy_fitting(self):
        self.loss_function.solve(solver=cv.SCS)
    '''

class PolyCon:
    ''' A object representation of a constant polynomial.
    '''

    def __init__(self, states, polynomial):
        ''' Initalizes the `polynomial' in terms of states

        Parameters:
        states (list[sympy.sybmol]) : the variables of the polynomial.
        polynomial (TODO: check this) : the polynomial.
        '''

        self.monomials, coeffs = extract_monoms_and_coeffs(polynomial, states)
        self.coefficients = {m : float(c)
                             for m, c in zip(self.monomials, coeffs)}

    def coefficient(self, monomial):
        ''' Returns the coefficient of the 'monomial' in polyynomial

        Parameters:
        monomial (sympy.symbol) : the monomial

        Returns:
        _ (float) : the coefficient
        '''
        if monomial in self.coefficients:
            return self.coefficients[monomial]
        else:
            return 0

class MatrixCon:
    ''' An object representation of a constant polynomial matrix
        P(x) = I * p(x),
    where p(x) is a scalar valued polynomial.
    '''

    def __init__(self, n, states, polynomial):
        ''' Initializes the polynomial matrix in terms of states, where the
        dimension of I is 'n'.

        Parameters:
        n (int) : the dimension of the identity matrix I.
        states (list[sympy.sybmol]) : the variables of the polynomial.
        polynomial (TODO: check this) : the polynomial.
        '''
        self.states = states.copy()
        self.dimension = [n, n]
        self.monomials, coeffs = extract_monoms_and_coeffs(polynomial, states)
        self.num_monomials = len(self.monomials)
        self.variables = {m : float(c) * np.eye(n)
                          for m, c in zip(self.monomials, coeffs)}
        self.is_constant = True

class MatrixVar:
    ''' On object represtntation of a polynomial matrix
            P(x) = sum_i C_i mon(x)_i,
        where C_i are *optimization variables*, and mon(x)_i are the
        constituent monomials of P(x)
    '''

    def __init__(self, n, m, states=[sp.Symbol('1')], order=0, symmetric=False):
        ''' Initializes the polynomial matrix variables

        Parameters:
        n, m (int, int) : The dimensions, n x m, of P(x).
        states (list[sympy.sybmol]) : the variables of the polynomial.
        order (int) : the maximum degree of the monomials in P(x).
        symmetrix (bool) : enforces the matrices C_i to be symmetric.
        '''
        self.states = states.copy()
        self.dimension = [n, m]
        self.order = order
        self.num_monomials = n_monomials(len(states), order)
        self.monomials = monomial_generator(states, order, 0)
        self.variables = {monomial : cv.Variable((n, m), symmetric=symmetric)
                          for monomial in self.monomials}
        self.is_constant = False

class Polynomial:
    ''' An object representing a (p x p) polynomial of the form
            polynomial = z1^T * matrix * z2,
        where 'z1' and 'z2' are (fixed) polynomial vectors, and 'matrix' is a
        (variable) polynomial matrix.

        Used to generate SDP constraints.
    '''

    def __init__(self, matrix, states, z1, z2, deriv=False, deriv_index=None):
        ''' Initializes the polynomial.

        matrix (MatrixVar) : the polynomial matrix.
        states (list[sympy.sybmol]) : the variables of the vectors z1 and z2.
        z1, z2 (1xp sympy.Matrix) : the vectors of polynomials.
        deriv (bool) : replaces matrix in the polynomial with d matrix / d var_i
        deriv_index (int) : index i such that var_i = states[i]
        '''
        self.matrix = matrix
        z1_monomials = extract_monomials(z1, states)
        z2_monomials = extract_monomials(z2, states)

        poly_mons = [x * y for x in z1_monomials for y in z2_monomials]
        poly_mons = sorted(list(set(poly_mons)),
                           key=monomial_key('grlex', states[::-1]))
        self.C = {monomial : dok_matrix((len(z1), len(z2)))
                  for monomial in poly_mons}

        for row in range(len(z1)):
            for column in range(len(z2)):
                poly = z1[row] * z2[column]
                monoms, coeffs = extract_monoms_and_coeffs(poly, states)
                for monomial, coefficient in zip(monoms, coeffs):
                    if coefficient != 0:
                        self.C[monomial][row, column] = coefficient

        if deriv:
            mat_mons = [sp.diff(m, states[deriv_index])
                        for m in matrix.monomials]
        else:
            mat_mons = matrix.monomials
        mat_mons_coeffs = [extract_monoms_and_coeffs(m, states)
                           for m in mat_mons]

        self.monomials = [x[0] * y for x, _ in mat_mons_coeffs
                          for y in poly_mons]
        self.monomials = sorted(list(set(self.monomials)),
                                key=monomial_key('grlex', states[::-1]))
        self.max_degree = self.monomial_degree(states)
        self.monomial_index = {monomial : [] for monomial in self.monomials}

        for mat_mon, mat_coeff in mat_mons_coeffs:
            for poly_mon in poly_mons:
                monomial = mat_mon[0] * poly_mon
                self.monomial_index[monomial].append((mat_mon[0], poly_mon,
                                                      mat_coeff[0]))

    def monomial_degree(self, states):
        ''' Returns the maximum (total) degree of the polynomial p in variables
        'states'

        Parameters:
        states (list[sympy.sybmol]) : the variables used for maximum degree
        calculation.

        Returns:
        _ (int) : maximum degree of monomial in p(x)
        '''
        return sp.Poly(sum(self.monomials), states).total_degree()

    def coefficient(self, monomial):
        ''' Returns the variable C_i given monomial mon(x)_i.

        Parameters:
        monomial (sympy.symbol) : the monomial mon(x)_i.

        Returns:
        _ (cvxpy.variable) : the variable C_i.
        '''
        if monomial in self.monomial_index:
            indeces = self.monomial_index[monomial]
            return cv.sum([c * cv.trace(self.matrix.variables[a].T @ self.C[b])
                           for a, b, c in indeces])
        else:
            return cv.Constant(0)
