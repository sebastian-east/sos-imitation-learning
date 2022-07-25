import sympy as sp
from sympy import S
import numpy as np
import pytest
from sos.sos import n_monomials, monomial_generator, zero_rows_B, \
                          extract_monomials, extract_monoms_and_coeffs, \
                          einsum1, einsum2, MatrixVar

def test_monomials():
    #TODO: update these tests to include minimum order
    x1, x2, x3 = sp.symbols('x1, x2, x3')
    states = [x1, x2, x3]
    for max_degree in range(1, 10):
        n_monoms = n_monomials(len(states), max_degree)
        monoms = monomial_generator(states, max_degree, 0)
        assert n_monoms == len(monoms)

    monoms = monomial_generator(states, 1, 0)
    target = [S.One, x1, x2, x3]
    for m, t in zip(monoms, target):
        assert m == t
    '''
    monoms = monomial_generator(states, 3, 1)
    target = [x1**2, x1*x2, x2**2, x1*x3, x2*x3, x3**2, x1**3, x1**2*x2,
              x1*x2**2, x2**3, x1**2*x3, x1*x2*x3, x2**2*x3, x1*x3**2,
              x2*x3**2, x3**3]
    for m, t in zip(monoms, target):
        assert m == t
    '''

def test_zero_rows():
    x1, x2 = sp.symbols('x1, x2')
    assert zero_rows_B(sp.Matrix([[1]])) == []
    assert zero_rows_B(sp.Matrix([[x1]])) == []
    assert zero_rows_B(sp.Matrix([[0]])) == [0]
    assert zero_rows_B(sp.Matrix([[x1, x2**2]])) == []
    assert zero_rows_B(sp.Matrix([[x1, x2**2], [0, 0]])) == [1]
    assert zero_rows_B(sp.Matrix([[x1, x2**2], [0, 0], [x1**2, 0]])) == [1]
    assert zero_rows_B(sp.Matrix([[0, 0], [0, 0], [x1**2, 0]])) == [0, 1]
    assert zero_rows_B(sp.Matrix([[0, 0], [0, 0], [0, 0]])) == [0, 1, 2]

def test_extract_monomials():
    x, y = sp.symbols('x, y')
    assert extract_monomials(sp.Matrix([[x]]), [x, y]) == {x}
    assert extract_monomials(sp.Matrix([[x], [y]]), [x, y]) == {x, y}
    assert extract_monomials(sp.Matrix([[x], [y**2], [x**2]]), [x, y]) == {x, y**2, x**2}
    assert extract_monomials(sp.Matrix([[x], [y**2], [x**2 + y]]), [x, y]) == {x, y, y**2, x**2}

def test_extract_monomials_and_coeffs():
    x, y, z = sp.symbols('x, y, z')

    mt, ct = [x, x*y, z**2], [1, 2, 3] # these need to be in lexigraphical order
    polynomial = sum([a*b for a, b in zip(mt, ct)])
    monomials, coeffs = extract_monoms_and_coeffs(polynomial, [x, y, z])
    assert len(mt) == len(monomials)
    assert len(ct) == len(coeffs)
    assert set(mt) == set(monomials)
    for m, c in zip(mt, ct):
        index = monomials.index(m)
        assert coeffs[index] == c

    mt, ct = [x, x*y, z**2, x*z, x*y**10], [1., 2., 3., 0.1, 31.] # these need to be in lexigraphical order
    polynomial = sum([a*b for a, b in zip(mt, ct)])
    monomials, coeffs = extract_monoms_and_coeffs(polynomial, [x, y, z])
    assert len(mt) == len(monomials)
    assert len(ct) == len(coeffs)
    assert set(mt) == set(monomials)
    for m, c in zip(mt, ct):
        index = monomials.index(m)
        assert coeffs[index] == c

def test_matrix_var():
    x, y = sp.symbols('x, y')
    P = MatrixVar(2, 3, [x, y], order=2)
    assert P.monomials == monomial_generator([x, y], 2, 0)
    assert len(P.variables) == n_monomials(len([x, y]), 2)
    for mon in P.variables:
        assert P.variables[mon].shape == (2, 3)

def test_einsum1():
    n, m, p = 2, 3, 4
    A = {}
    B = {}
    for i in range(5):
        A[i] = np.random.rand(n, m)
        B[i] = np.random.rand(p)
    C = einsum1(A, B)
    for i in range(p):
        assert np.all(np.isclose(C[:, :, i], sum([A[j] * B[j][i] for j in A.keys()])))

def test_einsum2():
    np.random.seed(0)
    for _ in range(5):
        A = np.random.rand(2, 3, 5)
        B = np.random.rand(3, 3)
        C = np.random.rand(3, 5)
        D = einsum2(A, B, C)
        for i in range(A.shape[2]):
            assert np.all(np.isclose(D[:, i], A[:, :, i] @ B @ C[:, i]))
