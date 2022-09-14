"""
Implementation of numerical integration methods of ordinary differential equations
Pierre Puchaud
"""
from typing import Callable
import numpy as np
from numpy import linalg as la


def RK4(t, f, y0, args=()):
    """
    Runge-Kutta 4th order method

    Parameters
    ----------
    t : array_like
        time steps
    f : function
        function to be integrated
    y0 : array_like
        initial conditions of states

    Returns
    -------
    y : array_like
        states for each time step

    """
    n = len(t)
    y = np.zeros((len(y0), n))
    y[:, 0] = y0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        yi = np.squeeze(y[:, i])
        k1 = f(t[i], yi, *args)
        k2 = f(t[i] + h / 2., yi + k1 * h / 2., *args)
        k3 = f(t[i] + h / 2., yi + k2 * h / 2., *args)
        k4 = f(t[i] + h, yi + k3 * h, *args)
        y[:, i + 1] = yi + (h / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def RK8(t, f, y0, args=()):
    """
    Runge-Kutta 8th order method

    Parameters
    ----------
    t : array_like
        time steps
    f : function
        function to be integrated
    y0 : array_like
        initial conditions of states
    args : tuple, optional
        additional arguments to be passed to f

    Returns
    -------
    y : array_like
        states for each time step

    """
    n = len(t)
    y = np.zeros((len(y0), n))
    y[:, 0] = y0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        yi = np.squeeze(y[:, i])
        k_1 = f(t[i], yi)
        k_2 = f(t[i] + h * (4 / 27), yi + (h * 4 / 27) * k_1)
        k_3 = f(t[i] + h * (2 / 9), yi + (h / 18) * (k_1 + 3 * k_2))
        k_4 = f(t[i] + h * (1 / 3), yi + (h / 12) * (k_1 + 3 * k_3))
        k_5 = f(t[i] + h * (1 / 2), yi + (h / 8) * (k_1 + 3 * k_4))
        k_6 = f(t[i] + h * (2 / 3), yi + (h / 54) * (13 * k_1 - 27 * k_3 + 42 * k_4 + 8 * k_5))
        k_7 = f(t[i] + h * (1 / 6), yi + (h / 4320) * (389 * k_1 - 54 * k_3 + 966 * k_4 - 824 * k_5 + 243 * k_6))
        k_8 = f(t[i] + h, yi + (h / 20) * (-234 * k_1 + 81 * k_3 - 1164 * k_4 + 656 * k_5 - 122 * k_6 + 800 * k_7))
        k_9 = f(t[i] + h * (5 / 6),
                yi + (h / 288) * (-127 * k_1 + 18 * k_3 - 678 * k_4 + 456 * k_5 - 9 * k_6 + 576 * k_7 + 4 * k_8))
        k_10 = f(t[i] + h, yi + (h / 820) * (
                1481 * k_1 - 81 * k_3 + 7104 * k_4 - 3376 * k_5 + 72 * k_6 - 5040 * k_7 - 60 * k_8 + 720 * k_9))
        yii = yi + h / 840 * (41 * k_1 + 27 * k_4 + 272 * k_5 + 27 * k_6 + 216 * k_7 + 216 * k_9 + 41 * k_10)
        y[:, i + 1] = yii
    return y


def IRK(t, f, y0, order: int = 4):
    """
    Implicit Runge-Kutta method

    Parameters
    ----------
    t : array_like
        time steps
    f : function
        function to be integrated
    y0 : array_like
        initial conditions of states
    order : int, optional
        order of the implicit runge-kutta (default is 4)

    Returns
    -------
    y : array_like
        states for each time step

    """
    n = len(t)
    N = len(y0)
    y = np.zeros((N, n))
    y[:, 0] = y0

    tol = 1e-4
    # IRK method (Gauss-legendre order)
    A, b, c, s = GLD(order)

    for ii in range(n - 1):
        tt = t[ii]
        h = t[ii + 1] - t[ii]
        yi = np.expand_dims(y[:, ii], axis=1)
        # Calculate stage values
        e = np.ones((s, 1))
        Y = np.kron(e, yi)
        Jac = jacobian(f, yi, delta=1e-8)  # numerical jacobian
        J = np.eye(N * s) - h * np.matmul(np.kron(A, np.eye(N)), np.kron(np.eye(s), Jac))
        Jinv = -la.inv(J)
        err = 1
        ct = 1
        if ii == 12:
            print("hey")
        while err > tol:
            # F(tt, Y)
            Fty = np.array([])
            for i_s in range(s):
                fty = f(tt + c[i_s] * h, np.squeeze(Y[range(0 + N * i_s, N * (1 + i_s))]))
                Fty = np.concatenate((Fty, fty))
            Fty = np.expand_dims(Fty, axis=1)

            G = Y - np.kron(e, yi) - np.matmul(h * np.kron(A, np.eye(N)), Fty)
            DeltaY = np.matmul(Jinv, G)
            Y = Y + DeltaY
            err = la.norm(DeltaY)
            # print("Fty", Fty, "G", G, "DeltaY", DeltaY, "err", err)
            # print(err)
            if err != err:
                raise ValueError('Newton Algorithm diverged with irk')
            ct = ct + 1
            # if ct > 10:
            #     print("node",ii)
            #     print("error DeltaY", err)
            #     break
        # update
        print(ii)
        yii = yi + h * np.matmul(np.kron(b.T, np.eye(N)), Fty)
        if yii[0].item() == np.nan:
            print('nan')
        y[:, ii + 1] = np.squeeze(yii)

    return y


def GLD(order):
    """
    Gauss-Legendre coefficients for IRK method

    Parameters
    ----------
    order: int
        order of the IRK method

    Returns
    -------
    A: array_like
        A[i, j] is the coefficient of the j-th stage in the i-th stage
    b: array_like
        b[i] is the coefficient of the i-th stage in the i-th stage
    c: array_like
        c[i] is the coefficient of the i-th stage in the i-th stage
    s: int
        number of stages

    """
    if order == 4:
        A = np.array([[1 / 4, 1 / 4 - np.sqrt(3) / 6], [1 / 4 + np.sqrt(3) / 6, 1 / 4]])
        b = np.array([[1 / 2], [1 / 2]])
        c = np.array([[1 / 2 - np.sqrt(3) / 6], [1 / 2 + np.sqrt(3) / 6]])
        s = 2
    elif order == 6:
        A = np.array([[5 / 36, 2 / 9 - np.sqrt(15) / 15, 5 / 36 - np.sqrt(15) / 30],
                      [5 / 36 + np.sqrt(15) / 24, 2 / 9, 5 / 36 - np.sqrt(15) / 24],
                      [5 / 36 + np.sqrt(15) / 30, 2 / 9 + np.sqrt(15) / 15, 5 / 36]])
        b = np.array([[5 / 18], [4 / 9], [5 / 18]])
        c = np.array([[1 / 2 - np.sqrt(15) / 10], [1 / 2], [1 / 2 + np.sqrt(15) / 10]])
        s = 3
    else:
        RuntimeError(f"{order} order is not implemented. GLD is implemented for order 4 and 6 only.")
    return A, b, c, s


def jacobian(func: Callable, x, delta=1e-3):
    """
    Numerical jacobian of a function

    Parameters
    ----------
    func : Callable
        function to be differentiated
    x : array_like
        input vector
    delta : float, optional
        step size (default is 1e-3)

    Returns
    -------
    J : array_like
        jacobian of the function

    """
    f = func
    x = np.squeeze(x)
    nrow = len(f(0, x))
    ncol = len(x)
    J = np.zeros(nrow * ncol)
    J = J.reshape(nrow, ncol)
    for i in range(nrow):
        for j in range(ncol):
            ej = np.zeros(ncol)
            ej[j] = 1
            dij = (f(0, x + delta * ej)[i] - f(0, x - delta * ej)[i]) / (2 * delta)
            J[i, j] = dij
    return J
