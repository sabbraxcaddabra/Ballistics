import numpy as np
from numba import njit

from ErrorClasses import *

def runge_kutta4(f, y0, t0, t_end, tau, args=tuple(), stopfunc=None):

    if not stopfunc:
        stopfunc = lambda t, y: t <= t_end

    ts = np.arange(t0, t_end+tau, tau)
    ys = np.zeros((len(ts), len(y0)))
    K = np.zeros((4, len(y0)))
    ys[0] = y0

    for i in range(1, len(ys)):
        f(K[0], ys[i-1], *args)
        f(K[1], ys[i-1] + tau * K[0] / 2, *args)
        f(K[2], ys[i-1] + tau * K[1] / 2, *args)
        f(K[3], ys[i-1] + tau * K[2], *args)
        ys[i] = ys[i-1] + tau*(K[0] + 2*K[1] + 2*K[2] + K[3])/6

        if stopfunc(ts[i], ys[i]):
            break

    return ts[:i], ys[:i].T


@njit
def P(y, igniter, lambda_khi, S, W0, qfi, omega_sum, psis, powders):
    thet = theta(psis, igniter, powders)
    fs = igniter.fs
    for i, powder in enumerate(powders):
        fs += powder.f_powd * powder.omega * psis[i]
        W0 -= powder.omega * ((1. - psis[i]) / powder.rho + psis[i] * powder.alpha)
    fs -= thet * y[0] ** 2 * (qfi / 2 + lambda_khi * omega_sum / 6.)
    W0 += y[1] * S
    if W0 < 0.:
        raise TooMuchPowderError()
    p_mean = 1e5 + fs / W0
    p_sn = (p_mean/(1 + (1/3) * (lambda_khi*omega_sum)/qfi))*(y[0] > 0.) + (y[0] == 0.)*p_mean
    p_kn = (p_sn*(1 + 0.5*lambda_khi*omega_sum/qfi))*(y[0] > 0.) + (y[0] == 0.)*p_mean

    return p_mean, p_sn, p_kn

@njit
def theta(psis, igniter, powders):
    num = igniter.num
    denum = igniter.denum
    for i, powder in enumerate(powders):
        num += powder.f_powd * powder.omega * psis[i] / powder.Ti
        denum += powder.f_powd * powder.omega * psis[i] / (powder.Ti * powder.teta)
    if denum != 0:
        return num / denum
    else:
        return 0.4

@njit
def psi(z, zk, kappa1, lambd1, mu1, kappa2, lambd2, mu2):
    if z < 1:
        return kappa1*z*(1 + lambd1*z + mu1*z**2)
    elif 1 <= z <= zk:
        z1 = z - 1
        psiS = kappa1 + kappa1*lambd1 + kappa1*mu1
        return psiS + kappa2*z1*(1 + lambd2*z1 + mu2*z1**2)
    else:
        return 1

@njit
def int_bal_rs(dy, y, psis, P0, igniter, k50, S, W0, l_k, l_ps, omega_sum, qfi, powders):
    """
    Функция правых частей системы уравнений внутреней баллистики при аргументе t
    """
    for i, powder in enumerate(powders):
        psis[i] = psi(y[2 + i], *powder[7:])

    lambda_khi = (y[1] + l_k)/(y[1] + l_ps)

    p_mean, p_sn, p_kn = P(y, igniter, lambda_khi, S, W0, qfi, omega_sum, psis, powders)

    if y[0] == 0. and p_mean < P0:
        dy[0] = 0.
        dy[1] = 0.
    else:
        dy[0] = p_sn*S/qfi
        dy[1] = y[0]
    for i, powder in enumerate(powders):
        if p_mean <= 50e6:
            dy[2+i] = ((k50*p_mean**0.75)/powder.Jk) * (y[2+i] < powder.Zk)
        else:
            dy[2+i] = (p_mean/powder.Jk) * (y[2+i] < powder.Zk)


def count_ib(P0, igniter, k50, S, W0, l_k, l_ps, omega_sum, qfi, l_d, powders, tmax = 1. , tstep = 1e-5):
    n_powd = len(powders)
    y = np.zeros(2+n_powd)
    psis = np.zeros(n_powd)

    args = (psis, P0, igniter, k50, S, W0, l_k, l_ps, omega_sum, qfi, powders)
    ts, ys = runge_kutta4(int_bal_rs, y, t0=0., t_end=tmax, tau=tstep, args=args, stopfunc=lambda t, y: y[1] >= l_d)

    psis = np.zeros((len(powders), ys.shape[1]))

    for i, powd in enumerate(powders):
        psis[i, :] = np.array([psi(ys[2+i][k], *powd[7:]) for k in range(ys.shape[1])])

    lambda_khi = (ys[1] + l_k)/(ys[1] + l_ps)

    p_mean, p_sn, p_kn = np.zeros(ys.shape[1]), np.zeros(ys.shape[1]), np.zeros(ys.shape[1])

    for i in range(ys.T.shape[0]):
        p_mean[i], p_sn[i], p_kn[i] = P(ys.T[i], igniter, lambda_khi[i], S, W0, qfi, omega_sum, psis.T[i], powders)

    lk_indexes = np.argmax(psis >= 1., axis=1)

    ys[2:] = psis

    return ys, p_mean, p_sn, p_kn, lk_indexes
