import numpy as np
from numba import jit_module

from ..error_classes import *

def Cx(v, y):
    """
    Функция возвращающая коэффициент лобового сопротивления по значению
    скорости и высоты полета снаряда методом линейной интерполяции
    :param v: Скорость снаряда
    :param y: Высота полета
    :return: Значение коэф лобового сопротивления по закону 43 года
    """
    h_list = np.array([0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                       5000, 10000, 20000])
    a_list = np.array([340.29, 340.10, 339.91, 339.53, 339.14, 338.76, 338.38, 337.98, 337.6,
                       337.21, 336.82, 336.43, 320.54, 299.53, 295.07])
    a = np.interp(y, h_list, a_list)
    mah = v / a
    mah_list = np.array([0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                         1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                         2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6])
    cx_list = np.array([0.157, 0.158, 0.158, 0.160, 0.190, 0.325, 0.378, 0.385, 0.381, 0.371,
                        0.361, 0.351, 0.342, 0.332, 0.324, 0.316, 0.309, 0.303, 0.297,
                        0.292, 0.287, 0.283, 0.279, 0.277, 0.273, 0.270, 0.267, 0.265,
                        0.263, 0.263, 0.261, 0.260])
    koef = np.interp(mah, mah_list, cx_list)
    return koef


def external_bal_rs(dy, y, q, d, i43):
    """
    Фунция правых частей системы уравнений внешнй баллистики при аргументе t
    :param t: Значение аргумента t
    :param y0: Значения X, Y, V и Teta в виде массива
    :param q: Масса снаряда
    :param d: Калибр снаряда
    param: i43: Коэф формы снаряда по закону 43 года
    """
    g = 9.81
    dy[0] = y[2] * np.cos(y[3])  # Расчет приращения координаты X
    dy[1] = y[2] * np.sin(y[3])  # Расчет приращения координаты Y
    Hy = 1.2066 * ((20000 - y[1]) / (20000 + y[1]))
    Jt = -0.5 * i43 * Cx(y[2], y[1]) * Hy * (y[2] ** 2)  # Расчет силы лобового сопротивления
    dy[2] = ((Jt * ((np.pi * d ** 2) / 4)) - (q * g * np.sin(y[3]))) / q  # Расчет приращения скорости V
    dy[3] = -(g * np.cos(y[3])) / y[2]  # Расчет приращения угла teta


def dense_count_eb(V0, q, d, i43, theta, distance, tstep=1., tmax=1000.):
    '''
    Решение основной задачи внешней баллистики
    :param V0: Начальная скорость
    :param q: Масса снаряда
    :param d: Калибр
    :param i43: Коэф формы по закону 1943 года
    :param theta: Угол стрельбы
    :param distance: Максимальная дистанция
    :param tstep: Шаг по времени
    :param tmax: Максимальное время полета
    :return:
    '''
    ts = np.arange(0., tmax+tstep, tstep)
    ys = np.zeros((len(ts), 4))
    ys[0] = np.array([0., 0., V0, theta])
    K = np.zeros((4, 4))

    for i in range(1, len(ts)):
        external_bal_rs(K[0], ys[i-1], q, d, i43)
        external_bal_rs(K[1], ys[i-1] + K[0] * tstep / 2, q, d, i43)
        external_bal_rs(K[2], ys[i-1] + K[1] * tstep / 2, q, d, i43)
        external_bal_rs(K[3], ys[i-1] + K[2] * tstep, q, d, i43)
        ys[i] = ys[i-1] + tstep * (K[0] + 2 * K[1] + 2 * K[2] + K[3]) / 6

        if ys[i, 1] <= 0. or ys[i, 0] > distance:
            break

    ts = ts[:i+1]
    ys = ys[:i+1]

    if ys[-1, 1] < 0:
        ts[-1] = ts[-1] + (0. - ys[-2, 1]) * ((ts[-1]-ts[-2])/(ys[-1, 1]-ys[-2, 1]))
        ys[-1] = ys[-2] + (0. - ys[-2, 1]) * ((ys[-1]-ys[-2])/(ys[-1, 1]-ys[-2, 1]))

    if ys[-1, 0] > distance:
        ts[-1] = ts[-1] + (distance - ys[-2, 0]) * ((ts[-1]-ts[-2])/(ys[-1, 0]-ys[-2, 0]))
        ys[-1] = ys[-2] + (distance - ys[-2, 0]) * ((ys[-1]-ys[-2])/(ys[-1, 0]-ys[-2, 0]))

    ys[:, 3] = np.rad2deg(ys[:, 3])

    return ts, ys.T

def fast_count_eb(V0, q, d, i43, theta, distance, tstep=1., tmax=1000.):
    '''
    Решение основной задачи внешней баллистики
    :param V0: Начальная скорость
    :param q: Масса снаряда
    :param d: Калибр
    :param i43: Коэф формы по закону 1943 года
    :param theta: Угол стрельбы
    :param distance: Максимальная дистанция
    :param tstep: Шаг по времени
    :param tmax: Максимальное время полета
    :return:
    '''

    ys = np.array([
        [0., 0., V0, theta],
        [0., 0., V0, theta]
    ])

    t0 = 0.
    K = np.zeros((4, 4))
    while ys[1, 1] >= 0. and ys[1, 0] < distance:
        external_bal_rs(K[0], ys[1], q, d, i43)
        external_bal_rs(K[1], ys[1]+K[0]/2, q, d, i43)
        external_bal_rs(K[2], ys[1]+K[1]/2, q, d, i43)
        external_bal_rs(K[3], ys[1]+K[2], q, d, i43)

        ys[0] = ys[1]

        ys[1] += tstep*(K[0] + 2*K[1] + 2*K[2] + K[3])/6

        t0 += tstep
        if t0 > tmax:
            raise TooMuchTime()

    if ys[1, 1] < 0:
        ys[1] = ys[0] + (0. - ys[0, 1]) * ((ys[1]-ys[0])/(ys[1, 1]-ys[0, 1]))

    if ys[1, 0] > distance:
        ys[1] = ys[0] + (distance - ys[0, 0]) * ((ys[1]-ys[0])/(ys[1, 0]-ys[0, 0]))


    ys[1, 3] = np.rad2deg(ys[1, 3])

    return ys[1]

jit_module()


