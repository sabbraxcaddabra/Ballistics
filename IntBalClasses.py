from dataclasses import dataclass
from collections import namedtuple


__all__ = ["ArtSystem", "Powder", "BallisticsProblem"]


@dataclass
class ArtSystem:
    # Датакласс для данных об артиллерийской системе
    name: str  # Наименование артиллерийской системы
    d: float  # Приведенная площадь канала ствола
    q: float  # Масса снаряда
    S: float  # Калибр орудия
    W0: float  # Объем зарядной каморы
    l_d: float  # Полный путь снаряда
    l_k: float  # Длина зарядной каморы
    l0: float  # Приведенная длина зарядной каморы
    Kf: float  # Коэффициент слухоцкого

    def __str__(self):
        return f"Арт.система {self.name}, калибр: {self.d*1e3} мм"


@dataclass
class Powder:
    # Датакласс для данных о порохе
    name: str  # Марка пороха
    omega: float  # Масса метательного заряда
    rho: float  # Плотность пороха
    f_powd: float  # Сила пороха
    Ti: float  # Температура горения пороха
    Jk: float  # Конечный импульс пороховых газов
    alpha: float  # Коволюм
    teta: float  # Параметр расширения
    Zk: float  # Относительная толщина горящего свода, соответствующая концу горения
    #PsiS: float # Относительная масса сгоревшего пороха к моменту распада
    kappa1: float  # 1-я, 2-я и 3-я хар-ки формы пороховых элементов до распада
    lambd1: float
    mu1: float
    kappa2: float  # 1-я, 2-я и 3-я характеристики формы пороховых элементов после распада
    lambd2: float
    mu2: float
    gamma_f: float # Температурная поправка на силу пороха
    gamma_Jk: float # Температурная поправка на конечный импульс

    def __str__(self):
        return f"Марка пороха: {self.name}, масса: {self.omega:.4g}, конечный импульс: {self.Jk*1e-3} кПа*с"

    def __repr__(self):
        return f"Марка пороха: {self.name}, масса: {self.omega:.4g}, конечный импульс: {self.Jk*1e-3} кПа*с"

    @classmethod
    def from_data_string(cls, string: str):
        string_list = string.strip().split(' ')
        data_list = list(map(float, string_list[1:]))
        return cls(string_list[0], 0.0, *data_list)

class EBProblem:
    '''
    Класс-решатель задач внешней баллистики
    '''
    def __init__(self, theta=45., i43=1.):
        self.theta = theta
        self.i43 = i43

    def solve(self, v0, q):
        pass

class IBProblem:
    '''
    Класс-решатель задач внутренней баллистики
    '''
    igniter_f = 240e3
    igniter_teta = 0.22
    igniter_Ti = 2427.

    Igniter = namedtuple('Igniter', [
        'fs',
        'sum1',
        'sum2'
    ])
    Powder_ = namedtuple('Powd', [
        'omega',
        'rho',
        'f_powd',
        'Ti',
        'Jk',
        'alpha',
        'teta',
        'Zk',
        'kappa1',
        'lambd1',
        'mu1',
        'kappa2',
        'lambd2',
        'mu2'
    ])

    def __init__(self, barl, P0, T0=15., charge=[], PV=None, igniter_mass=None):
        self.barl = barl  # Арт.система для задачи
        self.charge = charge  # Метательный заряд
        self.T0 = T0 # Температура метательного заряда
        self.P0 = P0  # Давление форсирования
        if igniter_mass:
            self.igniter_mass = igniter_mass # Масса воспламенителя
        else:
            self.PV = PV-1e5 # Давление воспламенителя

    def add_powder(self, powder: Powder) -> None:
        self.charge.append(powder)

    def create_params_tuple(self) -> tuple:

        if not hasattr(self, 'igniter_mass'):
            self.igniter_mass = self.PV * (self.barl.W0 - sum(powd.omega / powd.rho for powd in self.charge)) / self.igniter_f

        fs = self.igniter_mass*self.igniter_f
        sum1 = fs/self.igniter_Ti
        sum2 = sum1/self.igniter_teta

        igniter = self.Igniter(fs, sum1, sum2)

        # Метод для создания исходных данных
        params = [
            self.P0,
            igniter,
            50e6 ** 0.25,
            self.barl.S,
            self.barl.W0,
            self.barl.l_k,
            self.barl.l0,
            sum(powd.omega for powd in self.charge),
            self.barl.Kf * self.barl.q,
            self.barl.l_d
        ]
        powders = []
        for powder in self.charge:
            tmp = self.Powder_(
                powder.omega,
                powder.rho,
                powder.f_powd * (1. + powder.gamma_f * (self.T0 - 15.)),
                powder.Ti,
                powder.Jk * (1. - powder.gamma_Jk * (self.T0 - 15.)),
                powder.alpha,
                powder.teta,
                powder.Zk,
                powder.kappa1,
                powder.lambd1,
                powder.mu1,
                powder.kappa2,
                powder.lambd2,
                powder.mu2
            )
            powders.append(tmp)
        params.append(tuple(powders))
        return tuple(params)