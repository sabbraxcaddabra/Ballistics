import cffi
import numpy as np

from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import namedtuple


__all__ = [
    'TooMuchPowderError', 'TooMuchTimeError',
    'ArtSystem', 'Powder', 'Shell', 'LoadParams',
    'ShootingParameters', 'BallisticsProblem'
]

class TooMuchPowderError(Exception):
    def __str__(self):
        return "Слишком много пороха"

class TooMuchTimeError(Exception):
    def __str__(self):
        return "Превышено максимальное время процесса"

@dataclass
class ArtSystem:
    # Датакласс для данных об артиллерийской системе
    name: str  # Наименование артиллерийской системы
    d: float  # Приведенная площадь канала ствола
    S: float  # Калибр орудия
    W0: float  # Объем зарядной каморы
    l_d: float  # Полный путь снаряда
    khi: float # Коэффициент уширения каморы
    Kf: float  # Коэффициент Слухоцкого

    def __str__(self):
        return f"Арт.система {self.name}, калибр: {self.d * 1e3} мм"

    @classmethod
    def from_data_string(cls, string: str):
        string_list = string.strip().split(' ')
        data_list = list(map(float, string_list[1:]))
        return cls(string_list[0], *data_list)

    def as_dict(self):
        self_dict = {
            'name': self.name,
            'd': self.d,
            'S': self.S,
            'W0': self.W0,
            'l_d': self.l_d,
            'khi': self.khi,
            'Kf': self.Kf
        }
        return self_dict

@dataclass
class Shell:
    name: str  # Индекс снаряда
    d: float  # Калибр
    q: float  # Масса снаряда
    i43: float  # Коэф. формы по закону 43 года
    alpha: float = 0.  # Коэффициент наполнения

    @classmethod
    def from_data_string(cls, string: str):
        string_list = string.strip().split(' ')
        data_list = list(map(float, string_list[1:]))
        return cls(string_list[0], *data_list)

    def as_dict(self):
        self_dict = {
            'name': self.name,
            'd': self.d,
            'q': self.q,
            'i43': self.i43
        }
        return self_dict

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
    # PsiS: float # Относительная масса сгоревшего пороха к моменту распада
    kappa1: float  # 1-я, 2-я и 3-я хар-ки формы пороховых элементов до распада
    lambd1: float
    mu1: float
    kappa2: float  # 1-я, 2-я и 3-я характеристики формы пороховых элементов после распада
    lambd2: float
    mu2: float
    gamma_f: float  # Температурная поправка на силу пороха
    gamma_Jk: float  # Температурная поправка на конечный импульс

    def __str__(self):
        return f"Марка пороха: {self.name}, масса: {self.omega:.4g}, конечный импульс: {self.Jk * 1e-3} кПа*с"

    def __repr__(self):
        return f"Марка пороха: {self.name}, масса: {self.omega:.4g}, конечный импульс: {self.Jk * 1e-3} кПа*с"

    @classmethod
    def from_data_string(cls, string: str):
        string_list = string.strip().split(' ')
        data_list = list(map(float, string_list[1:]))
        return cls(string_list[0], 0.0, *data_list)

    def as_dict(self):
        self_dict = {
            'name': self.name,
            'omega': self.omega,
            'rho': self.rho,
            'f_powd': self.f_powd,
            'Ti': self.Ti,
            'Jk': self.Jk,
            'alpha': self.alpha,
            'teta': self.teta,
            'Zk': self.Zk,
            'kappa1': self.kappa1,
            'lambd1': self.lambd1,
            'mu1': self.mu1,
            'kappa2': self.kappa2,
            'lambd2': self.lambd2,
            'mu2': self.mu2,
            'gamma_f': self.gamma_f,
            'gamma_Jk': self.gamma_Jk,
        }
        return self_dict


class LoadParams:
    # Класс хранящий информацию о параметрах заряжания
    def __init__(self, P0, T0=15., PV=4e5, ig_mass=0.0):
        self.P0 = P0  # Давление форсирования
        self.T0 = T0  # Температура метательного заряда
        self.ig_mass = ig_mass  # Масса воспламенителя
        self.PV = PV - 1e5  # Давление воспламенителя

    def as_dict(self):
        self_dict = {
            'P0': self.P0,
            'T0': self.T0,
            'ig_mass': self.ig_mass,
            'PV': self.PV
        }
        return self_dict


@dataclass
class ShootingParameters:
    theta_angle: float = 45.  # Угол стрельбы
    distance: float = 150e3  # Макс. дистанция стрельбы

    def as_dict(self):
        self_dict = {
            'theta_angle': self.theta_angle,
            'distance': self.distance
        }

        return self_dict

class BallisticsProblem(ABC):
    v0 = 0.
    pmax = 1e5
    psi_sum = 0.
    eta_k = 0.
    Lmax = 0.
    vend = 0.

    def __init__(self, barl, charge, shell, load_params=LoadParams(30e6, PV=4e5), shot_params=ShootingParameters()):
        self.barl = barl  # Орудие
        self.charge = charge  # Массив порохов(метательный заряд)
        self.shell = shell  # Снаряд
        self.load_params = load_params  # Параметры заряжания(внутреннаяя баллистика)
        self.shot_params = shot_params  # Параметры стральбы(внешняя баллистика)

        self.ffi, self.bal_lib = self.load_lib()

    def load_lib(self):
        ffi = cffi.FFI()
        ffi.cdef(
            '''
            typedef struct barrel{\n
            double d;\n
            double q;\n
            double s;\n
            double w0;\n
            double l_d;\n
            double l_k;\n
            double l0;\n
            double kf;\n
            } barrel;
            '''
        )

        ffi.cdef(
            '''
            typedef struct ibproblem{\n
            barrel artsystem;\n
            double p0;\n
            double pv;\n
            double ig_mass;\n
            double t0;\n
            double v0;\n
            double p_av_max;\n
            double p_sn_max;\n
            double p_kn_max;\n
            double psi_sum;\n
            double eta_k;\n
            int status;
            } ibproblem;
            '''
        )

        ffi.cdef(
            '''
            typedef struct powder{\n
            double om;\n
            double ro;\n
            double f_powd;\n
            double ti;\n
            double jk;\n
            double alpha;\n
            double theta;\n
            double zk;\n
            double kappa1;\n
            double lambda1;\n
            double mu1;\n
            double kappa2;\n
            double lambda2;\n
            double mu2;\n
            double gamma_f;\n
            double gamma_jk;\n
            } powder;
            '''
        )

        ffi.cdef(
            '''
            void set_problem(\n
            ibproblem *problem,\n
            double p0,\n
            double pv,\n
            double ig_mass,\n
            double t0\n
            );
            '''
        )

        ffi.cdef(
            '''
            void set_barrel(\n
            ibproblem *problem,\n
            double d,\n
            double q,\n
            double s,\n
            double w0,\
            double l_d,\n
            double l_k,\n
            double l0,\n
            double kf\n
            );
            '''
        )

        ffi.cdef(
            '''
            void add_powder(\n
            int n_powd,\n
            powder *powd_array,\n
            int i,\n
            double om,\n
            double ro,\n
            double f_powd,\n
            double ti,\n
            double jk,\n
            double alpha,\n
            double theta,\n
            double zk,\n
            double kappa1,\n
            double lambda1,\n
            double mu1,\n
            double kappa2,\n
            double lambda2,\n
            double mu2,\n
            double gamma_f,\n
            double gamma_jk\n
            );
            '''
        )

        ffi.cdef(
            '''
            void count_ib(ibproblem *ibp, int n_powd, powder *charge, double tstep, double tend);
            '''
        )

        ffi.cdef(
            '''
            void dense_count_ib(ibproblem *ibp, int n_powd, powder *charge, double tstep, double tend,\n
            int *n_tsteps, double *y_array, double *pressure_array);
            '''
        )

        ffi.cdef(
            '''
            void count_eb(double *y0, double d, double q, double i43,\n
            double max_distance, double tstep, double tmax);
            '''
        )

        ffi.cdef(
            '''
            void dense_count_eb(double *y_array, double d, double q, double i43,\n
            double max_distance, double tstep, double tmax, int *n_tsteps);
            '''
        )

        bal_lib = ffi.dlopen('fortran/ballib.dll')
        return ffi, bal_lib

    @abstractmethod
    def solve_ib(self, tstep=1e-5, tmax=1.):
        pass

    @abstractmethod
    def solve_eb(self, tstep=1., tmax=1000.):
        pass

class FastBallisticsSolver(BallisticsProblem):

    def solve_ib(self, tstep=1e-5, tmax=1.):

        n_powd = len(self.charge)

        new_problem = self.ffi.new('ibproblem *')

        powd_array = self.ffi.new(f'powder[{n_powd}]')

        self.bal_lib.set_problem(new_problem,
                                 self.load_params.P0,
                                 self.load_params.PV,
                                 self.load_params.ig_mass,
                                 self.load_params.T0)

        self.bal_lib.set_barrel(new_problem,
                                self.barl.d,
                                self.shell.q,
                                self.barl.S,
                                self.barl.W0,
                                self.barl.l_d,
                                self.barl.W0 / (self.barl.S * self.barl.khi),
                                self.barl.W0 / self.barl.S,
                                self.barl.Kf
                                )

        for i, powd in enumerate(self.charge, start=1):
            self.bal_lib.add_powder(
                n_powd, powd_array, i,
                powd.omega, powd.rho, powd.f_powd, powd.Ti,
                powd.Jk, powd.alpha, powd.teta,
                powd.Zk, powd.kappa1, powd.lambd1, powd.mu1, powd.kappa2, powd.lambd2, powd.mu2,
                powd.gamma_f, powd.gamma_Jk
            )

        self.bal_lib.count_ib(new_problem, n_powd, powd_array, tstep, tmax)

        self.v0 = new_problem[0].v0
        self.pmax = new_problem[0].p_av_max
        self.psi_sum = new_problem[0].psi_sum
        self.eta_k = new_problem[0].eta_k

        calc_status = new_problem[0].status
        if calc_status == 0:
            return self.v0, self.pmax, self.psi_sum, self.eta_k
        elif calc_status == 1:
            raise TooMuchPowderError()
        elif calc_status == 2:
            raise TooMuchTimeError()

    def solve_eb(self, tstep=1., tmax=1000.):

        y0 = np.array([0., 0., self.v0, np.deg2rad(self.shot_params.theta_angle)], dtype=np.float64, order='F')

        y0_ptr = self.ffi.cast("double*", y0.__array_interface__['data'][0])

        self.bal_lib.count_eb(
            y0_ptr, self.shell.d, self.shell.q, self.shell.i43,
            self.shot_params.distance, tstep, tmax
        )

        return y0[0], y0[1], y0[2], y0[3]

class DenseBallisticsSolver(BallisticsProblem):

    def solve_eb(self, tstep=1., tmax=1000.):

        n_tsteps = self.ffi.new('int *')
        n_tsteps[0] = int(tmax / tstep)
        #
        y_array = np.empty((4, n_tsteps[0]), dtype=np.float64, order='F')
        y_array[:, 0] = [0., 0., self.v0, np.deg2rad(self.shot_params.theta_angle)]

        y_array_ptr = self.ffi.cast("double*", y_array.__array_interface__['data'][0])

        self.bal_lib.dense_count_eb(
            y_array_ptr, self.shell.d, self.shell.q, self.shell.i43,
            self.shot_params.distance, tstep, tmax, n_tsteps
        )

        ts = np.linspace(0., tstep*n_tsteps[0], n_tsteps[0])
        y_array = y_array[:, :n_tsteps[0]]
        y_array[3] = np.rad2deg(y_array[3])

        return ts, y_array

    def solve_ib(self, tstep=1e-5, tmax=1.):

        n_powd = len(self.charge)

        new_problem = self.ffi.new('ibproblem *')

        powd_array = self.ffi.new(f'powder[{n_powd}]')

        self.bal_lib.set_problem(new_problem,
                                 self.load_params.P0,
                                 self.load_params.PV,
                                 self.load_params.ig_mass,
                                 self.load_params.T0)

        self.bal_lib.set_barrel(new_problem,
                                self.barl.d,
                                self.shell.q,
                                self.barl.S,
                                self.barl.W0,
                                self.barl.l_d,
                                self.barl.W0 / (self.barl.S * self.barl.khi),
                                self.barl.W0 / self.barl.S,
                                self.barl.Kf
                                )

        for i, powd in enumerate(self.charge, start=1):
            self.bal_lib.add_powder(
                n_powd, powd_array, i,
                powd.omega, powd.rho, powd.f_powd, powd.Ti,
                powd.Jk, powd.alpha, powd.teta,
                powd.Zk, powd.kappa1, powd.lambd1, powd.mu1, powd.kappa2, powd.lambd2, powd.mu2,
                powd.gamma_f, powd.gamma_Jk
            )

        n_tsteps = self.ffi.new('int *')
        n_tsteps[0] = int(tmax/tstep)
        y_array = np.empty((2+n_powd, n_tsteps[0]), dtype=np.float64, order='F')
        pressure_array = np.empty((3, n_tsteps[0]), dtype=np.float64, order='F')

        y_array_ = self.ffi.cast("double*", y_array.__array_interface__['data'][0])
        pressure_array_ = self.ffi.cast("double*", pressure_array.__array_interface__['data'][0])

        self.bal_lib.dense_count_ib(new_problem, n_powd, powd_array, tstep, tmax,
                                    n_tsteps, y_array_, pressure_array_)

        self.v0 = new_problem[0].v0
        self.pmax = new_problem[0].p_av_max
        self.psi_sum = new_problem[0].psi_sum
        self.eta_k = new_problem[0].eta_k

        calc_status = new_problem[0].status
        if calc_status == 0:

            y_array = y_array[:, :n_tsteps[0]]
            pressure_array = pressure_array[:, :n_tsteps[0]]
            ts = np.linspace(0., tstep * n_tsteps[0], n_tsteps[0])
            lk_indexes = np.argmin(1.0 - y_array[2:], axis=1)
            return ts, y_array, pressure_array[0], pressure_array[1], pressure_array[2], lk_indexes

        elif calc_status == 1:
            raise TooMuchPowderError()
        elif calc_status == 2:
            raise TooMuchTimeError()

if __name__ == '__main__':
    artsys = ArtSystem(name='2А42', d=.03, S=0.000735299, W0=0.125E-3, l_d=2.263, khi=1.0, Kf=1.136)
    shell = Shell('30ка', 0.03, 0.389, 1.)

    powders = [Powder(name='6/7', omega=.12, rho=1.6e3, f_powd=988e3, Ti=2800., Jk=343.8e3, alpha=1.038e-3, teta=0.236,
                      Zk=1.53, kappa1=0.239, lambd1=2.26, mu1=0., kappa2=0.835, lambd2=-0.943, mu2=0., gamma_f=3e-4,
                      gamma_Jk=0.0016)
               ]

    bal_prob = DenseBallisticsSolver(
        artsys, powders, shell
    )

    bal_prob.solve_ib(tstep=1e-7)
    bal_prob.solve_eb()



