from .BallisticsClasses import *

def precompile():
    artsys = ArtSystem(name='2А42', d=.03, S=0.000735299, W0=0.125E-3, l_d=1., khi=1, Kf=1.136)
    shell = Shell('30ка', 0.03, 0.389, 1.)

    powders = [Powder(name='6/7', omega=0.05, rho=1.6e3, f_powd=988e3, Ti=2800., Jk=343.8e3, alpha=1.038e-3, teta=0.236,
                      Zk=1.53, kappa1=0.239, lambd1=2.26, mu1=0., kappa2=0.835, lambd2=-0.943, mu2=0., gamma_f=3e-4,
                      gamma_Jk=0.0016)]

    bal_prob = FastBallisticsSolver(
        artsys, powders, shell,
        shot_params=ShootingParameters(5., 1000.)
    )
    bal_prob.solve_ib(tstep=1e-4)
    bal_prob.solve_eb(tstep=2.)
    print('ballistics package is precompiled successfully')

precompile()

__all__ = [
    'ArtSystem',
    'Shell',
    'Powder',
    'LoadParams',
    'ShootingParameters',
    'FastBallisticsSolver',
    'DenseBallisticsSolver'
]