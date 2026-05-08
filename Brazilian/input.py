from petsc4py.PETSc import ScalarType
from mpi4py import MPI
import numpy as np

def stored_energy_tension(sts,E):
    return sts**2/2/E

def hydrostatic_strength(sts,scs):
    return -2/3/(1/scs-1/sts)

def stored_energy_hydrostatic(shs,kappa):
    return shs**2/2/kappa 

class input:
    def __init__(self,
                 E = ScalarType(25e3),                  # Young's modulus in MPa
                 nu = ScalarType(0.17),                 # Poisson's ratio
                 Gc = ScalarType(0.02),                 # Energy release rate in N/mm
                 sts = ScalarType(5.0),                 # Uniaxial tensile strength in MPa
                 scs= ScalarType(48.0),                 # Uniaxial compressive strength in MPa
                 eps = 2.0,                             # Regularization length in mm
                 comm = MPI.COMM_WORLD,                 # MPI communicator
                 Dia = 150,                             # Diameter of the specimen in mm
                 thick = 50):                           # Thickness of the specimen in mm

        self.comm = comm
        self.E = E
        self.nu = nu
        self.mu = E/(2*(1 + nu)), 
        self.lmbda = E*nu/((1 + nu)*(1 - 2*nu)), 
        self.kappa = E/(3*(1 - 2*nu))
        self.Gc = Gc
        self.sts = sts
        self.scs = scs
        self.shs = hydrostatic_strength(self.sts, self.scs)
        self.Wts = stored_energy_tension(self.sts, self.E)
        self.Whs = stored_energy_hydrostatic(self.shs, self.kappa)
        self.lch = 3*self.Gc*self.E/8/(self.sts**2)
        self.eps = eps
        self.h = self.eps/5                            # Mesh size, which is 5 times smaller than the regularization length
        self.delta=(1+3*self.h/(8*self.eps))**(-2) * ((self.sts + (1+2*np.sqrt(3))*self.shs)/((8+3*np.sqrt(3))*self.shs)) * 3*self.Gc/(16*self.Wts*self.eps) + (1+3*self.h/(8*self.eps))**(-1) * (2/5)
        self.Dia = Dia
        self.Diaeff = self.Dia
        self.thick = thick