import numpy as np
import ufl
from dolfinx import fem

def mu(mesh, data):
    return fem.Constant(mesh, data.E/(2*(1 + data.nu)))

def lmbda(mesh, data):
    return fem.Constant(mesh, data.E*data.nu/((1 + data.nu)*(1 - 2*data.nu)))

def kappa(mesh, data):
    return fem.Constant(mesh, data.E/(3*(1 - 2*data.nu)))

def energy(v, mesh, data):
    return mu(mesh, data)*(ufl.inner(ufl.sym(ufl.grad(v)),ufl.sym(ufl.grad(v)))) + 0.5*(lmbda(mesh, data))*(ufl.tr(ufl.sym(ufl.grad(v))))**2 
    
def epsilon(v):
    return ufl.sym(ufl.grad(v))

def sigma(v, mesh, data):
    return 2.0*mu(mesh, data)*ufl.sym(ufl.grad(v)) + (lmbda(mesh, data))*ufl.tr(ufl.sym(ufl.grad(v)))*ufl.Identity(len(v))

def sigmavm(sig,v):
    return ufl.sqrt(1/2*(ufl.inner(sig-1/3*ufl.tr(sig)*ufl.Identity(len(v)), sig-1/3*ufl.tr(sig)*ufl.Identity(len(v)))))


def elastic_energy(u, v, du, z, dx, mesh, data, eta):
    # Stored strain energy density
    psi1 = (z**2+eta)*energy(u, mesh, data)
    # Total potential energy
    Pi = psi1*dx 
    # Compute first variation of Pi (directional derivative about u in the direction of v)
    R_u = ufl.derivative(Pi, u, v)
    # Compute Jacobian of R
    Jac_u = ufl.derivative(R_u, u, du)

    return R_u, Jac_u

def fracture_energy(u, z, y, dz, z_prev, dx, Gc, mesh, data, restrict):
    I1 = (z**2)*ufl.tr(sigma(u, mesh, data))
    SQJ2 = (z**2)*sigmavm(sigma(u, mesh, data),u)

    psi11 = energy(u, mesh, data)

    # Configurational force
    alpha1 = data.delta*Gc/data.shs/8/data.eps-2*data.Whs/3/data.shs
    alpha2 = data.delta*Gc/8/data.eps*np.sqrt(3)*(3*data.shs-data.sts)/data.shs/data.sts+2*data.Whs/np.sqrt(3)/data.shs-2*np.sqrt(3)*data.Wts/data.sts
    ce = alpha2*SQJ2 + alpha1*I1 - z*(1-ufl.sqrt(I1**2)/I1)*psi11

    #Balance of configurational forces PDE
    pen=10**7*(3*Gc/8/data.eps)*ufl.conditional(ufl.lt(data.delta,1),1, data.delta)
    Wv=pen/2*((abs(z)-z)**2 + (abs(1-z) - (1-z))**2 )*dx
    Wv2=ufl.conditional(ufl.le(restrict, 0.5), ufl.conditional(ufl.le(z, 0.05), 1, 0), ufl.conditional(ufl.le(z, 0.6), 1, 0))*10*pen/2*((1/4)*(abs(z_prev-z)-(z_prev-z))**2)*dx
   
    # fracture functional
    R_z = y*2*z*(psi11)*dx + y*(ce)*dx + 3*data.delta*Gc/8*(-y/data.eps + 2*data.eps*ufl.inner(ufl.grad(z),ufl.grad(y)))*dx + ufl.derivative(Wv,z,y) +  ufl.derivative(Wv2,z,y)
    # Compute Jacobian of R_z
    Jac_z = ufl.derivative(R_z, z, dz)

    return R_z, Jac_z

