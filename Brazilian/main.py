import numpy as np
from mpi4py import MPI
from dolfinx import log, fem, geometry, mesh
import input
import createMesh
from dolfinx.io import XDMFFile, VTXWriter
import basix
import ufl
import elasticity
from petsc4py import PETSc
import solver
import evaluate

log.set_log_level(log.LogLevel.ERROR)

data = input.input()

# Problem description
comm = data.comm

# Material properties (unit is in MPa and mm)
E, nu = data.E, data.nu	                                    # Young's modulus and Poisson's ratio
mu, lmbda, kappa = data.mu, data.lmbda, data.kappa          # Lame parameters and bulk modulus
Gc = data.Gc	                                            # Critical energy release rate       
sts, scs, shs = data.sts, data.scs, data.shs	            # Tensile strength, compressive strength and hydrostatic strength
Wts, Whs = data.Wts, data.Whs	                            # Tensile and hydrostatic stored energy density

# Geometry 
Dia = data.Dia
Diaeff = data.Diaeff
thick = data.thick

#The regularization length and mesh size
lch = data.lch
eps, h = data.eps, data.h
delta = data.delta

if comm.rank == 0:
    print(f"lch = {lch}, eps = {eps}, h = {h}, delta = {delta}")

# Time-stepping parameters
T = 1
Totalsteps = 200
startstepsize = 1/Totalsteps
stepsize=startstepsize
t=0
iter=1
printsteps=10                             # Print results every printsteps steps. You can change this number to print more or less frequently. Setting printsteps=1 will print results at every step, which can be very time and memory consuming for large simulations. 
samesizecount=1
terminate=0

# For this problem, we create a half of the domain and apply symmetric boundary conditions on the z=0 surfaces. You can create a different mesh for your problem using gmsh or other mesh generators. Make sure to save the mesh in XDMF format and change the filename in the following lines when reading the mesh.
domain = createMesh.meshBrazilian(comm, Dia, thick, h)                          # Create mesh and save as XDMF file.

# Read mesh if you have already created the mesh and saved as XDMF file. Comment out the above line and uncomment the following lines to read the mesh from XDMF file. Make sure to change the filename if you have a different name for your mesh file.
# if comm.rank == 0:
#     print("reading mesh...")

# with XDMFFile(comm, "your_mesh.xdmf", "r") as xdmf:
#     domain = xdmf.read_mesh(name="mesh")

# if comm.rank == 0:
#     print("mesh read.")

domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)

# define function spaces
# function space for displacement
element_u = basix.ufl.element("Lagrange", domain.basix_cell(), degree=1, shape=(domain.geometry.dim,))
V_u = fem.functionspace(domain, element_u)

# function space for phase-field
element_z = basix.ufl.element('Lagrange', domain.basix_cell(), degree=1)
V_z = fem.functionspace(domain, element_z)

maxdisp = 0.4

# An elastic contact problem is performed in Abaqus to get the contact displacement field and contact area. In the following functions loading1 and loading2 we define a functional form of the contact region based on the FEM solution from Abaqus. If you are using a different geomtery, it is advised to perform the contact simulation separately. Also, you can perform the contact problem using penalty method (or some other algorithm) in the same code and get the contact region and contact displacement field at each time step, which might be more accurate but also more computationally expensive.
# define boundary subdomains
def loading1(x, t):
    R = Dia/2
    Reff = Diaeff/2
    disp = maxdisp
    return (np.abs(x[0]*x[0]+x[1]*x[1]-R*R)<1e-4) & (np.abs(x[0])<np.sqrt(disp*t*Reff/2)+0.5*h) & (x[1]>0)    # Contact region on the upper half of the disk as a function of loadstep, which means it evolves as the load increases in the platen. The functional form is based on the FEM solution from Abaqus. You can change this function to define the contact region for your problem.    

def loading2(x, t):
    R=Dia/2
    Reff=Diaeff/2
    disp = maxdisp
    return (np.abs(x[0]*x[0]+x[1]*x[1]-R*R)<1e-4) & (np.abs(x[0])<np.sqrt(disp*t*Reff/2)+0.5*h) & (x[1]<0)    # Contact region on the lower half of the disk as a function of loadstep, which means it evolves as the load increases in the platen. The functional form is based on the FEM solution from Abaqus. You can change this function to define the contact region for your problem.

def center(x):
    return (np.abs(x[0]-0)<h) & (np.abs(x[1]-0)<h) & (np.abs(x[2]-0)<h)                                       # fix x and z displacements at the center of the disk to prevent rigid body motion. 

def Zsym(x):
    return np.isclose(np.abs(x[2]),0.0)                                                                       # apply symmetric boundary condition on the z=0 plane. If you are modeling the full disk, you can remove this boundary condition.

def outer(x):
    return (np.abs(x[0])>2*Dia)                                                                               # Define the outer domain. We will apply phase-field boundary condition in this region to ensure the phase-field is 1 (undamaged) throughout this region and does not evolve. For this problem, we do not set any such region inside the domain.

def restricted_region(x):
    R = Dia/2
    Reff = Diaeff/2
    disp = maxdisp
    return (x[0]*x[0]+x[1]*x[1]>=(R-2*eps)**2) & (np.abs(x[0])<=np.sqrt(disp*Reff/2)+2*eps)                   # Define a small region around the contact area where we impose a higher threshold for phase-field irreversibility to prevent oscillation in the calculation of reaction force.

# BC parameters for contact
k1 = 2.046
a1 = 3.27401636e-02
a2 = 1.10960440e-05
a3 = 5.38886115e-08
a4 = 8.69206377e+00
a5 = 1.40720433e-01

# Define the functional form of the contact displacement field. The functional form is based on the FEM solution from Abaqus. You can change this function to define the contact displacement field for your problem.
class MyExpression:
    def __init__(self):
        self.t = 0.0
        self.disp = maxdisp
        self.R = Diaeff/2
        self.h = h
        self.k1 = k1
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5

    def eval(self, x):
        return ((1-self.k1)/(1+self.k1)/2/self.R*(x[0]*np.sqrt((np.sqrt(self.disp*self.t*self.R/2)+0.5*self.h)**2-x[0]*x[0])+((np.sqrt(self.disp*self.t*self.R/2)+0.5*self.h)**2)*np.arcsin(x[0]/(np.sqrt(self.disp*self.t*self.R/2)+0.5*self.h))),
                -self.t*self.disp+(x[0]*x[0])/(2*self.R),
                self.t*self.disp * (self.a1*x[2]+self.a2*x[2]**3+self.a3*x[2]**5)/(self.a4+self.a5*x[0]**2))

class MyExpression2:
    def __init__(self):
        self.t = 0.0
        self.disp = maxdisp
        self.R = Diaeff/2
        self.h = h
        self.k1 = k1
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5

    def eval(self, x):
        return ((1-self.k1)/(1+self.k1)/2/self.R*(x[0]*np.sqrt((np.sqrt(self.disp*self.t*self.R/2)+0.5*self.h)**2-x[0]*x[0])+((np.sqrt(self.disp*self.t*self.R/2)+0.5*self.h)**2)*np.arcsin(x[0]/(np.sqrt(self.disp*self.t*self.R/2)+0.5*self.h))),
                self.t*self.disp-(x[0]*x[0])/(2*self.R),
                self.t*self.disp * (self.a1*x[2]+self.a2*x[2]**3+self.a3*x[2]**5)/(self.a4+self.a5*x[0]**2))


re1 = MyExpression()
re1.t = 0
c1 = fem.Function(V_u)
c1.interpolate(re1.eval)

re2 = MyExpression2()
re2.t = 0
c2 = fem.Function(V_u)
c2.interpolate(re2.eval)

def loading1_func(y):
    def partial_func1(x):
        return loading1(x, y)
    return partial_func1
def loading2_func(y):
    def partial_func2(x):
        return loading2(x, y)
    return partial_func2

# define functions
du = ufl.TrialFunction(V_u)                            # Incremental displacement
v  = ufl.TestFunction(V_u)                             # Test function for displacement
u  = fem.Function(V_u, name="Displacement")            # Displacement from previous iteration
dz = ufl.TrialFunction(V_z)                            # Incremental phase-field
y  = ufl.TestFunction(V_z)                             # Test function for phase-field
z  = fem.Function(V_z, name="Phasefield")              # Phase-field from previous iteration 
restrict = fem.Function(V_z)                           

state = {"u": u, "z": z}

# define Dirichlet and Neumann boundary conditions
fdim = domain.topology.dim-1

loading1_facets = mesh.locate_entities_boundary(domain, fdim, loading1_func(0))
loading2_facets = mesh.locate_entities_boundary(domain, fdim, loading2_func(0))
center_facets = mesh.locate_entities(domain, fdim, center)
Zsym_facets = mesh.locate_entities_boundary(domain, fdim, Zsym)
outer_facets = mesh.locate_entities(domain, fdim, outer)

dofs_loading1 = fem.locate_dofs_topological(V_u, fdim, loading1_facets)
dofs_loading2 = fem.locate_dofs_topological(V_u, fdim, loading2_facets)
dofs_center_x = fem.locate_dofs_topological(V_u.sub(0), fdim, center_facets)
dofs_center_z = fem.locate_dofs_topological(V_u.sub(2), fdim, center_facets)
dofs_Zsym = fem.locate_dofs_topological(V_u.sub(2), fdim, Zsym_facets)

dofs_outer = fem.locate_dofs_topological(V_z, fdim, outer_facets)

bcl1 = fem.dirichletbc(c1, dofs_loading1)
bcl2 = fem.dirichletbc(c2, dofs_loading2)
bccx = fem.dirichletbc(PETSc.ScalarType(0), dofs_center_x, V_u.sub(0))
bccz = fem.dirichletbc(PETSc.ScalarType(0), dofs_center_z, V_u.sub(2))
bcZsym = fem.dirichletbc(PETSc.ScalarType(0), dofs_Zsym, V_u.sub(2))

# Displacement boundary condition
bc_u = [bcl1, bcl2, bccx, bccz, bcZsym]

bct_z = fem.dirichletbc(PETSc.ScalarType(1), dofs_outer, V_z)

# Phase-field boundary condition
bc_z = [bct_z]

metadata = {"quadrature_degree": 4}
n=ufl.FacetNormal(domain)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

# Initialize the functions
u.x.array[:] = 0.
z.x.array[:] = 1.
u.x.scatter_forward()
z.x.scatter_forward()   

u_prev = fem.Function(V_u)
u_prev.x.array[:] = u.x.array
u_prev.x.scatter_forward()
z_prev = fem.Function(V_z)
z_prev.x.array[:] = z.x.array
z_prev.x.scatter_forward()

un = fem.Function(V_u)
un.x.array[:] = u.x.array
un.x.scatter_forward()
zn = fem.Function(V_z)
zn.x.array[:] = z.x.array
zn.x.scatter_forward()

cells_restrict = mesh.locate_entities(domain, domain.topology.dim-1, restricted_region)

dofs_restrict = fem.locate_dofs_topological(V_z, domain.topology.dim-1, cells_restrict)
bc_restrict = fem.dirichletbc(PETSc.ScalarType(1), dofs_restrict, V_z)

restrict_init = fem.Function(V_z)
with restrict_init.x.petsc_vec.localForm() as bc_local:
    bc_local.set(0)
restrict.interpolate(restrict_init)
fem.set_bc(restrict.x.petsc_vec, [bc_restrict])
restrict.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

# Stored strain energy density
# Compute R_u, first variation of Pi (directional derivative about u in the direction of v) and Jacobian of R_u
eta = 1e-6  # numerical parameter to avoid zero elastic energy at any material point for numerical tractability
R_u, Jac_u = elasticity.elastic_energy(u, v, du, z, dx, domain, data, eta)

# fracture energy
# Compute R_z, first variation of Pi_z (directional derivative about z in the direction of y) and Jacobian of R_z
R_z, Jac_z = elasticity.fracture_energy(u, z, y, dz, zn, dx, Gc, domain, data, restrict)

# solver parameters to control the convergence of the nonlinear solver. You can change these parameters to achieve better convergence for your problem. For example, if you find that the solver is not converging, you can try increasing the maximum number of iterations or loosening the tolerance. However, be careful when changing these parameters, as setting them too loose may lead to inaccurate results, while setting them too tight may lead to very long computation times.
parameters = {"atol": 1e-7, "max_iter": 50}

bb_tree = geometry.bb_tree(domain, domain.topology.dim)

# Create VTXWriter to write results in .bp format for visualization in ParaView. You can also write results in XDMF format using XDMFFile, but writing in .bp format is more efficient for large simulations. Make sure to change the filename if you want to save the results with a different name.
vtx = VTXWriter(comm, "./results/brazilianflat_3D.bp", [u, z], engine="BP4")

t += stepsize
while t-stepsize < T:
    terminate = 0

    # call boundary condition
    re1.t=t
    c1.interpolate(re1.eval)
    re2.t=t
    c2.interpolate(re2.eval)

    loading1_facets = mesh.locate_entities_boundary(domain, fdim, loading1_func(t))
    loading2_facets = mesh.locate_entities_boundary(domain, fdim, loading2_func(t))

    dofs_loading1 = fem.locate_dofs_topological(V_u, fdim, loading1_facets)
    dofs_loading2 = fem.locate_dofs_topological(V_u, fdim, loading2_facets)
    bcl1 = fem.dirichletbc(c1, dofs_loading1)
    bcl2 = fem.dirichletbc(c2, dofs_loading2)
    bccx = fem.dirichletbc(PETSc.ScalarType(0), dofs_center_x, V_u.sub(0))
    bccz = fem.dirichletbc(PETSc.ScalarType(0), dofs_center_z, V_u.sub(2))
    bcZsym = fem.dirichletbc(PETSc.ScalarType(0), dofs_Zsym, V_u.sub(2))

    bc_u = [bcl1, bcl2, bccx, bccz, bcZsym]

    # Update the residual and Jacobian with the current solution
    problem_u = solver.NonlinearPDEProblem(R_u, u, bc_u, Jac_u)
    problem_z = solver.NonlinearPDEProblem(R_z, z, bc_z, Jac_z)

    # Create nonlinear solvers for displacement and phase-field. You can change the type of nonlinear solver in solver.py file. For example, you can use a line search algorithm or a trust region algorithm instead of the default Newton solver. However, be careful when changing the solver type, as some solvers may not be suitable for this type of problem and may lead to convergence issues.
    solver_u = solver.NewtonSolver(problem_u)
    solver_z = solver.NewtonSolver(problem_z)

    # deploy the solver to solve the minimization problem.
    if comm.rank == 0:
        print(f"-- Solving for step = {iter:3d} at time = {t:3.8f} with stepsize = {stepsize:3.8f} --")
    u_error_L2, z_error_L2, iteration, terminate=solver.minimization_Problem(comm, state, u_prev, z_prev, parameters, terminate, solver_u, solver_z, dx, monitor=solver.monitor)

    un.x.array[:] = u.x.array
    zn.x.array[:] = z.x.array

    un.x.scatter_forward()
    zn.x.scatter_forward()

    zmin = domain.comm.allreduce(np.min(z.x.array), op=MPI.MIN)
    zmax = domain.comm.allreduce(np.max(z.x.array), op=MPI.MAX)

    # calculate reaction at top contact region.
    Res, _ = elasticity.elastic_energy(u, v, du, z, dx, domain, data, eta)
    b_e = fem.petsc.assemble_vector(fem.form(-Res))
    b_e.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dofs_load_surface_x = fem.locate_dofs_topological(V_u.sub(0), fdim, loading1_facets)
    dofs_load_surface_y = fem.locate_dofs_topological(V_u.sub(1), fdim, loading1_facets)
    dofs_load_surface_z = fem.locate_dofs_topological(V_u.sub(2), fdim, loading1_facets)
    with b_e.localForm() as b1:
        Fx = np.sum(b1[dofs_load_surface_x])
        Fy = np.sum(b1[dofs_load_surface_y])
        Fz = np.sum(b1[dofs_load_surface_z])
    Fx = domain.comm.allreduce(Fx, op=MPI.SUM)
    Fy = domain.comm.allreduce(Fy, op=MPI.SUM)
    Fz = domain.comm.allreduce(Fz, op=MPI.SUM)
    ReactionForce = np.sqrt(Fx**2+Fy**2+Fz**2)

    z_center = evaluate.evaluate_function(comm, domain, z, (0.0,0.0,0.0), bb_tree)

    # change the printsteps as you need.
    if t*maxdisp > 0.2:
        printsteps = 5
    if t*maxdisp > 0.26:
        printsteps = 1
    if z_center < 0.5:
        printsteps = 1

    # print results
    if iter % printsteps == 0:
        vtx.write(t)
    
    # print results to a text file.
    if comm.rank==0:
        with open('./results/brazilianflat_3D.txt', 'a') as rfile:
            rfile.write("timestep = %s, u2 = %s, zmin = %s, zmax = %s, z_center = %s, Reaction = %s\n" % (str(t), str(t*maxdisp), str(zmin), str(zmax), str(z_center), str(2*ReactionForce)))


    del solver_u
    del solver_z

    #time stepping
    t+=stepsize
    iter+=1

vtx.close()