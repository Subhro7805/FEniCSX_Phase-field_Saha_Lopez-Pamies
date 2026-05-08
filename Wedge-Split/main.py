import numpy as np
from mpi4py import MPI
from dolfinx import log, fem, geometry, mesh
import input
import createMesh
from dolfinx.io import XDMFFile, VTXWriter
import basix
import ufl
import boundary
import elasticity
from petsc4py import PETSc
import solver
import evaluate


log.set_log_level(log.LogLevel.ERROR)

data = input.input()

# Problem description
comm = data.comm

# Material properties (unit is in MPa and mm)
E, nu = data.E, data.nu	                                # Young's modulus and Poisson's ratio
mu, lmbda, kappa = data.mu, data.lmbda, data.kappa      # Lame parameters and bulk modulus
Gc = data.Gc	                                        # Critical energy release rate
sts, scs, shs = data.sts, data.scs, data.shs	        #Tensile strength, compressive strength and hydrostatic strength
Wts, Whs = data.Wts, data.Whs	                        #Tensile and hydrostatic stored energy density

# Geometry and initial crack length
L = data.L
H = data.H
D = data.D
groove_depth = data.groove_depth
groove_length = data.groove_length
notch = data.notch
notch_width = data.notch_width


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

# For this problem, we create a quarter of the domain and apply symmetric boundary conditions on the left and front surfaces. You can create a different mesh for your problem using gmsh or other mesh generators. Make sure to save the mesh in XDMF format and change the filename in the following lines when reading the mesh.
domain = createMesh.meshwedge3D(comm, L, H, D, groove_depth, groove_length, notch, notch_width, h)           # Create mesh and save as XDMF file. 

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

# define boundary subdomains
def loading(x):                                                            
    return np.isclose(x[0], groove_length/2) & (x[1]>=H-groove_depth/2)                 # Apply loading on the right groove. Remember we are modeling a quarter of the domain, so the loading is applied on the right groove only. If you are modeling the full domain, change the load accordingly. 

def Xsym(x):
    return np.isclose(x[0], 0)                                                          # Apply symmetric boundary condition on the left surface (x=0). If you are modeling the full domain, you don't need this boundary condition.

def Zsym(x):
    return np.isclose(x[2], 0)                                                          # Apply symmetric boundary condition on the ceter surface (z=0). If you are modeling the full domain, you don't need this boundary condition.

def rightbottom(x):
    return (x[0]>L/2-h) & (x[1]<h)
    
def bottom(x):
    return np.isclose(x[1], 0)

def cracktip(x):
    return (x[0]<= notch_width/2+1e-4) & (x[1] < H - groove_depth - notch + h*8*2) & (x[1] > H - groove_depth - notch -1e-6)               # Define the crack tip region. We will apply phase-field boundary condition in this region to ensure the crack initiates from the notch tip.

def outer(x):
    return (x[0]>2*L)    # Define the outer domain. We will apply phase-field boundary condition in this region to ensure the phase-field is 1 (undamaged) throughout this region and does not evolve. For this problem, we do not set any such region inside the domain. 


# define functions
du = ufl.TrialFunction(V_u)                            # Incremental displacement
v  = ufl.TestFunction(V_u)                             # Test function for displacement
u  = fem.Function(V_u, name="Displacement")            # Displacement from previous iteration
dz = ufl.TrialFunction(V_z)                            # Incremental phase-field
y  = ufl.TestFunction(V_z)                             # Test function for phase-field
z  = fem.Function(V_z, name="Phasefield")              # Phase-field from previous iteration 

state = {"u": u, "z": z}

# define Dirichlet boundary conditions for displacement and phase-field. You can change the boundary conditions in boundary.py file according to your problem. For example, if you are modeling a different geometry or loading condition, you may need to change the boundary conditions accordingly. Make sure to apply appropriate boundary conditions for both displacement and phase-field to ensure the well-posedness of the problem.
fdim = domain.topology.dim-1

maxdisp = 0.1                                          # Maximum uh applied at the loading surface. You can change this value accordingly. The actual displacement applied at the loading surface will be maxdisp*t, where t is the current time. So the loading will increase linearly with time until it reaches maxdisp at t=1.

class MyExpression:
    def __init__(self):
        self.t = 0.0
        self.disp = maxdisp

    def eval(self, x):
        values = np.zeros((1, x.shape[1]))
        values[0,:] = self.t*self.disp
        return values
    
V_0, _ = V_u.sub(0).collapse()                         # Function space for applying Dirichlet boundary condition on the x-displacement at the loading surface. We only apply Dirichlet boundary condition on the x-displacement, so we use V_u.sub(0) to get the function space for x-displacement and then collapse it to get a scalar function space.

re1 = MyExpression()
re1.t = 0
c1 = fem.Function(V_0)
c1.interpolate(re1.eval)

facet_tag = boundary.boundary_marker(domain, loading, cracktip)
ds, dx = boundary.dintegration(domain, facet_tag)
# You can visualize the mesh and boundary markers using XDMF files. Uncomment the following lines to write the mesh and boundary markers to XDMF files. 
# with XDMFFile(comm, "./results/mesh_tags.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)
#     xdmf.write_meshtags(facet_tag, domain.geometry)

boundary_conditions = [boundary.BoundaryCondition(domain, "loading", 1, facet_tag, V_u, V_0, c1)]

# phase-field boundary condition
bc_z = boundary.bc_phasefield(domain, V_z, cracktip, outer)

# Displacement boundary condition
bc_u = boundary.bc_dirichlet(domain, V_u, boundary_conditions, rightbottom, bottom, loading, Xsym, Zsym)

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

# Expressions for strain energy and fracture energy for 3-D elasticity

# Stored strain energy density
# Compute R_u, first variation of Pi (directional derivative about u in the direction of v) and Jacobian of R_u
eta = 1e-6  # numerical parameter to avoid zero elastic energy at any material point for numerical tractability
R_u, Jac_u = elasticity.elastic_energy(u, v, du, z, dx, domain, data, eta)

# fracture energy
# Compute R_z, first variation of Pi_z (directional derivative about z in the direction of y) and Jacobian of R_z
R_z, Jac_z = elasticity.fracture_energy(u, z, y, dz, zn, dx, Gc, domain, data, sts, shs, Wts, Whs, delta)

# solver parameters to control the convergence of the nonlinear solver. You can change these parameters to achieve better convergence for your problem. For example, if you find that the solver is not converging, you can try increasing the maximum number of iterations or loosening the tolerance. However, be careful when changing these parameters, as setting them too loose may lead to inaccurate results, while setting them too tight may lead to very long computation times.
parameters = {"atol": 1e-7, "max_iter": 50}

bb_tree = geometry.bb_tree(domain, domain.topology.dim)

# Create VTXWriter to write results in .bp format for visualization in ParaView. You can also write results in XDMF format using XDMFFile, but writing in .bp format is more efficient for large simulations. Make sure to change the filename if you want to save the results with a different name.
vtx = VTXWriter(comm, "./results/mortarwedge_3D.bp", [u, z], engine="BP4")

t += stepsize
while t-stepsize < T:
    terminate = 0

    # call boundary condition
    re1.t = t
    c1.interpolate(re1.eval)

    boundary_conditions = [boundary.BoundaryCondition(domain, "loading", 1, facet_tag, V_u, V_0, c1)]
    bc_u = boundary.bc_dirichlet(domain, V_u, boundary_conditions, rightbottom, bottom, loading, Xsym, Zsym)

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

    # print results
    if iter % printsteps == 0 or iter == 1:
        vtx.write(t)
    
    # calculate reaction force at the loading surface
    Res, _ = elasticity.elastic_energy(u, v, du, z, dx, domain, data, eta)
    b_e = fem.petsc.assemble_vector(fem.form(Res))
    b_e.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    loading_facets = mesh.locate_entities_boundary(domain, fdim, loading)
    dofs_load_surface = fem.locate_dofs_topological(V_u.sub(0), fdim, loading_facets)
    with b_e.localForm() as b1:
        Fx = np.sum(b1[dofs_load_surface])
    Fx = domain.comm.allreduce(Fx, op=MPI.SUM)

    z_crackfront = evaluate.evaluate_function(comm, domain, z, (0, H - groove_depth - notch - eps, 0), bb_tree)

    # Print results to a text file
    if comm.rank==0:
        with open('./results/mortarwedge_3D.txt', 'a') as rfile:
            rfile.write("timestep = %s, uh = %s, z_eps = %s, Ph = %s, Pv = %s\n" % (str(t), str(maxdisp*t), str(z_crackfront), str(2*Fx), str(4*Fx*np.tan(5*np.pi/180))))

    del solver_u
    del solver_z

    #time stepping
    if z_crackfront<0.1:
        printsteps=2
    t+=stepsize
    iter+=1

vtx.close()
