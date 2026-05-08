import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem
from petsc4py import PETSc
from dolfinx import cpp as _cpp
from dolfinx.fem.petsc import create_matrix, create_vector

class NonlinearPDEProblem:
    """Nonlinear problem class for PDEs."""

    def __init__(self, F, u, bc, J):
        V = u.function_space
        du = ufl.TrialFunction(V)
        self.L = fem.form(F)
        self.a = fem.form(J)
        self.bc = bc
        self.internal_forces = None

    def form(self, x):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def F(self, x, b):
        """Assemble residual vector."""
        with b.localForm() as b_local:
            b_local.set(0.0)
        fem.petsc.assemble_vector(b, self.L)
        fem.petsc.apply_lifting(b, [self.a], bcs=[self.bc], x0=[x], alpha=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, self.bc, x, -1.0)

    def J(self, x, A):
        """Assemble Jacobian matrix."""
        A.zeroEntries()
        fem.petsc.assemble_matrix(A, self.a, bcs=self.bc)
        A.assemble()

    def matrix(self):
        return fem.petsc.create_matrix(self.a)

    def vector(self):
        return fem.petsc.create_vector(self.L)

class NewtonSolver:

    def __init__(self, problem):
        self.solver = _cpp.nls.petsc.NewtonSolver(MPI.COMM_WORLD)
        def update(solver, dx, x):
            x.axpy(-1, dx)
        self._b = create_vector(problem.L)
        self._A = create_matrix(problem.a)
        self.solver.setF(problem.F, self._b)
        self.solver.setJ(problem.J, self._A)
        self.solver.set_form(problem.form)
        self.solver.set_update(update)
        self.solver.max_it = 20
        self.solver.error_on_nonconvergence = False
        self.solver.atol = 1.0e-7
        self.solver.rtol = 1.0e-7

        self.ksp = self.solver.krylov_solver
        self.opts = PETSc.Options()
        self.option_prefix = self.ksp.getOptionsPrefix()
        self.opts[f"{self.option_prefix}ksp_type"] = "cg"
        self.opts[f"{self.option_prefix}ksp_rtol"] = 1.0e-8
        self.opts[f"{self.option_prefix}ksp_atol"] = 1.0e-8
        self.opts[f"{self.option_prefix}pc_type"] = "gamg"
        self.opts[f"{self.option_prefix}matptap_via"] = "scalable"
        self.opts[f"{self.option_prefix}options_left"] = None
        self.ksp.setFromOptions()

    def __del__(self):
        self.ksp.destroy()
        self._A.destroy()
        self._b.destroy()

def monitor(comm, state, iteration, u_error_L2, z_error_L2):
    if comm.rank ==0:
        print(f"Iteration: {iteration:3d}, r_Error: u_Error: {u_error_L2:3.4e}, z_Error: {z_error_L2:3.4e}")

def minimization_Problem(comm, state, u_prev, z_prev, parameters, terminate, solver_u, solver_z, dx, monitor=None):

    u = state["u"]
    z = state["z"]

    u_error = []
    z_error = []
 
    u_prev.x.array[:] = u.x.array
    z_prev.x.array[:] = z.x.array
    u_prev.x.scatter_forward()
    z_prev.x.scatter_forward()

    for iteration in range(parameters["max_iter"]):
        
        # solve displacement
        n_u, converged_u = solver_u.solver.solve(u.x.petsc_vec)
        u.x.scatter_forward()
        
        # solve phasefield
        n_z, converged_z = solver_z.solver.solve(z.x.petsc_vec)
        z.x.scatter_forward()

        # check error and update
        L2_u = ufl.inner(u - u_prev, u - u_prev) * dx
        L2_uprev = ufl.inner(u_prev, u_prev) * dx
        u_error_L2 = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(L2_u)), op=MPI.SUM))/np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(L2_uprev)), op=MPI.SUM))
        u_prev.x.array[:] = u.x.array
        u_prev.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        u_error.append(u_error_L2)

        
        L2_z = ufl.inner(z - z_prev, z - z_prev) * dx  
        L2_zprev = ufl.inner(z_prev, z_prev) * dx
        z_error_L2 = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(L2_z)), op=MPI.SUM))/np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(L2_zprev)), op=MPI.SUM))
        z_prev.x.array[:] = z.x.array
        z_prev.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        z_error.append(z_error_L2)

        if monitor is not None:
            monitor(comm, state, iteration, u_error_L2, z_error_L2)

        if u_error_L2 <= 100*parameters["atol"] and z_error_L2 <= parameters["atol"]:
            break

        if (np.isnan(u_error_L2)==True) or (np.isnan(z_error_L2)==True):
            terminate = 1
            break

        if (iteration > 5):
            if np.abs(z_error_L2 - z_error[-4]) < 1e-6 and np.abs(u_error_L2 - u_error[-4]) < 1e-6:
                break
            if np.abs(z_error_L2 - z_error[-3]) < 1e-6 and np.abs(u_error_L2 - u_error[-3]) < 1e-6:
                break


    else:
        pass #raise RuntimeError(f"Could not converge after {iteration:3d} iteration, u_error {u_error_L2:3.4e}, z_error {z_error_L2:3.4e}")

    return (u_error_L2, z_error_L2, iteration, terminate)  