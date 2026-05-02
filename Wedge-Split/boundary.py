import numpy as np
from dolfinx.mesh import locate_entities, meshtags, locate_entities_boundary
from dolfinx.fem import locate_dofs_topological, dirichletbc
from ufl import Measure
from dolfinx import default_scalar_type

def boundary_marker(mesh, loading, cracktip):
    fdim = mesh.topology.dim-1

    boundaries = [(1, loading),
                  (2, cracktip)]

    facet_indices, facet_markers = [], []
    for (marker, locator) in boundaries:
        facets = locate_entities(mesh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

    return facet_tag


class BoundaryCondition():
    def __init__(self, mesh, location, marker, facet_tag, V_u, V1, c):
        self._location = location
        fdim = mesh.topology.dim-1
        if location == "loading":
            facets = facet_tag.find(marker)
            dofs = locate_dofs_topological((V_u.sub(0), V1), fdim, facets)
            self._bc = dirichletbc(c, dofs, V_u.sub(0))
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(location))
    @property
    def bc(self):
        return self._bc

    @property
    def location(self):
        return self._location


def bc_phasefield(mesh, V_z, cracktip, outer):
    fdim = mesh.topology.dim-1
    entity_cracktip = locate_entities(mesh, fdim, cracktip)
    dofs_cracktip = locate_dofs_topological(V_z, fdim, entity_cracktip)
    bc_cracktip = dirichletbc(default_scalar_type(0), dofs_cracktip, V_z)

    entity_outer = locate_entities(mesh, fdim, outer)
    dofs_outer = locate_dofs_topological(V_z, fdim, entity_outer)
    bc_outer = dirichletbc(default_scalar_type(1), dofs_outer, V_z)

    bc_z = [bc_cracktip, bc_outer]
    
    return bc_z

def bc_dirichlet(mesh, V_u, boundary_conditions, rightbottom, bottom, loading, Xsym, Zsym):
    entity_rightbottom = locate_entities_boundary(mesh, mesh.topology.dim-1, rightbottom)
    dofs_rightbottom_x = locate_dofs_topological(V_u.sub(0), mesh.topology.dim-1, entity_rightbottom)
    bc_rightbottom = dirichletbc(default_scalar_type(0), dofs_rightbottom_x, V_u.sub(0))

    entity_bottom = locate_entities_boundary(mesh, mesh.topology.dim-1, bottom)
    dofs_bottom = locate_dofs_topological(V_u.sub(1), mesh.topology.dim-1, entity_bottom)
    bc_bottom = dirichletbc(default_scalar_type(0), dofs_bottom, V_u.sub(1))

    entity_Xsym = locate_entities_boundary(mesh, mesh.topology.dim-1, Xsym)
    dofs_Xsym_x = locate_dofs_topological(V_u.sub(0), mesh.topology.dim-1, entity_Xsym)
    bc_Xsym = dirichletbc(default_scalar_type(0), dofs_Xsym_x, V_u.sub(0))

    entity_Zsym = locate_entities_boundary(mesh, mesh.topology.dim-1, Zsym)
    dofs_Zsym_z = locate_dofs_topological(V_u.sub(2), mesh.topology.dim-1, entity_Zsym)
    bc_Zsym = dirichletbc(default_scalar_type(0), dofs_Zsym_z, V_u.sub(2))

    bc_u = [bc_rightbottom, bc_bottom, bc_Xsym, bc_Zsym]
    for condition in boundary_conditions:
        if condition.location == "loading":
            bc_u.append(condition.bc)
    return bc_u


def dintegration(mesh, facet_tag):
    metadata = {"quadrature_degree": 4}
    ds = Measure("ds", domain=mesh, subdomain_data=facet_tag, metadata=metadata)

    dx = Measure("dx", domain=mesh, metadata=metadata)

    return ds, dx