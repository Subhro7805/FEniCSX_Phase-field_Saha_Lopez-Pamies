import numpy as np
from dolfinx.mesh import locate_entities, meshtags, locate_entities_boundary
from dolfinx.fem import locate_dofs_topological, dirichletbc
from ufl import Measure
from dolfinx import default_scalar_type

def boundary_marker(mesh, loading1, loading2):
    fdim = mesh.topology.dim-1

    boundaries = [(1, loading1),
                   (2, loading2)]

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
    def __init__(self, mesh, location, marker, facet_tag, V_u, V1, c1):
        self._location = location
        fdim = mesh.topology.dim-1
        if location == "loading1":
            facets = facet_tag.find(marker)
            dofs = locate_dofs_topological((V_u.sub(1), V1), fdim, facets)
            self._bc = dirichletbc(c1, dofs, V_u.sub(1))
        elif location == "loading2":
            facets = facet_tag.find(marker)
            dofs = locate_dofs_topological((V_u.sub(1), V1), fdim, facets)
            self._bc = dirichletbc(c1, dofs, V_u.sub(1))
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(location))
    @property
    def bc(self):
        return self._bc

    @property
    def location(self):
        return self._location


def bc_phasefield(mesh, V_z, outer):
    fdim = mesh.topology.dim-1
    entity_outer = locate_entities(mesh, fdim, outer)
    dofs_outer = locate_dofs_topological(V_z, fdim, entity_outer)
    bc_outer = dirichletbc(default_scalar_type(1), dofs_outer, V_z)

    bc_z = [bc_outer]
    
    return bc_z

def bc_dirichlet(mesh, V_u, boundary_conditions, rightbottom, leftbottom, loading1, loading2, rightsupport, leftsupport, Zsym):
    entity_rightbottom = locate_entities_boundary(mesh, mesh.topology.dim-1, rightbottom)
    dofs_rightbottom_y = locate_dofs_topological(V_u.sub(1), mesh.topology.dim-1, entity_rightbottom)
    bc_rightbottom = dirichletbc(default_scalar_type(0), dofs_rightbottom_y, V_u.sub(1))

    entity_leftbottom = locate_entities_boundary(mesh, mesh.topology.dim-1, leftbottom)
    dofs_leftbottom_y = locate_dofs_topological(V_u.sub(1), mesh.topology.dim-1, entity_leftbottom)
    bc_leftbottom = dirichletbc(default_scalar_type(0), dofs_leftbottom_y, V_u.sub(1))  

    entity_rightsupport = locate_entities_boundary(mesh, mesh.topology.dim-1, rightsupport)
    dofs_rightsupport_x = locate_dofs_topological(V_u.sub(0), mesh.topology.dim-1, entity_rightsupport)
    dofs_rightsupport_z = locate_dofs_topological(V_u.sub(2), mesh.topology.dim-1, entity_rightsupport)
    bc_rightsupport_x = dirichletbc(default_scalar_type(0), dofs_rightsupport_x, V_u.sub(0))
    bc_rightsupport_z = dirichletbc(default_scalar_type(0), dofs_rightsupport_z, V_u.sub(2))

    entity_leftsupport = locate_entities_boundary(mesh, mesh.topology.dim-1, leftsupport)
    dofs_leftsupport_x = locate_dofs_topological(V_u.sub(0), mesh.topology.dim-1, entity_leftsupport)
    dofs_leftsupport_z = locate_dofs_topological(V_u.sub(2), mesh.topology.dim-1, entity_leftsupport)
    bc_leftsupport_x = dirichletbc(default_scalar_type(0), dofs_leftsupport_x, V_u.sub(0))
    bc_leftsupport_z = dirichletbc(default_scalar_type(0), dofs_leftsupport_z, V_u.sub(2))

    entity_Zsym = locate_entities_boundary(mesh, mesh.topology.dim-1, Zsym)
    dofs_Zsym_z = locate_dofs_topological(V_u.sub(2), mesh.topology.dim-1, entity_Zsym)
    bc_Zsym = dirichletbc(default_scalar_type(0), dofs_Zsym_z, V_u.sub(2))

    bc_u = [bc_rightbottom, bc_leftbottom, bc_rightsupport_x, bc_rightsupport_z, bc_leftsupport_x, bc_leftsupport_z, bc_Zsym]
    for condition in boundary_conditions:
        if condition.location == "loading1" or condition.location == "loading2":
            bc_u.append(condition.bc)
    return bc_u

def dintegration(mesh, facet_tag):
    metadata = {"quadrature_degree": 4}
    ds = Measure("ds", domain=mesh, subdomain_data=facet_tag, metadata=metadata)

    dx = Measure("dx", domain=mesh, metadata=metadata)

    return ds, dx