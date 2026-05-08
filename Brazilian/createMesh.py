import gmsh
from dolfinx.cpp.mesh import create_cell_partitioner
from dolfinx.io import gmshio, XDMFFile
import dolfinx

def meshBrazilian(comm, Dia, thick, h):
    if comm.rank == 0:
        gmsh.initialize()
        gmsh.model.add("Brazilian3D")
        gmsh.model.setCurrent("Brazilian3D")

        cylinder_dim_tags = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, thick/2, Dia/2)
        gmsh.model.occ.synchronize()

        gmsh.model.mesh.field.add("Box", 1)
        gmsh.model.mesh.field.setNumber(1, "VIn", h )
        gmsh.model.mesh.field.setNumber(1, "VOut", 5*h)
        gmsh.model.mesh.field.setNumber(1, "XMin", -Dia/2/3)
        gmsh.model.mesh.field.setNumber(1, "XMax", Dia/2/3)
        gmsh.model.mesh.field.setNumber(1, "YMin", -Dia)
        gmsh.model.mesh.field.setNumber(1, "YMax", Dia)
        gmsh.model.mesh.field.setNumber(1, "ZMin", -thick)
        gmsh.model.mesh.field.setNumber(1, "ZMax", thick)
        gmsh.model.mesh.field.setNumber(1, "Thickness", 100*h)

        gmsh.model.mesh.field.add("Box", 2)
        gmsh.model.mesh.field.setNumber(2, "VIn", 2*h )
        gmsh.model.mesh.field.setNumber(2, "VOut", 5*h)
        gmsh.model.mesh.field.setNumber(2, "XMin", -Dia/2/2)
        gmsh.model.mesh.field.setNumber(2, "XMax", Dia/2/2)
        gmsh.model.mesh.field.setNumber(2, "YMin", -Dia)
        gmsh.model.mesh.field.setNumber(2, "YMax", Dia)
        gmsh.model.mesh.field.setNumber(2, "ZMin", -thick)
        gmsh.model.mesh.field.setNumber(2, "ZMax", thick)
        gmsh.model.mesh.field.setNumber(2, "Thickness", 100*h)

        gmsh.model.mesh.field.add("Box", 3)
        gmsh.model.mesh.field.setNumber(3, "VIn", 0.15)
        gmsh.model.mesh.field.setNumber(3, "VOut", 5*h)
        gmsh.model.mesh.field.setNumber(3, "XMin", -0.05*Dia)
        gmsh.model.mesh.field.setNumber(3, "XMax", 0.05*Dia)
        gmsh.model.mesh.field.setNumber(3, "YMin", -Dia/2)
        gmsh.model.mesh.field.setNumber(3, "YMax", -Dia/2+0.05*Dia)
        gmsh.model.mesh.field.setNumber(3, "ZMin", 0)
        gmsh.model.mesh.field.setNumber(3, "ZMax", thick/2)
        gmsh.model.mesh.field.setNumber(3, "Thickness", 20*h)

        gmsh.model.mesh.field.add("Box", 4)
        gmsh.model.mesh.field.setNumber(4, "VIn", 0.15)
        gmsh.model.mesh.field.setNumber(4, "VOut", 5*h)
        gmsh.model.mesh.field.setNumber(4, "XMin", -0.05*Dia)
        gmsh.model.mesh.field.setNumber(4, "XMax", 0.05*Dia)
        gmsh.model.mesh.field.setNumber(4, "YMin", Dia/2-0.05*Dia)
        gmsh.model.mesh.field.setNumber(4, "YMax", Dia/2)
        gmsh.model.mesh.field.setNumber(4, "ZMin", 0)
        gmsh.model.mesh.field.setNumber(4, "ZMax", thick/2)
        gmsh.model.mesh.field.setNumber(4, "Thickness", 20*h)

        gmsh.model.mesh.field.add("Min", 5)
        gmsh.model.mesh.field.setNumbers(5, "FieldsList", [1,2,3,4])
        gmsh.model.mesh.field.setAsBackgroundMesh(5)

        gmsh.model.occ.synchronize()


        # Add physical tag 1 for corners
        boundary = gmsh.model.getBoundary([(3, cylinder_dim_tags)])
        boundary_ids = [b[1] for b in boundary]
        gmsh.model.addPhysicalGroup(1, boundary_ids, tag=1)
        gmsh.model.setPhysicalName(1, 1, "corners of cylinders")

        # Add physical tag 2 for surfaces
        gmsh.model.addPhysicalGroup(2, boundary_ids, tag=2)
        gmsh.model.setPhysicalName(2, 2, "surfaces of cylinders")

        # Add physical tag 3 for the volume
        volume_entities = [model[1] for model in gmsh.model.getEntities(3)]  ## here it just choose the main body
        gmsh.model.addPhysicalGroup(3, volume_entities, tag=3)
        gmsh.model.setPhysicalName(3, 3, "cylinder volume")

        # Generating Mesh
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(1)
        gmsh.model.mesh.optimize("Netgen")
        # gmsh.write("./results_mesh/brazilian.msh")

    model = comm.bcast(gmsh.model, root=0)

    mesh_partitioner = create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet)
    mesh, ct, ft = gmshio.model_to_mesh(model, comm, rank=0, gdim=3, partitioner=mesh_partitioner)
    mesh.name = "mesh"
    ct.name = f"{mesh.name}_cells"
    ft.name = f"{mesh.name}_facets"

    with XDMFFile(comm, "./results_mesh/brazilian_mesh3D.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)

    if comm.rank == 0:
        gmsh.finalize()

    return mesh

