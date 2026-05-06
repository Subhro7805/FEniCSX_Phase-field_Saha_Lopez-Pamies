import gmsh
from dolfinx.cpp.mesh import create_cell_partitioner
from dolfinx.io import gmshio, XDMFFile
import dolfinx

def mesh4pt(comm, L, H, D, Span, x_load, h):
    if comm.rank == 0:
        # create mesh
        gmsh.initialize()
        gmsh.model.add("Mortar4pt_mesh3D")
        gmsh.model.setCurrent("Mortar4pt_mesh3D")
        gm = gmsh.model.occ

        box_dim_tags = gm.addBox(-L/2, 0, 0, L, H, D/2)
        gm.synchronize()

        gmsh.model.mesh.field.add("Box", 1)
        gmsh.model.mesh.field.setNumber(1, "VIn", h )
        gmsh.model.mesh.field.setNumber(1, "VOut", 1.5)
        gmsh.model.mesh.field.setNumber(1, "XMin", -15*h)
        gmsh.model.mesh.field.setNumber(1, "XMax", 50+15*h)
        gmsh.model.mesh.field.setNumber(1, "YMin", 0)
        gmsh.model.mesh.field.setNumber(1, "YMax", H)
        gmsh.model.mesh.field.setNumber(1, "ZMin", 0)
        gmsh.model.mesh.field.setNumber(1, "ZMax", D/2)
        gmsh.model.mesh.field.setNumber(1, "Thickness", 0.1*L)

        gmsh.model.mesh.field.add("Box", 2)
        gmsh.model.mesh.field.setNumber(2, "VIn", h )
        gmsh.model.mesh.field.setNumber(2, "VOut", 1.5)
        gmsh.model.mesh.field.setNumber(2, "XMin", Span/2-x_load/2-5.0)
        gmsh.model.mesh.field.setNumber(2, "XMax", Span/2+x_load/2+5.0)
        gmsh.model.mesh.field.setNumber(2, "YMin", 0)
        gmsh.model.mesh.field.setNumber(2, "YMax", H/5)
        gmsh.model.mesh.field.setNumber(2, "ZMin", 0)
        gmsh.model.mesh.field.setNumber(2, "ZMax", D/2)
        gmsh.model.mesh.field.setNumber(2, "Thickness", 0.1*L)

        gmsh.model.mesh.field.add("Box", 3)
        gmsh.model.mesh.field.setNumber(3, "VIn", h )
        gmsh.model.mesh.field.setNumber(3, "VOut", 1.5)
        gmsh.model.mesh.field.setNumber(3, "XMin", Span/6 - x_load/2 - 5.0)
        gmsh.model.mesh.field.setNumber(3, "XMax", Span/6 + x_load/2 + 5.0)
        gmsh.model.mesh.field.setNumber(3, "YMin", H-H/5)
        gmsh.model.mesh.field.setNumber(3, "YMax", H)
        gmsh.model.mesh.field.setNumber(3, "ZMin", 0)
        gmsh.model.mesh.field.setNumber(3, "ZMax", D/2)
        gmsh.model.mesh.field.setNumber(3, "Thickness", 0.1*L)

        gmsh.model.mesh.field.add("Box", 4)
        gmsh.model.mesh.field.setNumber(4, "VIn", h )
        gmsh.model.mesh.field.setNumber(4, "VOut", 1.5)
        gmsh.model.mesh.field.setNumber(4, "XMin", -Span/2-x_load/2-5.0)
        gmsh.model.mesh.field.setNumber(4, "XMax", -Span/2+x_load/2+5.0)
        gmsh.model.mesh.field.setNumber(4, "YMin", 0)
        gmsh.model.mesh.field.setNumber(4, "YMax", H/5)
        gmsh.model.mesh.field.setNumber(4, "ZMin", 0)
        gmsh.model.mesh.field.setNumber(4, "ZMax", D/2)
        gmsh.model.mesh.field.setNumber(4, "Thickness", 0.1*L)

        gmsh.model.mesh.field.add("Box", 5)
        gmsh.model.mesh.field.setNumber(5, "VIn", h )
        gmsh.model.mesh.field.setNumber(5, "VOut", 1.5)
        gmsh.model.mesh.field.setNumber(5, "XMin", -Span/6 - x_load/2 - 5.0)
        gmsh.model.mesh.field.setNumber(5, "XMax", -Span/6 + x_load/2 + 5.0)
        gmsh.model.mesh.field.setNumber(5, "YMin", H-H/5)
        gmsh.model.mesh.field.setNumber(5, "YMax", H)
        gmsh.model.mesh.field.setNumber(5, "ZMin", 0)
        gmsh.model.mesh.field.setNumber(5, "ZMax", D/2)
        gmsh.model.mesh.field.setNumber(5, "Thickness", 0.1*L)

        gmsh.model.mesh.field.add("Min", 6)
        gmsh.model.mesh.field.setNumbers(6, "FieldsList", [1,2,3,4,5])
        gmsh.model.mesh.field.setAsBackgroundMesh(6)

        gm.synchronize()

        # Add physical tag 1 for corners
        boundary = gmsh.model.getBoundary([(3, box_dim_tags)])
        boundary_ids = [b[1] for b in boundary]
        gmsh.model.addPhysicalGroup(1, boundary_ids, tag=1)
        gmsh.model.setPhysicalName(1, 1, "corners of box")

        # Add physical tag 2 for surfaces
        gmsh.model.addPhysicalGroup(2, boundary_ids, tag=2)
        gmsh.model.setPhysicalName(2, 2, "surfaces of box")

        # Add physical tag 3 for the volume
        volume_entities = [model[1] for model in gmsh.model.getEntities(3)]  ## here it just choose the main body
        gmsh.model.addPhysicalGroup(3, volume_entities, tag=3)
        gmsh.model.setPhysicalName(3, 3, "box volume")

        # Generating Mesh
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(1)
        gmsh.model.mesh.optimize("Netgen")
        # gmsh.write("./results/bend.msh")

    model = comm.bcast(gmsh.model, root=0)

    mesh_partitioner = create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet)
    mesh, ct, ft = gmshio.model_to_mesh(model, comm, rank=0, gdim=3, partitioner=mesh_partitioner)
    mesh.name = "mesh"
    ct.name = f"{mesh.name}_cells"
    ft.name = f"{mesh.name}_facets"

    with XDMFFile(comm, "./results_mesh/Mortar4pt_mesh3D.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)


    if comm.rank == 0:
        gmsh.finalize()

    return mesh

