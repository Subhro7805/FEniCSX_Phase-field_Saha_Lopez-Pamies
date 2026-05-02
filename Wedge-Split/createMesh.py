import gmsh
from dolfinx.cpp.mesh import create_cell_partitioner
from dolfinx.io import gmshio, XDMFFile
import dolfinx

def meshwedge3D(comm, L, H, D, groove_depth, groove_length, notch, notch_width, h):
    if comm.rank == 0:
        # create mesh
        gmsh.initialize()
        gmsh.model.add("Mortarwedge_mesh3D")
        gmsh.model.setCurrent("Mortarwedge_mesh3D")
        gm = gmsh.model.occ

        outer_box = gm.addBox(0, 0, 0, L/2, H, D/2)

        groove_box = gm.addBox(0, H - groove_depth, 0, groove_length/2, groove_depth, D/2)

        notch_box = gm.addBox(0, H - groove_depth - notch, 0, notch_width/2, notch, D/2)

        box_dim_tags = gm.cut([(3, outer_box)], [(3, groove_box), (3, notch_box)])

        gm.synchronize()

        gmsh.model.mesh.field.add("Box", 1)
        gmsh.model.mesh.field.setNumber(1, "VIn", h )
        gmsh.model.mesh.field.setNumber(1, "VOut", 5*h)
        gmsh.model.mesh.field.setNumber(1, "XMin", 0)
        gmsh.model.mesh.field.setNumber(1, "XMax", 25*h)
        gmsh.model.mesh.field.setNumber(1, "YMin", H/3)
        gmsh.model.mesh.field.setNumber(1, "YMax", H)
        gmsh.model.mesh.field.setNumber(1, "ZMin", 0)
        gmsh.model.mesh.field.setNumber(1, "ZMax", D/2)
        gmsh.model.mesh.field.setNumber(1, "Thickness", 100*h)

        gmsh.model.mesh.field.add("Box", 2)
        gmsh.model.mesh.field.setNumber(2, "VIn", 2*h)
        gmsh.model.mesh.field.setNumber(2, "VOut", 5*h)
        gmsh.model.mesh.field.setNumber(2, "XMin", 0)
        gmsh.model.mesh.field.setNumber(2, "XMax", groove_length/2 + 5)
        gmsh.model.mesh.field.setNumber(2, "YMin", H - groove_depth/2)
        gmsh.model.mesh.field.setNumber(2, "YMax", H)
        gmsh.model.mesh.field.setNumber(2, "ZMin", 0)
        gmsh.model.mesh.field.setNumber(2, "ZMax", D/2)
        gmsh.model.mesh.field.setNumber(2, "Thickness", 100*h)


        gmsh.model.mesh.field.add("Min", 3)
        gmsh.model.mesh.field.setNumbers(3, "FieldsList", [1,2])
        gmsh.model.mesh.field.setAsBackgroundMesh(3)

        gm.synchronize()

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

    with XDMFFile(comm, "./results_mesh/Mortarwedge_mesh3D.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)

    if comm.rank == 0:
        gmsh.finalize()

    return mesh

