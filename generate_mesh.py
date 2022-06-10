import trimesh
import os
import gmsh
import json
from stl import mesh
import numpy as np
import pyvista as pv


# Resolve intersection and smooth surface mesh
def mesh_postprocessing(path):
    # Postprocessing
    myo_mesh = trimesh.exchange.load.load(path + "Myo_surface.stl")
    lv_mesh = trimesh.exchange.load.load(path + "LV_surface.stl")

    trimesh.smoothing.filter_humphrey(myo_mesh)
    trimesh.smoothing.filter_humphrey(lv_mesh)
    trimesh.repair.fix_normals(myo_mesh)
    trimesh.repair.fix_normals(lv_mesh)

    processed_lv = trimesh.boolean.intersection([lv_mesh, myo_mesh])

    # Rescale
    center_before = processed_lv.center_mass
    processed_lv.vertices = processed_lv.vertices * 0.98
    center_after = processed_lv.center_mass

    # Recenter
    shift = center_before - center_after
    processed_lv.vertices[:, 0] = processed_lv.vertices[:, 0] + shift[0]
    processed_lv.vertices[:, 1] = processed_lv.vertices[:, 1] + shift[1]
    processed_lv.vertices[:, 2] = processed_lv.vertices[:, 2] + shift[2]
    trimesh.exchange.export.export_mesh(processed_lv, path + "processed_lv.stl", file_type="stl")
    trimesh.exchange.export.export_mesh(myo_mesh, path + "processed_myo.stl", file_type="stl")


# Increase mesh resolution
def refine_mesh(file_path, save_path, Iter=0, name="lv_myo_volume_mesh_refined", save_type=".msh"):
    # gmsh.initialize()
    # gmsh.open(file_path)

    # new_mesh_res = 0.2 / 1.7
    # mesh1 = trimesh.exchange.load.load("processed_lv.stl")
    # mesh2 = trimesh.exchange.load.load("processed_myo.stl")
    # closest_pts, distances, tri_id = trimesh.proximity.closest_point(mesh1, mesh2.vertices)
    # hausdorff = np.round(np.max(distances), 4)
    # if Iter == 0:
    #     Iter = int(np.round(np.log(hausdorff / new_mesh_res) / np.log(2))) - 2

    gmsh.initialize()
    gmsh.open(file_path)
    for i in range(Iter):
        gmsh.model.mesh.refine()
        gmsh.model.mesh.optimize("")
        gmsh.model.mesh.optimize("Netgen")
        gmsh.model.mesh.optimize("")
        gmsh.model.mesh.optimize("Netgen")

    gmsh.write(save_path+name + save_type)
    gmsh.finalize()


def generate_3d_surface_mesh(path):
    os.system('generate_surface_mesh.exe')
    mesh_postprocessing(path)


def generate_2d_surface_mesh(path, refine_iter):
    lv_points = np.loadtxt(path+"LV_point_cloud.xyz", dtype='float', delimiter=",")[:, 0:3]
    myo_points = np.loadtxt(path+"Myo_point_cloud.xyz", dtype='float', delimiter=",")[:, 0:3]

    increment_interval = int(len(lv_points) / len(np.unique(lv_points[:, 2])))
    increment_list = [i for i in range(0, len(lv_points) + increment_interval, increment_interval)]
    try:
        os.mkdir(path+"2d_surface_meshes")
    except FileExistsError:
        pass
    try:
        os.mkdir(path+"2d_surface_meshes_refined")
    except FileExistsError:
        pass

    for i in range(len(increment_list) - 1):
        P1 = lv_points[increment_list[i]:increment_list[i + 1], :]
        P2 = myo_points[increment_list[i]:increment_list[i + 1], :]
        P3 = np.append(P1, P2, 0)
        dist = abs(np.max(P1[:, 1]) - np.max(P2[:, 1]))
        mesh = pv.PolyData(P3).delaunay_2d(tol=6e-2, alpha=dist)
        # mesh.plot(show_edges=True)
        # mesh1 = pv.PolyData(P1).delaunay_2d()
        # mesh2 = pv.PolyData(P2).delaunay_2d()
        # mesh1.plot()
        # mesh2.plot()
        mesh.save(path+"2d_surface_meshes/"+'lv_myo_2d_surface_mesh'+str(i+1)+".stl")
        refine_mesh(path+"2d_surface_meshes/"+'lv_myo_2d_surface_mesh'+str(i+1)+".stl",
                    path+"2d_surface_meshes_refined/", refine_iter, "lv_myo_2d_surface_mesh_refined"+str(i+1), ".stl")


def generate_3d_volume_mesh(path):
    gmsh.initialize()
    gmsh.open(path+"processed_lv.stl")
    gmsh.merge(path+"processed_myo.stl")

    n = gmsh.model.getDimension()
    s = gmsh.model.getEntities(n)
    l = gmsh.model.geo.addSurfaceLoop([s[i][1] for i in range(len(s))])

    gmsh.model.geo.addVolume([l])
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(3)

    gmsh.model.mesh.renumberNodes()
    for i in range(6):
        gmsh.model.mesh.optimize("")
        gmsh.model.mesh.optimize("Netgen")

    # refined_mesh = gmsh.model.mesh.refine()

    gmsh.write(path+"lv_myo_volume_mesh.msh")
    gmsh.finalize()


# Shortcut for generating all 2D surface, and 3D surface & volume meshes
def generate_meshes(working_path):
    parameter_dict = json.loads(open('mesh_parameters.json', 'r').read())

    # 2D
    print("Generating 2D surface meshes...")
    generate_2d_surface_mesh(working_path, parameter_dict["REFINEITER2D"][0])

    # 3D
    print("Generating 3D surface meshes...")
    generate_3d_surface_mesh(working_path)
    mesh_postprocessing(working_path)

    print("Generating 3D volume mesh...")
    generate_3d_volume_mesh(working_path)

    print("Refining meshes...")
    refine_mesh(working_path + "lv_myo_volume_mesh.msh", working_path, parameter_dict["REFINEITER3D"][0])
    print("All mesh generations finished")