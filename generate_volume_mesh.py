import gmsh
import json

gmsh.initialize()
parameter_dict = json.loads(open('parameters.json', 'r').read())
path = "Output/" + parameter_dict["name"].split(".")[0] + "/"

gmsh.open(path + "LV_surface.stl")
gmsh.merge(path + "Myo_surface.stl")


n = gmsh.model.getDimension()
s = gmsh.model.getEntities(n)
l = gmsh.model.geo.addSurfaceLoop([s[i][1] for i in range(len(s))])

# gmsh.model.occ.intersect([(3, 1)], [(3, 2)], 3)
gmsh.model.geo.addVolume([l])
gmsh.model.geo.synchronize()

gmsh.model.mesh.generate(3)

gmsh.model.mesh.renumberNodes()
gmsh.model.mesh.optimize("")
gmsh.model.mesh.optimize("Netgen")

gmsh.write(path + "lv_myo_volume_mesh.msh")
gmsh.finalize()


