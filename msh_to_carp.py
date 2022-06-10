import os
import csv

import numpy as np
import trimesh


def convert_mesh_file(filename, ele_dict={1:"Ln", 2:"Tr", 3:"Qd", 4:"Tt"}):

    Nodes = []
    Elements = []
    Elements_carp = []

    read_nodes = False
    read_elements = False
    ele_count = 0
    tag_id = -1
    with open(filename) as file:
        for line in file:
            l_striped = line.rstrip()
            # Check identifier
            if "$" in l_striped:
                if l_striped == "$Nodes":
                    read_nodes = True
                    continue
                elif l_striped == "$EndNodes":
                    read_nodes = False
                    continue
                elif l_striped == "$Elements":
                    read_elements = True
                    continue
                elif l_striped == "$EndElements":
                    read_elements = False
                    continue

            # Read nodes or elements
            if read_nodes:
                point = l_striped.split(" ")
                if len(point) == 3:
                    Nodes.append(point)
            elif read_elements:
                element = list(np.array(l_striped.split(" ")).astype(int) - 1)
                ele_count += 1
                if (int(element[0]) + 1 > 3 and ele_count > 2) or (ele_count == 3 or ele_count == 4 or ele_count == 5):
                    Elements.append(element[1:])
                    Elements_carp.append([ele_type] + element[1:] + [str(tag_id)])
                else:
                    ele_type = ele_dict[list(np.array(l_striped.split(" ")).astype(int))[2]]
                    tag_id += 1

    return Nodes, Elements, Elements_carp


def write_carp_files(outdir, filename, nodes, elements):
    out_path = outdir + filename.split(".")[0] + "/"
    try:
        os.mkdir(out_path)
    except FileExistsError:
        pass

    with open(out_path + filename.split(".")[0] + ".pts", "w", newline='') as f:
        f.write(str(len(nodes)) + "\n")
        wr = csv.writer(f, delimiter=" ")
        wr.writerows(nodes)
    with open(out_path + filename.split(".")[0] + ".elem", "w", newline='') as f:
        f.write(str(len(elements)) + "\n")
        wr = csv.writer(f, delimiter=" ")
        wr.writerows(elements)


out_dir = "LV_Models_Test3/"
filename = "lv_myo_volume_mesh_refined_2.msh"
nodes, elements, Elements_carp = convert_mesh_file(filename)
write_carp_files(out_dir, filename, nodes, Elements_carp)

# nodes = np.array(nodes)
# elements = np.array(elements)
# elements = np.delete(elements, 0, 1)
# elements = np.delete(elements, -1, 1)

# mesh = trimesh.Trimesh(vertices=nodes,
#                        faces=elements)
# trimesh.exchange.export.export_mesh(mesh, out_dir + filename + "_test.stl", file_type="stl")
