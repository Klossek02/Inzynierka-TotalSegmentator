# === demo.py ===

import numpy as np
from stl import mesh as stl_mesh
from skimage import measure

# converts a numpy array (NIfTI data) to STL file using marching cubes algorithm:  https://www.cs.carleton.edu/cs_comps/0405/shape/marching_cubes.html
def convert_to_stl(nifti, out_path):

    if np.max(nifti) == 0:
        print(f"WARNING: the input data for {out_path} is empty.")
        return

    # using marching cubes algo with lvl 0.5 for binary masks
    verts, faces, normals, values = measure.marching_cubes(nifti)

    # creating mesh
    obj_3d = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            obj_3d.vectors[i][j] = verts[f[j], :]

    # saving mesh to file
    obj_3d.save(out_path)
    print(f"STL file saved at: {out_path}")
    #return obj_3d
