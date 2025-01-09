# === demo.py ===

import numpy as np

from stl import mesh as stl_mesh
from skimage import measure


# function converting a numpy array (NIfTI data) to STL file using marching cubes algorithm:  https://www.cs.carleton.edu/cs_comps/0405/shape/marching_cubes.html
def convert_to_stl(nifti, out_path):

    if np.max(nifti) == 0:
        print(f"WARNING: the input data for {out_path} is empty.")
        return

    # using marching cubes algorithm to extract a surface mesh from the 3D image: 
    # We use level parameter 0.5:
    verts, faces, normals, values = measure.marching_cubes(nifti, level=0.5)

    # creating an empty STL mesh object; source: https://pypi.org/project/numpy-stl/
    obj_3d = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype)) # number of faces determines the mesh size
    
    # now, we are going to fill (populate) the mesh with vertices based on the faces returned by the marching cubes algorithm: 
    for i, f in enumerate(faces):
        for j in range(3): # each face has 3 vertices
            obj_3d.vectors[i][j] = verts[f[j], :] # assign vertex coordinates for the face 

    # saving mesh to a given file: 
    obj_3d.save(out_path)
    print(f"STL file saved at: {out_path}")  
