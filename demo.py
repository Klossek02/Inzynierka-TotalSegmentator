from vedo import Volume, show, Plotter, load
from os import  listdir
import os
from stl import mesh
import nibabel as nib
import numpy as np
from skimage import measure


def convert_to_stl(path):
    np_array = nib.load(path).get_fdata()
    verts, faces, normals, values = measure.marching_cubes(np_array, 0)

    obj_3d = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces):
        obj_3d.vectors[i] = verts[f]
    obj_3d.save('stl_test.stl')

def render_3d(widget):

    volume = []
    colors_rgb = [
    (240, 248, 255),  # AliceBlue
    (250, 235, 215),  # AntiqueWhite
    (0, 255, 255),    # Aqua
    (127, 255, 212),  # Aquamarine
    (240, 255, 255),  # Azure
    (245, 245, 220),  # Beige
    (255, 228, 196),  # Bisque
    (255, 235, 205),  # BlanchedAlmond
    (0, 0, 255),      # Blue
    (138, 43, 226),   # BlueViolet
    (165, 42, 42),    # Brown
    (222, 184, 135),  # BurlyWood
    (95, 158, 160),   # CadetBlue
    (127, 255, 0),    # Chartreuse
    (210, 105, 30),   # Chocolate
    (255, 127, 80),   # Coral
    (100, 149, 237),  # CornflowerBlue
    (255, 248, 220),  # Cornsilk
    (220, 20, 60),    # Crimson
    (0, 255, 255),    # Cyan
    (0, 0, 139),      # DarkBlue
    (0, 139, 139),    # DarkCyan
    (184, 134, 11),   # DarkGoldenRod
    (169, 169, 169),  # DarkGray
    (0, 100, 0),      # DarkGreen
    (189, 183, 107),  # DarkKhaki
    (139, 0, 139),    # DarkMagenta
    (85, 107, 47),    # DarkOliveGreen
    (255, 140, 0),    # DarkOrange
    (153, 50, 204),   # DarkOrchid
    (139, 0, 0),      # DarkRed
    (233, 150, 122),  # DarkSalmon
    (143, 188, 143),  # DarkSeaGreen
    (72, 61, 139),    # DarkSlateBlue
    (47, 79, 79),     # DarkSlateGray
    (0, 206, 209),    # DarkTurquoise
    (148, 0, 211),    # DarkViolet
    (255, 20, 147),   # DeepPink
    (0, 191, 255),    # DeepSkyBlue
    (105, 105, 105),  # DimGray
    (30, 144, 255),   # DodgerBlue
    (178, 34, 34),    # FireBrick
    (255, 250, 240),  # FloralWhite
    (34, 139, 34),    # ForestGreen
    (255, 0, 255),    # Fuchsia
    (220, 220, 220),  # Gainsboro
    (248, 248, 255),  # GhostWhite
    (255, 215, 0),    # Gold
    (218, 165, 32),   # GoldenRod
    (128, 128, 128),  # Gray
    (0, 128, 0),      # Green
    (173, 255, 47),   # GreenYellow
    (240, 255, 240),  # HoneyDew
    (255, 105, 180),  # HotPink
    (205, 92, 92),    # IndianRed
    (75, 0, 130),     # Indigo
    (255, 255, 240),  # Ivory
    (240, 230, 140),  # Khaki
    (230, 230, 250),  # Lavender
    (255, 240, 245),  # LavenderBlush
    (124, 252, 0),    # LawnGreen
    (255, 250, 205),  # LemonChiffon
    (173, 216, 230),  # LightBlue
    (240, 128, 128),  # LightCoral
    (224, 255, 255),  # LightCyan
    (250, 250, 210),  # LightGoldenRodYellow
    (211, 211, 211),  # LightGray
    (144, 238, 144),  # LightGreen
    (255, 182, 193),  # LightPink
    (255, 160, 122),  # LightSalmon
    (32, 178, 170),   # LightSeaGreen
    (135, 206, 250),  # LightSkyBlue
    (119, 136, 153),  # LightSlateGray
    (176, 196, 222),  # LightSteelBlue
    (255, 255, 224),  # LightYellow
    (0, 255, 0),      # Lime
    (50, 205, 50),    # LimeGreen
    (250, 240, 230),  # Linen
    (255, 0, 255),    # Magenta
    (128, 0, 0),      # Maroon
    (102, 205, 170),  # MediumAquaMarine
    (0, 0, 205),      # MediumBlue
    (186, 85, 211),   # MediumOrchid
    (147, 112, 219),  # MediumPurple
    (60, 179, 113),   # MediumSeaGreen
    (123, 104, 238),  # MediumSlateBlue
    (0, 250, 154),    # MediumSpringGreen
    (72, 209, 204),   # MediumTurquoise
    (199, 21, 133),   # MediumVioletRed
    (25, 25, 112),    # MidnightBlue
    (245, 255, 250),  # MintCream
    (255, 228, 225),  # MistyRose
    (255, 228, 181),  # Moccasin
    (255, 222, 173),  # NavajoWhite
    (0, 0, 128),      # Navy
    (253, 245, 230),  # OldLace
    (128, 128, 0),    # Olive
    (107, 142, 35),   # OliveDrab
    (255, 165, 0),    # Orange
    (255, 69, 0),     # OrangeRed
    (218, 112, 214),  # Orchid
    (238, 232, 170),  # PaleGoldenRod
    (152, 251, 152),  # PaleGreen
    (175, 238, 238),  # PaleTurquoise
    (219, 112, 147),  # PaleVioletRed
    (255, 239, 213),  # PapayaWhip
    (255, 218, 185),  # PeachPuff
    (205, 133, 63),   # Peru
    (255, 192, 203),  # Pink
    (221, 160, 221),  # Plum
    (176, 224, 230),  # PowderBlue
    (128, 0, 128),    # Purple
    (102, 51, 153),   # RebeccaPurple
    (255, 0, 0),      # Red
    (188, 143, 143),  # RosyBrown
    (65, 105, 225),   # RoyalBlue
    (139, 69, 19),    # SaddleBrown
    (250, 128, 114),  # Salmon
    (244, 164, 96),   # SandyBrown
    (46, 139, 87),    # SeaGreen
    (255, 245, 238),  # SeaShell
    (160, 82, 45),    # Sienna
    (192, 192, 192),  # Silver
    (135, 206, 235),  # SkyBlue
    (106, 90, 205),   # SlateBlue
    (112, 128, 144),  # SlateGray
    (255, 250, 250),  # Snow
    (0, 255, 127),    # SpringGreen
    (70, 130, 180),   # SteelBlue
    (210, 180, 140),  # Tan
    (0, 128, 128),    # Teal
    (216, 191, 216),  # Thistle
    (255, 99, 71),    # Tomato
    (64, 224, 208),   # Turquoise
    (238, 130, 238),  # Violet
    (245, 222, 179),  # Wheat
    (245, 245, 245),  # WhiteSmoke
    (255, 255, 0),    # Yellow
    (154, 205, 50)    # YellowGreen
]
#TODO: change colors to be more distinct and maybe defined in a shorter way

    i = 0
    initial_path = 'C:/Users/Dell/Downloads/Totalsegmentator_dataset_v201/s0001/segmentations'
    for path in listdir(initial_path):
        vpath = initial_path + '/' + path
        size = os.stat(vpath).st_size
        if size > 45000:
            convert_to_stl(vpath)
        #vol.color(colors_rgb[i]
            vol = load('stl_test.stl').color(colors_rgb[i])
            volume.append(vol)
        i = i+1

    plotter = Plotter(qt_widget = widget)
    plotter.show(volume)
    return plotter