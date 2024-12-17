# === Model.py ===

import monai
from monai.networks.nets import UNet
def get_unet_model(num_classes=118, in_channels=1):
    return UNet(
        spatial_dims=3,                  # spatial dimensions of the input (3D for volumetric data)
        in_channels=in_channels,         # no. of input channels (e.g, 1 or 2 for single-channel images)
        out_channels=num_classes,        # no. of output channels (e.g, number of segmentation classes)
        channels=(32, 64, 128, 256, 512), # no. of channels in each layer of the encoder/decoder
        strides=(2, 2, 2, 2),            # stride for each downsampling operation in the encoder
        num_res_units=2,                 # no. of residual units in each layer
        norm=monai.networks.layers.Norm.BATCH, # normalization layer to use (batch normalization)
        dropout=0.2,                     # dropout rate for regularization
    )
