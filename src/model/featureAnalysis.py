import torch
import torch.nn as nn

from external.UNet3D.pytorch3dunet.unet3d.model import ResidualUNet3D, UNet3D


class featureAnalysis(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        inchannels = 1280
        outchannels = 64
        self.model = ResidualUNet3D(in_channels=inchannels, out_channels=outchannels, pool_kernel_size=1)
        
    def forward(self,x):
        
        x = self.model(x)
        
        return(x)
        