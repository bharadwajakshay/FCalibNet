import torch.nn as nn
import torch

class crossAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        inchannels = 1024
        
        self.Q = nn.Conv1d(inchannels, inchannels, kernel_size=1, bias=False)
        self.K = nn.Conv1d(inchannels, inchannels, kernel_size=1, bias=False)
        self.V = nn.Conv1d(inchannels, inchannels, kernel_size=1, bias=False)
        
        self.out = nn.Sequential(nn.Conv1d(inchannels, inchannels, kernel_size=1, bias=False),
                                 nn.GELU())
        
        self.scale = 64 ** -0.5
        
    def forward(self,lidar,cam):
        
        q = self.Q(lidar.reshape(lidar.shape[0],lidar.shape[1],-1)).view(lidar.shape)
        k = self.K(cam.reshape(cam.shape[0],cam.shape[1],-1)).view(cam.shape)
        v = self.V(cam.reshape(cam.shape[0],cam.shape[1],-1)).view(cam.shape)
        
        dots = torch.einsum('b c i d,b c j d->b c i j', q,k) * self.scale
        
        att = dots.softmax(dim=-1)
        
        out = torch.einsum('b c i d,b c k j->b c i j', att,v)
        
        out = self.out(out.view(out.shape[0],out.shape[1],-1)).view(out.shape)
        
        return (out)