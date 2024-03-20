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
        
        q = self.Q(lidar.view(lidar.shape[0],lidar.shape[1],-1)).view(lidar.shape)
        k = self.K(cam.view(cam.shape[0],cam.shape[1],-1)).view(cam.shape)
        v = self.V(cam.view(cam.shape[0],cam.shape[1],-1)).view(cam.shape)
        
        dots = torch.einsum('a b c i,a b c j->a b i j', q,k) * self.scale
        
        att = dots.softmax(dim=-1)
        
        out = torch.einsum('a b i c,a b j c->a b j i', att,v)
        
        out = self.out(out.reshape(out.shape[0],out.shape[1],-1)).view(out.shape)
        
        return (out)