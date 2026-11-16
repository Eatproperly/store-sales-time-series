import torch
from torch import nn
class Block(nn.Module):
 def __init__(self,width=32):
  super().__init__();self.n1=nn.LayerNorm(width);self.n2=nn.LayerNorm(width)
  self.time=nn.Sequential(nn.Linear(112,32),nn.GELU(),nn.Dropout(.1),nn.Linear(32,112))
  self.feat=nn.Sequential(nn.Linear(width,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,width))
 def forward(self,x):
  x=x+self.time(self.n1(x).transpose(1,2)).transpose(1,2)
  return x+self.feat(self.n2(x))
class ForecastNet(nn.Module):
 def __init__(self,sizes,kind='mixer2',width=32):
  super().__init__();self.kind=kind;self.embed=nn.ModuleList([nn.Embedding(n,d) for n,d in zip(sizes,[8,8,16,4,4])]);self.static=nn.Linear(40,32)
  if kind=='mlp':self.encoder=nn.Sequential(nn.Flatten(),nn.Linear(112*8,128),nn.GELU(),nn.Linear(128,16*width))
  else:
   self.proj=nn.Linear(8,width);self.blocks=nn.Sequential(*[Block(width) for _ in range(3 if kind=='mixer3' else 2)]);self.time_out=nn.Linear(112,16)
  self.future=nn.Linear(12,16);self.horizon=nn.Embedding(16,8)
  self.head=nn.Sequential(nn.Linear(width+32+16+8,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))
  self.width=width
  nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
 def forward(self,h,f,s,anchor):
  embeddings=[e(s[:,i]) for i,e in enumerate(self.embed)]
  if self.kind=='no_pair':embeddings[2]=embeddings[2]*0
  static=self.static(torch.cat(embeddings,1))[:,None].expand(-1,16,-1)
  if self.kind=='no_promo':f=f.clone();f[:,:,0]=0
  if self.kind=='mlp':enc=self.encoder(h).reshape(-1,16,self.width)
  else:enc=self.time_out(self.blocks(self.proj(h)).transpose(1,2)).transpose(1,2)
  horizon=self.horizon(torch.arange(16,device=h.device))[None].expand(len(h),-1,-1)
  delta=self.head(torch.cat([enc,static,self.future(f),horizon],-1)).squeeze(-1)
  return delta if self.kind=='no_anchor' else anchor+delta
