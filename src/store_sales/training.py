import time,random,json
import numpy as np
import torch
from .models.mixer import ForecastNet

def predict(model,panel,t,batch=256):
 model.eval();outs=[]
 with torch.no_grad():
  for a in range(0,panel.n,batch):
   ss=np.arange(a,min(a+batch,panel.n)); inp=panel.inputs(np.full(len(ss),t),ss)
   outs.append(model(*[torch.from_numpy(x) for x in inp]).numpy())
 return np.concatenate(outs)

def train(panel,cutoff,kind,epochs=8,seed=42,start=None,width=32):
 torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
 model=ForecastNet(panel.sizes,kind,width);optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
 origins=panel.origins(cutoff,start or getattr(panel,'train_start','2015-01-01')); ids=np.arange(len(origins)*panel.n)
 rng=np.random.default_rng(seed); curves=[];timer=time.perf_counter()
 for epoch in range(epochs):
  model.train();rng.shuffle(ids);loss_sum=0.;n=0
  epoch_ids=ids[:4096]
  for a in range(0,len(epoch_ids),256):
   ii=epoch_ids[a:a+256];tt=origins[ii//panel.n];ss=ii%panel.n
   inp=panel.inputs(tt,ss); pred=model(*[torch.from_numpy(x) for x in inp])
   days=tt[:,None]+np.arange(1,17);y=torch.from_numpy(panel.z[days,ss[:,None]]);mask=torch.from_numpy(panel.mask[days,ss[:,None]])
   loss=((pred-y).square()*mask).sum()/mask.sum().clamp(min=1)
   optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
   loss_sum+=loss.item()*int(mask.sum());n+=int(mask.sum())
  curves.append({'epoch':epoch+1,'train_log_mse':loss_sum/max(n,1),'seconds':time.perf_counter()-timer})
  print(kind,cutoff,'epoch',epoch+1,'loss',round(curves[-1]['train_log_mse'],5),'sec',round(curves[-1]['seconds'],1),flush=True)
 return model,curves

def outer_fit(panel,t,kind,max_epochs=8,seed=42,width=32):
 # Internal 16-day validation selects epoch; outer validation remains unused.
 inner_t=t-16
 best=float('inf');best_epoch=1
 # Train sequentially once, retain internal validation curve and epoch checkpoints.
 # For bounded CPU execution use independently trained epoch budgets 2 and max_epochs.
 trials=sorted(set([2,max_epochs]));inner=[]
 from .evaluation import score
 for ep in trials:
  m,c=train(panel,panel.dates[inner_t],kind,ep,seed,width=width)
  z=predict(m,panel,inner_t);value=score(panel.z[inner_t+1:inner_t+17].T,z,panel.mask[inner_t+1:inner_t+17].T)['RMSLE']
  inner.append({'epochs':ep,'inner_rmsle':value})
  if value<best:best=value;best_epoch=ep
 model,curve=train(panel,panel.dates[t],kind,best_epoch,seed,width=width)
 return model,{'inner_trials':inner,'selected_epochs':best_epoch,'training_curve':curve,'parameters':sum(p.numel() for p in model.parameters()),'seed':seed,'width':width,'kind':kind}
