import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from store_sales.models.mixer import ForecastNet
from store_sales.evaluation import score
class NeuralTests(unittest.TestCase):
 def test_shape_finite_and_anchor_init(self):
  torch.set_num_threads(2)
  for kind in ['mlp','mixer2','mixer3','no_pair','no_promo','no_anchor','mixer3_long','mixer3_scale']:
   m=ForecastNet([2,3,6,2,2],kind).eval()
   anchor=torch.ones(3,16)
   result=m(torch.zeros(3,112,8),torch.zeros(3,16,14),torch.zeros(3,5,dtype=torch.long),anchor)
   self.assertEqual(tuple(result.shape),(3,16));self.assertTrue(torch.isfinite(result).all())
   self.assertTrue(torch.allclose(result,torch.zeros_like(anchor) if kind=='no_anchor' else anchor))
 def test_masked_metric_and_clipping(self):
  s=score(np.array([0.,1.,100.]),np.array([-1.,1.,0.]),np.array([True,True,False]))
  self.assertEqual(s['RMSLE'],0.)

class AsOfTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  from pathlib import Path
  from store_sales.data import Panel
  root=Path(__file__).resolve().parents[1]
  if not (root/'data/train.csv').exists():raise unittest.SkipTest('原始数据未放入data/')
  cls.p=Panel(root)
 def test_future_observations_invariance_and_known_promo(self):
  p=self.p;t=p.index('2017-07-14');orig=np.full(8,t);ss=np.arange(8)
  before=p.inputs(orig,ss)
  oldz=p.z[t+1:].copy();oldh=p.history[t+1:].copy()
  p.z[t+1:]+=99;p.history[t+1:]+=73
  after=p.inputs(orig,ss);p.z[t+1:]=oldz;p.history[t+1:]=oldh
  for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
  p.promo[t+1,0]+=5;changed=p.inputs(orig,ss);p.promo[t+1,0]-=5
  self.assertFalse(np.array_equal(before[1],changed[1]))
  m=ForecastNet(p.sizes).eval()
  with torch.no_grad():
   np.testing.assert_array_equal(m(*[torch.from_numpy(x) for x in before]).numpy(),m(*[torch.from_numpy(x) for x in after]).numpy())
 def test_calendar_missing_and_cutoff(self):
  p=self.p
  self.assertFalse(p.mask[p.index('2016-12-25')].any())
  self.assertTrue((p.mask & (p.y==0)).any())
  for date in ['2016-08-16','2017-04-16','2017-06-15','2017-07-15']:
   cut=p.index(date)-1;orig=p.origins(p.dates[cut])
   self.assertTrue((orig+16<=cut).all());self.assertTrue((orig>=112).all())

 def test_tree_statistics_ignore_missing_placeholder(self):
  p=self.p;t=p.index('2017-07-14')
  oldz=p.z[t,0].copy();oldmask=p.mask[t,0].copy()
  try:
   p.z[t,0]=999;p.mask[t,0]=False
   x,_,_=p.tree_rows([t])
   self.assertTrue(np.isnan(x[0,4]))
   self.assertAlmostEqual(float(x[0,10]),float(p.z[t-6:t,0].mean()),places=5)
  finally:p.z[t,0]=oldz;p.mask[t,0]=oldmask

class TrainingTests(unittest.TestCase):
 def test_patience_stops_on_constant_inner_score(self):
  import pandas as pd
  from store_sales.training import train
  class Tiny:
   sizes=[2,2,2,2,2];n=2;windows_per_epoch=2;batch_size=2
   dates=pd.date_range('2017-01-01',periods=200)
   z=np.zeros((200,2),dtype='float32');mask=np.ones((200,2),dtype=bool)
   def origins(self,cutoff,start):return np.array([112])
   def inputs(self,tt,ss):
    n=len(ss)
    return np.zeros((n,112,8),dtype='float32'),np.zeros((n,16,12),dtype='float32'),np.zeros((n,5),dtype='int64'),np.zeros((n,16),dtype='float32')
  _,curve=train(Tiny(),pd.Timestamp('2017-06-01'),'mixer2',epochs=10,validation_origin=112,patience=2)
  self.assertEqual(len(curve),3)
 def test_gpu_protocol_refuses_silent_cpu_fallback(self):
  if torch.cuda.is_available():self.skipTest('CUDA device present')
  import subprocess
  root=Path(__file__).resolve().parents[1]
  r=subprocess.run([sys.executable,str(root/'scripts/run_advanced.py'),'--config',str(root/'configs/advanced-gpu.yaml')],cwd=root,text=True,capture_output=True)
  self.assertNotEqual(r.returncode,0);self.assertIn('requires a CUDA GPU',r.stderr)
