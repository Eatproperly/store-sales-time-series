import unittest
import numpy as np
import torch
from store_sales.models.mixer import ForecastNet
from store_sales.evaluation import score
class NeuralTests(unittest.TestCase):
 def test_shape_finite_and_anchor_init(self):
  torch.set_num_threads(2)
  for kind in ['mlp','mixer2','mixer3','no_pair','no_promo','no_anchor']:
   m=ForecastNet([2,3,6,2,2],kind).eval()
   anchor=torch.ones(3,16)
   result=m(torch.zeros(3,112,8),torch.zeros(3,16,12),torch.zeros(3,5,dtype=torch.long),anchor)
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
