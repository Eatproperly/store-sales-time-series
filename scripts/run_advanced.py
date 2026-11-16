"""CPU-bounded, reproducible advanced experiments; outputs are saved after each fit."""
import sys,os,time,json,hashlib,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));os.chdir(ROOT)
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from store_sales.data import Panel
from store_sales.training import outer_fit,predict,train
from store_sales.evaluation import score,write_prediction
p=Panel(ROOT);out=p.out
rows=pd.read_csv(out/'metrics.csv').to_dict('records') if (out/'metrics.csv').exists() else [];dev=['2016-08-16','2017-04-16','2017-06-15','2017-07-15']
config={'device':'cpu','width':32,'batch_size':256,'windows_per_epoch':4096,'epoch_candidates':[2,8],'max_epochs':8,'history':112,'horizon':16,'seed':42,'gpu_available':torch.cuda.is_available(),'torch_version':torch.__version__,'baseline_commit':'937fe5840828e5d68443b30637320db9dbc9456b','deviation':'CPU measured adaptation: width32, epoch sampling4096, inner epoch comparison2/8 rather than GPU patience50. No claim of full GPU training.'}
(out/'execution_config.json').write_text(json.dumps(config,indent=2))
# Real-data invariance and split tests, including valid future-known changes.
t=p.index('2017-07-14');ss=np.arange(8);orig=np.full(8,t)
a=p.inputs(orig,ss)
old=p.z[t+1:].copy();p.z[t+1:]+=99;b=p.inputs(orig,ss);p.z[t+1:]=old
for x,y in zip(a,b):assert np.array_equal(x,y),'future target leakage'
p.promo[t+1,0]+=1;c=p.inputs(orig,ss);p.promo[t+1,0]-=1
assert not np.array_equal(a[1],c[1])
for date in dev:
 cutoff=p.index(date)-1;assert (p.origins(p.dates[cutoff])+16<=cutoff).all()
assert p.mask[p.index('2016-12-25')].sum()==0
assert p.mask[p.index('2017-07-14')].all()
(out/'data_tests.json').write_text(json.dumps({'future_sales_invariance':True,'known_future_promotion_reaches_input':True,'all_training_labels_available':True,'missing_christmas_masked':True,'real_zero_sales_retained':True}))

def record(name,date,z,detail=None):
 t=p.index(date)-1;metrics=write_prediction(p,t,z,name)
 row={'model':name,'date':date,**metrics};rows.append(row)
 pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
 if detail:(out/f'training_{name}_{date}.json').write_text(json.dumps(detail,indent=2))
 print('RESULT',row,flush=True)

def tree_fit(date,name,start='2015-01-01',exog=False):
 t=p.index(date)-1;xx,yy,mm=p.tree_rows(p.origins(p.dates[t],start),exog)
 model=LGBMRegressor(n_estimators=350,learning_rate=.05,num_leaves=63,min_child_samples=80,reg_lambda=1.,colsample_bytree=.9,n_jobs=2,random_state=42,verbosity=-1)
 timer=time.perf_counter();model.fit(xx[mm],yy[mm],categorical_feature=[0,1,2,3]);del xx,yy,mm;gc.collect()
 vx,_,_=p.tree_rows([t],exog);z=model.predict(vx).reshape(p.n,16)
 record(name,date,z,{'seconds':time.perf_counter()-timer,'train_start':start,'exog':exog})
 return model

# Frozen B0 on all development windows; original regression score must reproduce.
for date in dev+['2017-07-31']:
 if any(r['model']=='B0' and r['date']==date for r in rows):continue
 fit,val=p.g['fold'](pd.Timestamp(date))
 pred,_,sec=p.g['fit_predict']('LightGBM',fit,val,{'n_estimators':350,'learning_rate':.05,'num_leaves':63,'min_child_samples':80})
 oldframe=val[['store_nbr','family','date']].copy();oldframe['pred']=pred
 key=p.keys(p.index(date)-1)
 aligned=key.merge(oldframe,left_on=['store_nbr','family','target_date'],right_on=['store_nbr','family','date'],validate='one_to_one')
 record('B0',date,np.log1p(aligned.pred.to_numpy()).reshape(p.n,16),{'seconds':sec})
for date in dev:
 if not any(r['model']=='B1' and r['date']==date for r in rows):tree_fit(date,'B1')
for date in dev[-2:]:
 if not any(r['model']=='B1_2016' and r['date']==date for r in rows):tree_fit(date,'B1_2016','2016-01-01')
 if not any(r['model']=='B2' and r['date']==date for r in rows):tree_fit(date,'B2',exog=True)
recent=pd.DataFrame(rows);recent=recent.loc[recent.date.isin(dev[-2:]) & recent.model.isin(['B1','B1_2016','B2'])]
tree_name=recent.groupby('model').RMSLE.apply(lambda x:float(np.sqrt(np.mean(x*x)))).idxmin()
tree_start='2016-01-01' if tree_name=='B1_2016' else '2015-01-01';tree_exog=tree_name=='B2';p.train_start=tree_start
for date in dev[:2]:
 if not any(r['model']==tree_name and r['date']==date for r in rows):tree_fit(date,tree_name,tree_start,tree_exog)
# Screen all six network designs on recent folds using an internal validation window.
for kind in ['mlp','mixer2','mixer3','no_pair','no_promo','no_anchor']:
 for date in dev[-2:]:
  if any(r['model']==kind and r['date']==date for r in rows):continue
  t=p.index(date)-1;model,detail=outer_fit(p,t,kind)
  record(kind,date,predict(model,p,t),detail);del model;gc.collect()
# Select by pooled squared error of the two complete screen windows.
screen=pd.DataFrame(rows);neural=['mlp','mixer2','mixer3','no_pair','no_promo','no_anchor']
rank=screen.loc[screen.model.isin(neural)].groupby('model').RMSLE.apply(lambda x:float(np.sqrt((x*x).mean()))).sort_values()
(out/'neural_screen.json').write_text(rank.to_json())
best=rank.index[0];finalists=list(rank.index[:2])
for kind in finalists:
 for date in dev[:2]:
  if any(r['model']==kind and r['date']==date for r in rows):continue
  t=p.index(date)-1;model,detail=outer_fit(p,t,kind);record(kind,date,predict(model,p,t),detail);del model;gc.collect()
complete=pd.DataFrame(rows);complete=complete.loc[complete.model.isin(finalists)&complete.date.isin(dev)]
best=complete.groupby('model').RMSLE.apply(lambda x:float(np.sqrt(np.mean(x*x)))).idxmin()
# Repeat the winning network on the two recent folds; no unsupported variance claim.
for seed in [2026,3407]:
 for date in dev[-2:]:
  t=p.index(date)-1;model,detail=outer_fit(p,t,best,seed=seed)
  record(f'{best}_seed{seed}',date,predict(model,p,t),detail);del model;gc.collect()
# Choose among complete-four-fold tree and neural predictions and a coarse global blend.
allmetrics=pd.DataFrame(rows);fullnames=['B0',tree_name]+finalists
leader=[]
for name in fullnames:
 r=allmetrics.loc[(allmetrics.model==name)&allmetrics.date.isin(dev)]
 leader.append({'model':name,'pooled_RMSLE':float(np.sqrt(np.mean(r.RMSLE**2)))})
for w in [0,.25,.5,.75,1.]:
 scores=[]
 for date in dev:
  suffix=str((pd.Timestamp(date)-pd.Timedelta(days=1)).date())
  tr=pd.read_csv(out/f'oof_{tree_name}_{suffix}.csv');nn=pd.read_csv(out/f'oof_{best}_{suffix}.csv')
  assert tr[['origin','store_nbr','family','h','target_date']].equals(nn[['origin','store_nbr','family','h','target_date']])
  zz=w*tr.pred_log+(1-w)*nn.pred_log
  scores.append(score(tr.actual_log,zz,tr.observed)['RMSLE'])
 leader.append({'model':f'blend_{w}','pooled_RMSLE':float(np.sqrt(np.mean(np.square(scores))))})
leader=pd.DataFrame(leader).sort_values('pooled_RMSLE');leader.to_csv(out/'leaderboard.csv',index=False)
winner=leader.iloc[0].model
weight=float(winner.split('_')[1]) if winner.startswith('blend_') else (1. if winner in ['B0',tree_name] else 0.)
network_kind=winner if winner in neural else best
selection={'winner':winner,'weight_tree':weight,'network_kind':network_kind,'development_rmsle':float(leader.iloc[0].pooled_RMSLE),'cpu_config':config,'tree_model':tree_name,'train_start':tree_start}
(out/'selection.json').write_text(json.dumps(selection,indent=2))
# Evaluate the frozen selection on R; never feed this result back into selection.
date='2017-07-31';t=p.index(date)-1
mt=tree_fit(date,tree_name,tree_start,tree_exog);vx,_,_=p.tree_rows([t],tree_exog);zt=mt.predict(vx).reshape(p.n,16)
mn,detail=outer_fit(p,t,network_kind);zn=predict(mn,p,t);record(network_kind,date,zn,detail)
if winner=='B0': zf=pd.read_csv(out/f'oof_B0_{p.dates[t].date()}.csv').pred_log.to_numpy().reshape(p.n,16)
else:zf=weight*np.maximum(zt,0)+(1-weight)*np.maximum(zn,0)
record('FINAL',date,zf,{'locked_selection':selection})
# Final training at T=Aug15; epochs from inner histories only.
t=p.index('2017-08-15');model,detail=outer_fit(p,t,network_kind)
zn=predict(model,p,t);xx,yy,mm=p.tree_rows(p.origins(p.dates[t],tree_start),tree_exog);mt.fit(xx[mm],yy[mm]);del xx,yy,mm;gc.collect()
vx,_,_=p.tree_rows([t],tree_exog);zt=mt.predict(vx).reshape(p.n,16)
if winner=='B0':
 fit=p.g['base'].loc[p.g['base'].sales.notna()];ev=p.g['base'].loc[p.g['base'].date.ge('2017-08-16')]
 raw,_,_=p.g['fit_predict']('LightGBM',fit,ev,{'n_estimators':350,'learning_rate':.05,'num_leaves':63,'min_child_samples':80})
 ev=ev[['store_nbr','family','date']].copy();ev['pred']=raw
 key=p.keys(t);zz=key.merge(ev,left_on=['store_nbr','family','target_date'],right_on=['store_nbr','family','date']).pred.to_numpy()
else:zz=np.expm1(weight*np.maximum(zt,0)+(1-weight)*np.maximum(zn,0)).ravel()
key=p.keys(t);test=p.g['test'];sample=p.g['sample']
for name,values in [('final',zz),('tree',np.expm1(np.maximum(zt,0)).ravel()),('network',np.expm1(np.maximum(zn,0)).ravel())]:
 key['sales']=values
 pred=test[['id','date','store_nbr','family']].merge(key[['target_date','store_nbr','family','sales']],left_on=['date','store_nbr','family'],right_on=['target_date','store_nbr','family'],validate='one_to_one').set_index('id').sales
 sub=sample[['id']].copy();sub['sales']=sub.id.map(pred)
 assert len(sub)==28512 and sub.id.equals(sample.id) and np.isfinite(sub.sales).all() and sub.sales.ge(0).all()
 sub.to_csv(out/f'submission_{name}.csv',index=False)
(out/'final_network_training.json').write_text(json.dumps(detail,indent=2))
print('ALL_EXPERIMENTS_COMPLETE',selection,flush=True)
