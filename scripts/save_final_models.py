"""Rebuild locked final models, verify predictions, and persist inference weights."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from store_sales.data import Panel
from store_sales.training import train,predict
p=Panel(ROOT);out=p.out;sel=json.loads((out/'selection.json').read_text());detail=json.loads((out/'final_network_training.json').read_text())
p.train_start='2015-01-01';p.windows_per_epoch=detail.get('windows_per_epoch',8192 if sel['network_kind'].endswith('_long') else 4096)
t=p.index('2017-08-15');ck=out/'checkpoints';ck.mkdir(exist_ok=True)
m,curve=train(p,p.dates[t],sel['network_kind'],detail['selected_epochs'],42,width=detail['width'])
zn=predict(m,p,t)
torch.save({'state_dict':m.cpu().state_dict(),'sizes':p.sizes,'kind':sel['network_kind'],'width':detail['width']},ck/'network.pt')
exog=sel['tree_model']=='B2'
x,y,mask=p.tree_rows(p.origins(p.dates[t],sel['train_start']),exog)
model=LGBMRegressor(n_estimators=350,learning_rate=.05,num_leaves=63,min_child_samples=80,reg_lambda=1.,colsample_bytree=.9,n_jobs=2,random_state=42,verbosity=-1)
model.fit(x[mask],y[mask],categorical_feature=[0,1,2,3]);del x,y,mask
v,_,_=p.tree_rows([t],exog);zt=model.predict(v).reshape(p.n,16);model.booster_.save_model(str(ck/'tree.txt'))
features=['store','family','cluster','store_type']+[f'origin_lag_{i}' for i in [0,1,6,13,27,55]]+[f'rolling_{metric}_{w}' for w in [7,14,28,56] for metric in ['mean','std']]+['zero_fraction_28','store_total_log','family_total_log','horizon','future_promotion_log','weekday_sin','weekday_cos','month_sin','month_cos','day_scaled','payday','year_ago_log','year_ago_observed','seasonal_anchor','holiday','workday','origin_observed']+(['oil_at_origin','transactions_log_at_origin','transactions_observed'] if exog else [])
assert len(features)==v.shape[1]
pd.DataFrame({'feature':features,'gain':model.booster_.feature_importance('gain')}).sort_values('gain',ascending=False).to_csv(out/'feature_importance.csv',index=False)
errors={}
for name,z in [('tree',zt),('network',zn)]:
 key=p.keys(t);key['prediction_log']=np.maximum(z,0).ravel()
 test=p.g['test'];ordered=test[['id','date','store_nbr','family']].merge(key,left_on=['date','store_nbr','family'],right_on=['target_date','store_nbr','family'],validate='one_to_one').set_index('id').prediction_log
 saved=pd.read_csv(out/f'submission_{name}.csv');actual=ordered.reindex(saved.id).to_numpy()
 delta=float(np.max(np.abs(actual-np.log1p(saved.sales))))
 assert delta<1e-5,(name,delta)
 errors[name]=delta
(ck/'metadata.json').write_text(json.dumps({'locked_selection':sel,'network_training':detail,'max_abs_log_prediction_difference':errors,'weights_reproduce_saved_predictions':True},indent=2))
print('FINAL_WEIGHTS_VERIFIED',errors,flush=True)
