import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def score(y,z,mask=None):
 y=np.asarray(y).ravel();z=np.maximum(np.asarray(z).ravel(),0)
 if mask is not None:y=y[np.asarray(mask).ravel()];z=z[np.asarray(mask).ravel()]
 p=np.expm1(z);truth=np.expm1(y)
 return {'RMSLE':float(np.sqrt(np.mean((y-z)**2))),'MAE':float(mean_absolute_error(truth,p)),'RMSE':float(np.sqrt(mean_squared_error(truth,p))),'R2':float(r2_score(truth,p))}

def write_prediction(panel,t,z,name):
 frame=panel.keys(t);frame['actual_log']=panel.z[t+1:t+17].T.ravel();frame['observed']=panel.mask[t+1:t+17].T.ravel();frame['pred_log']=np.maximum(np.asarray(z).ravel(),0)
 frame.to_csv(panel.out/f'oof_{name}_{panel.dates[t].date()}.csv',index=False)
 return score(frame.actual_log,frame.pred_log,frame.observed)
