"""生成可独立复现的中文课程实验 Notebook。"""
from pathlib import Path
import nbformat as nb
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def M(s):cells.append(nb.v4.new_markdown_cell(s.strip()))
def C(s):cells.append(nb.v4.new_code_cell(s.strip()))
M('''# Store Sales：基于机器学习的商店销量时间序列预测
**课程：机器学习期末大作业**  
**姓名：________　学号：________**  
**来源：** Kaggle Store Sales - Time Series Forecasting，Corporación Favorita。

## 项目背景与研究目标
提前估计商品需求能够为商店备货提供参考。本项目预测 54 家门店、33 个商品类别在未来 16 天的日销量。目标 `sales` 是销量，不是单价；部分商品按重量计量，因此销量可能为小数。

参照用户提供的 House Prices 样例，完成数据探索、预处理、多模型比较、参数优化、特征重要性、误差分析及提交文件生成。将房价任务的随机划分改为时间回测，并加入周期基线。实验数值均由下列代码实际运行生成；测试标签未公开，Kaggle 线上成绩待实际提交后补充。
''')
M('''## 1. 实验环境与工具
使用 Python、Pandas、NumPy、Matplotlib、Seaborn、Scikit-learn、LightGBM。CPU 运行，随机种子固定为 42。将七个原始 CSV 放在 `data/`，或通过 `STORE_SALES_DATA_DIR` 指定目录。图表用英文标注保证跨平台显示，分析使用中文。
''')
C('''import os, sys, json, time, gc, platform, warnings
from pathlib import Path
import importlib.metadata as metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=DeprecationWarning,message="The 'generic' unit for NumPy timedelta.*")
SEED=42
TRAIN_START=pd.Timestamp('2016-01-01')
DATA=Path(os.environ.get('STORE_SALES_DATA_DIR','data'))
ART=Path('artifacts'); ART.mkdir(exist_ok=True)
FIG=ART/'figures'; FIG.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi':110,'axes.grid':True,'grid.alpha':.2})
sns.set_palette('colorblind')
versions={x:metadata.version(x) for x in ['numpy','pandas','scipy','scikit-learn','lightgbm','matplotlib','seaborn','nbformat','nbconvert','nbclient','ipykernel']}
display(pd.Series({'Python':platform.python_version(),**versions},name='实际版本').to_frame())
(ART/'environment.json').write_text(json.dumps(versions,indent=2))
def savefig(name):
    plt.tight_layout();plt.savefig(FIG/(name+'.png'),bbox_inches='tight');plt.show();plt.close()
''')
M('''## 2. 数据读取与基本检查
主键为“日期 × 商店 × 商品类别”。`id` 仅用于对齐提交，不参与预测。油价存在缺失且交易笔数仅覆盖历史，本版将它们用于背景分析，不把预测期实际值当作已知输入。
''')
C('''files=['train.csv','test.csv','stores.csv','holidays_events.csv','oil.csv','transactions.csv','sample_submission.csv']
assert all((DATA/f).exists() for f in files),f'请将七个 CSV 放入 {DATA.resolve()}'
dt={'id':'int32','store_nbr':'int16','family':'category','onpromotion':'int32'}
train=pd.read_csv(DATA/'train.csv',dtype=dt,parse_dates=['date'])
test=pd.read_csv(DATA/'test.csv',dtype=dt,parse_dates=['date'])
stores=pd.read_csv(DATA/'stores.csv')
holidays=pd.read_csv(DATA/'holidays_events.csv',parse_dates=['date'])
oil=pd.read_csv(DATA/'oil.csv',parse_dates=['date'])
transactions=pd.read_csv(DATA/'transactions.csv',parse_dates=['date'])
sample=pd.read_csv(DATA/'sample_submission.csv')
tables=[train,test,stores,holidays,oil,transactions,sample]
summary=pd.DataFrame([{'文件':f,'行数':len(t),'列数':t.shape[1],'空值数':int(t.isna().sum().sum())} for f,t in zip(files,tables)])
display(summary);display(train.head());display(train.dtypes.to_frame('类型'))
assert not train.duplicated(['date','store_nbr','family']).any()
assert train.id.is_unique and test.id.is_unique and (train.sales>=0).all()
assert test.date.min()>train.date.max()
missing_dates=pd.date_range(train.date.min(),train.date.max()).difference(train.date.unique())
print('训练日期：',train.date.min().date(),'至',train.date.max().date())
print('测试日期：',test.date.min().date(),'至',test.date.max().date())
print('商店数：',train.store_nbr.nunique(),'品类数：',train.family.nunique(),'系列数：',train.groupby(['store_nbr','family'],observed=True).ngroups)
print('零销量比例：',f'{train.sales.eq(0).mean():.2%}','；小数销量行数：',int(train.sales.mod(1).ne(0).sum()))
print('缺少日历日期：',missing_dates.strftime('%Y-%m-%d').tolist())
summary.to_csv(ART/'data_summary.csv',index=False)
''')
M('''## 3. 探索性数据分析
### 3.1 销量分布和历史趋势
真实零销量保留；缺失日期不直接等同于零。全历史用于描述性探索，模型选择仅使用后面的两个开发窗口。
''')
C('''display(train.sales.describe(percentiles=[.5,.9,.99]).to_frame())
daily=train.groupby('date').sales.sum()
fig,ax=plt.subplots(1,2,figsize=(14,4))
ax[0].hist(np.log1p(train.sales),bins=60);ax[0].set(title='Distribution of log1p(sales)',xlabel='log1p(sales)',ylabel='Rows')
daily.plot(ax=ax[1],alpha=.4,label='Daily');daily.rolling(28).mean().plot(ax=ax[1],label='28-day mean')
ax[1].set(title='Total sales over time',ylabel='Sales');ax[1].legend();savefig('01_distribution_trend')
''')
M('''### 3.2 星期、月份、商店和品类差异
月份比较采用日均总销量，减少月份长度影响。跨年均值还混合了门店发展等因素，不应直接解释为纯季节效应。
''')
C('''fig,ax=plt.subplots(2,2,figsize=(14,9))
daily.groupby(daily.index.dayofweek).mean().plot.bar(ax=ax[0,0]);ax[0,0].set(title='Mean daily total by weekday',xlabel='Weekday (Mon=0)')
daily.groupby(daily.index.month).mean().plot.bar(ax=ax[0,1]);ax[0,1].set(title='Mean daily total by month',xlabel='Month')
train.groupby('family',observed=True).sales.sum().nlargest(12).sort_values().plot.barh(ax=ax[1,0]);ax[1,0].set(title='Top product families',xlabel='Total sales')
train.groupby('store_nbr').sales.sum().plot.bar(ax=ax[1,1]);ax[1,1].set(title='Total sales by store',xlabel='Store');ax[1,1].tick_params(axis='x',labelsize=6)
savefig('02_seasonality_stores_families')
''')
M('''### 3.3 促销与外部变量
下列比较没有控制品类、门店等混杂因素，促销或油价与销量的相关关系不能直接解释为因果作用。交易笔数和销量的单位也不同。
''')
C('''promo=train.assign(promotion=train.onpromotion.gt(0)).groupby('promotion').sales.agg(['mean','median','count'])
display(promo)
fig,ax=plt.subplots(1,3,figsize=(16,4))
promo['mean'].plot.bar(ax=ax[0]);ax[0].set(title='Mean sales by promotion',ylabel='Mean sales')
oil.set_index('date').dcoilwtico.plot(ax=ax[1]);ax[1].set(title='Observed oil price',ylabel='Oil price')
transactions.groupby('date').transactions.sum().plot(ax=ax[2]);ax[2].set(title='Historical transactions',ylabel='Transactions');savefig('03_promotions_external')
top_family=train.groupby('family',observed=True).sales.sum().idxmax()
top_weekday=int(daily.groupby(daily.index.dayofweek).mean().idxmax())
display(Markdown(f'**探索结果：** 零销量占 **{train.sales.eq(0).mean():.2%}**，销量右偏，因此后续使用 log1p 目标和 RMSLE。总销量最大的品类为 **{top_family}**；平均日总销量最高的星期编号为 **{top_weekday}（周一为 0）**。门店与品类规模差异明显，需要分组分析误差。'))
''')
M('''## 4. 数据预处理与特征工程
### 4.1 日历、门店、促销与节假日
全国事件适用于所有商店，Regional 按州匹配，Local 按城市匹配；匹配后按“日期 × 门店”汇总，避免同日多事件导致数据膨胀。被转移的原假日不作为实际假日，Transfer、Additional、Bridge 作为有效假日，Event 与 Work Day 单独标记。
''')
C('''matched=holidays.assign(_key=1).merge(stores[['store_nbr','city','state']].assign(_key=1),on='_key').drop(columns='_key')
mask=matched.locale.eq('National')|(matched.locale.eq('Regional')&matched.locale_name.eq(matched.state))|(matched.locale.eq('Local')&matched.locale_name.eq(matched.city))
matched=matched.loc[mask].copy()
matched['is_holiday']=(matched.type.isin(['Holiday','Transfer','Additional','Bridge'])&~matched.transferred).astype('int8')
matched['is_event']=matched.type.eq('Event').astype('int8')
matched['is_workday']=matched.type.eq('Work Day').astype('int8')
hol=matched.groupby(['date','store_nbr'])[['is_holiday','is_event','is_workday']].max().reset_index()
base=pd.concat([train,test.assign(sales=np.nan)],ignore_index=True)
base=base.loc[base.date>=TRAIN_START].copy()
base=base.merge(stores,on='store_nbr',how='left',validate='many_to_one').merge(hol,on=['date','store_nbr'],how='left',validate='many_to_one')
for c in ['is_holiday','is_event','is_workday']:base[c]=base[c].fillna(0).astype('int8')
base['weekday']=base.date.dt.dayofweek;base['month']=base.date.dt.month;base['day']=base.date.dt.day;base['year']=base.date.dt.year
base['time_index']=(base.date-pd.Timestamp('2013-01-01')).dt.days
base['is_weekend']=base.weekday.ge(5).astype('int8')
base['is_payday']=(base.day.eq(15)|base.date.dt.is_month_end).astype('int8')
base['promotion_flag']=base.onpromotion.gt(0).astype('int8')
base['weekday_sin']=np.sin(2*np.pi*base.weekday/7).astype('float32')
base['weekday_cos']=np.cos(2*np.pi*base.weekday/7).astype('float32')
assert len(base)==int(train.date.ge(TRAIN_START).sum())+len(test)
print('合并后形状：',base.shape)
''')
M('''### 4.2 安全历史销量特征
先按真实日历补齐各系列索引，再计算 28、35、56 天滞后及先滞后 28 天的 7/28 天滚动均值。对于起点 D、最后预测日 D+15，最晚特征来源为 D−13，早于窗口开始。因此整个 16 天窗口不读取窗口内的真实销量。

不直接使用 7/14 天真实销量滞后、预测日交易笔数、实际未来油价、全样本目标均值。缺失日期保留为 NaN，sklearn 预处理仅在每折训练子集拟合填补、标准化和编码；LightGBM 原生处理缺失和类别。测试促销数量是测试表已提供的信息，可使用。
''')
C('''panel=train.pivot(index='date',columns=['store_nbr','family'],values='sales').astype('float32')
panel=panel.reindex(pd.date_range(train.date.min(),test.date.max(),name='date'))
base=base.set_index(['date','store_nbr','family']).sort_index()
lag_features=[]
def attach(values,name):
    s=values.loc[TRAIN_START:].stack([0,1],future_stack=True).rename(name)
    base[name]=s.reindex(base.index).to_numpy(dtype='float32')
for lag in [28,35,56]:
    name=f'lag_{lag}';attach(panel.shift(lag),name);lag_features.append(name)
for win in [7,28]:
    name=f'rolling_mean_{win}_shift28';attach(panel.shift(28).rolling(win,min_periods=1).mean(),name);lag_features.append(name)
base=base.reset_index()
cat_cols=['store_nbr','family','city','state','type','cluster']
for c in cat_cols:base[c]=base[c].astype('category')
num_cols=['onpromotion','weekday','month','day','year','time_index','is_weekend','is_payday','promotion_flag','weekday_sin','weekday_cos','is_holiday','is_event','is_workday']+lag_features
features=cat_cols+num_cols
display(base[['date','store_nbr','family','sales']+lag_features].head())
display(base[features].isna().sum().to_frame('缺失值数'))
print('特征数：',len(features),'；训练标签窗口起点：',TRAIN_START.date())
''')
M('''## 5. 时间验证方案与评价指标
开发窗口为 2017-06-29—07-14 和 2017-07-15—07-30，每次完整预测 16 天。各折训练标签从 2016-01-01 到窗口首日前一天；更早历史可用于过去的滞后特征。最后保留窗口 2017-07-31—08-15 不参与选型与调参。

主指标 **RMSLE** 为预测与真实 log1p 值之间的均方根误差，与竞赛一致，越低越好。预测用 expm1 还原后截断至非负，辅助计算原始销量尺度的 MAE、RMSE 和 R²。R² 可能为负，汇总 R² 不能代替分组检查；零销量较多，不使用 MAPE 作为主指标。
''')
C('''DEV=[pd.Timestamp('2017-06-29'),pd.Timestamp('2017-07-15')]
HOLD=pd.Timestamp('2017-07-31')
def metrics(y,p):
    y=np.asarray(y,dtype=float);p=np.maximum(np.asarray(p,dtype=float),0)
    return {'RMSLE':float(np.sqrt(np.mean((np.log1p(p)-np.log1p(y))**2))),'MAE':float(mean_absolute_error(y,p)),'RMSE':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p))}
def fold(start):
    cutoff=start-pd.Timedelta(days=1);end=start+pd.Timedelta(days=15)
    fit=base.loc[base.date.le(cutoff)&base.sales.notna()]
    val=base.loc[base.date.between(start,end)&base.sales.notna()]
    assert len(val)==16*54*33
    assert val.date.max()-pd.Timedelta(days=28)<=cutoff
    assert fit.date.max()<val.date.min()
    return fit,val
def seasonal(rows,cutoff):
    source=cutoff-pd.to_timedelta((cutoff.dayofweek-rows.date.dt.dayofweek)%7,unit='D')
    assert source.max()<=cutoff
    keys=pd.MultiIndex.from_arrays([rows.store_nbr.astype(int),rows.family.astype(str)])
    cols=panel.columns.get_indexer(keys);assert (cols>=0).all()
    arr=panel.reindex(pd.DatetimeIndex(source)).to_numpy()
    pred=arr[np.arange(len(rows)),cols]
    fallback=panel.loc[:cutoff].tail(28).mean().reindex(keys).to_numpy()
    return np.nan_to_num(np.where(np.isfinite(pred),pred,fallback),nan=0.)
BASE_GB={'n_estimators':220,'learning_rate':.08,'num_leaves':31,'min_child_samples':50}
def make_model(name,params=None):
    if name=='LightGBM':return LGBMRegressor(**(params or BASE_GB),random_state=SEED,n_jobs=2,verbosity=-1,colsample_bytree=.9,reg_lambda=1.)
    pre=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),num_cols),('cat',OneHotEncoder(handle_unknown='ignore',dtype=np.float32),cat_cols)],sparse_threshold=1.)
    models={'Ridge':Ridge(alpha=10,solver='lsqr',tol=1e-3,max_iter=150),
      'DecisionTree':DecisionTreeRegressor(max_depth=10,min_samples_leaf=30,random_state=SEED),
      'RandomForest':RandomForestRegressor(n_estimators=24,max_depth=10,min_samples_leaf=20,max_features=.7,max_samples=.35,n_jobs=2,random_state=SEED)}
    return Pipeline([('preprocess',pre),('model',models[name])])
def fit_predict(name,fit,ev,params=None):
    if name=='SeasonalNaive':return seasonal(ev,fit.date.max()),None,0.
    started=time.perf_counter();model=make_model(name,params)
    model.fit(fit[features],np.log1p(fit.sales))
    pred=np.maximum(np.expm1(model.predict(ev[features])),0)
    assert np.isfinite(pred).all()
    return pred,model,time.perf_counter()-started
display(pd.DataFrame([{'用途':'开发','起日':str(s.date()),'止日':str((s+pd.Timedelta(days=15)).date()),'训练行数':len(fold(s)[0]),'预测行数':len(fold(s)[1])} for s in DEV]))
''')
M('''## 6. 周期基线与四种回归模型
周期基线重复预测起点之前最近一周的同星期销量。Ridge 为线性基准，决策树展示非线性划分，随机森林集成树，LightGBM 展示梯度提升。所有算法使用相同标签日期与可用原始特征；sklearn 使用独热编码，LightGBM 使用原生类别。

CPU 预算下随机森林使用 24 棵、深度 10、每树有放回抽取 35% 样本，结果代表受控基准，不代表该算法充分调优的上限。耗时包含训练和预测。为便于观察进度，每折、每个算法单独运行一个单元格。
''')
C('''names=['SeasonalNaive','Ridge','DecisionTree','RandomForest','LightGBM']
records=[];predictions={};truth={}
def evaluate_development(start,name,algorithm=None,params=None):
    fit,val=fold(start);key=str(start.date());truth[key]=val.sales.to_numpy()
    pred,model,seconds=fit_predict(algorithm or name,fit,val,params)
    predictions[(key,name)]=pred
    row={'fold':key,'model':name,**metrics(val.sales,pred),'seconds':seconds};records.append(row)
    display(pd.DataFrame([row]).round(5))
    del model;gc.collect()
''')
for date in ['2017-06-29','2017-07-15']:
    for name in ['SeasonalNaive','Ridge','DecisionTree','RandomForest','LightGBM']:
        C(f"evaluate_development(pd.Timestamp('{date}'),'{name}')")
C('''def pooled(names):
    rows=[]
    for name in names:
        yy=np.concatenate([truth[str(s.date())] for s in DEV])
        pp=np.concatenate([predictions[(str(s.date()),name)] for s in DEV])
        rows.append({'model':name,**metrics(yy,pp),'seconds':sum(r['seconds'] for r in records if r['model']==name)})
    return pd.DataFrame(rows).sort_values('RMSLE').reset_index(drop=True)
comparison=pooled(names)
display(comparison.round(5))
pd.DataFrame(records).to_csv(ART/'development_fold_metrics.csv',index=False)
comparison.to_csv(ART/'baseline_model_comparison.csv',index=False)
fig,ax=plt.subplots(1,2,figsize=(12,4))
comparison.set_index('model').RMSLE.plot.barh(ax=ax[0]);ax[0].set(title='Development RMSLE',xlabel='RMSLE (lower is better)')
comparison.set_index('model').seconds.plot.barh(ax=ax[1]);ax[1].set(title='Training + prediction time',xlabel='Seconds');savefig('04_model_comparison')
ridge=comparison.set_index('model').loc['Ridge']
display(Markdown(f'当前默认比较中最好的算法为 **{comparison.iloc[0].model}**，RMSLE 为 **{comparison.iloc[0].RMSLE:.5f}**。Ridge 的 RMSLE 为 **{ridge.RMSLE:.5f}**、R² 为 **{ridge.R2:.4f}**，在当前特征表示下误差较大，尤其还原至原尺度后存在大偏差。后续可对历史销量特征也作对数变换，或增加交互项后重新验证；本实验不能据此断言所有线性方法均不适用。'))
''')
M('''## 7. 梯度提升超参数优化
默认配置作为对照，额外比较三组树数量、学习率、叶子数量与叶节点最小样本数。每组使用两个开发窗口，按拼接预测后的整体 RMSLE 排序，不使用最后保留窗口做选择，也不根据它提前停止。有限搜索只演示规范的优化流程，不承诺穷尽最优解。
''')
C('''param_by_name={'LightGBM':BASE_GB,
 'LightGBM_tuned_1':{'n_estimators':300,'learning_rate':.08,'num_leaves':31,'min_child_samples':50},
 'LightGBM_tuned_2':{'n_estimators':400,'learning_rate':.05,'num_leaves':31,'min_child_samples':50},
 'LightGBM_tuned_3':{'n_estimators':350,'learning_rate':.05,'num_leaves':63,'min_child_samples':80}}
''')
for i in [1,2,3]:
    C(f"for start in DEV:\n    evaluate_development(start,'LightGBM_tuned_{i}','LightGBM',param_by_name['LightGBM_tuned_{i}'])")
C('''leaderboard=pooled(names+list(param_by_name)[1:])
leaderboard['parameters']=leaderboard.model.map(lambda x:json.dumps(param_by_name.get(x,{})))
display(leaderboard.round(5));leaderboard.to_csv(ART/'development_leaderboard.csv',index=False)
pd.DataFrame(records[10:]).to_csv(ART/'search_fold_metrics.csv',index=False)
selected_name=leaderboard.iloc[0].model
selected_algorithm='LightGBM' if selected_name in param_by_name else selected_name
selected_params=param_by_name.get(selected_name)
best_gb=leaderboard.loc[leaderboard.model.isin(param_by_name)].iloc[0]
selection={'selected_model':selected_name,'parameters':selected_params,'train_start':str(TRAIN_START.date()),'development_rmsle':float(leaderboard.iloc[0].RMSLE)}
(ART/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
display(Markdown(f'开发阶段锁定 **{selected_name}**，合并 RMSLE 为 **{leaderboard.iloc[0].RMSLE:.5f}**。接下来只评估这份锁定方案，不根据保留窗口重调参数。'))
''')
M('''## 8. 最后保留窗口与误差分析
在 2017-07-31—08-15 完整预测 16 天。周期基线用于对照，不回流调参。进一步观察日总量、分门店、分品类和分步长误差，避免只看总体成绩。
''')
C('''hold_fit,hold_val=fold(HOLD)
hold_pred,hold_model,hold_seconds=fit_predict(selected_algorithm,hold_fit,hold_val,selected_params)
hold_scores=metrics(hold_val.sales,hold_pred)
naive_scores=metrics(hold_val.sales,seasonal(hold_val,hold_fit.date.max()))
hold_table=pd.DataFrame([{'model':selected_name,**hold_scores},{'model':'SeasonalNaive',**naive_scores}])
display(hold_table.round(5));hold_table.to_csv(ART/'holdout_metrics.csv',index=False)
errors=hold_val[['date','store_nbr','family','sales']].copy();errors['prediction']=hold_pred
errors['absolute_error']=np.abs(errors.sales-errors.prediction)
errors['squared_log_error']=(np.log1p(errors.sales)-np.log1p(errors.prediction))**2
errors['horizon']=(errors.date-HOLD).dt.days+1
errors.to_csv(ART/'holdout_predictions.csv',index=False)
family_error=errors.groupby('family',observed=True).agg(RMSLE=('squared_log_error',lambda x:np.sqrt(x.mean())),MAE=('absolute_error','mean'),mean_sales=('sales','mean')).sort_values('RMSLE',ascending=False)
display(family_error.head(12).round(4));family_error.to_csv(ART/'family_errors.csv')
store_error=errors.groupby('store_nbr',observed=True).squared_log_error.mean().pow(.5)
horizon_error=errors.groupby('horizon').squared_log_error.mean().pow(.5)
fig,ax=plt.subplots(2,2,figsize=(14,9))
errors.groupby('date')[['sales','prediction']].sum().plot(ax=ax[0,0]);ax[0,0].set(title='Holdout total sales',ylabel='Sales')
family_error.head(12).RMSLE.sort_values().plot.barh(ax=ax[0,1]);ax[0,1].set(title='Families with largest RMSLE')
store_error.plot.bar(ax=ax[1,0]);ax[1,0].set(title='RMSLE by store',xlabel='Store');ax[1,0].tick_params(axis='x',labelsize=6)
horizon_error.plot(marker='o',ax=ax[1,1]);ax[1,1].set(title='RMSLE by forecast horizon',xlabel='Days ahead');savefig('05_holdout_errors')
examples=errors.groupby(['store_nbr','family'],observed=True).sales.sum().nlargest(3).index
fig,axes=plt.subplots(3,1,figsize=(12,10))
for ax,(store,fam) in zip(axes,examples):
    e=errors.loc[(errors.store_nbr==store)&(errors.family==fam)].set_index('date')
    e[['sales','prediction']].plot(ax=ax,marker='o');ax.set(title=f'Store {store} / {fam}',ylabel='Sales')
savefig('06_example_series')
bias=errors.prediction.sum()/errors.sales.sum()-1
display(Markdown(f'保留窗口 RMSLE 为 **{hold_scores["RMSLE"]:.5f}**，周期基线为 **{naive_scores["RMSLE"]:.5f}**。预测总量偏差为 **{bias:+.2%}**。误差最大的品类为 **{family_error.index[0]}**，误差最大的门店为 **{store_error.idxmax()}**，逐步长误差最高出现在第 **{horizon_error.idxmax()}** 天。总量曲线能够跟随周内变化，但部分高销量日存在偏差；这不能代替对单条序列的检查，也不据此修改已锁定方案。'))
''')
M('''## 9. 特征重要性分析
取开发阶段最佳梯度提升配置，基于第二个开发窗口的训练部分拟合，展示累计分裂增益。此处只解释模型，不重新选择特征；增益不等同于因果作用，相关特征之间可能分摊重要性。
''')
C('''imp_fit,imp_val=fold(DEV[-1])
_,importance_model,_=fit_predict('LightGBM',imp_fit,imp_val,param_by_name[best_gb.model])
importance=pd.DataFrame({'feature':features,'gain':importance_model.booster_.feature_importance(importance_type='gain')}).sort_values('gain',ascending=False)
importance['gain_share']=importance.gain/importance.gain.sum()
display(importance.head(15).round(4));importance.to_csv(ART/'feature_importance.csv',index=False)
plt.figure(figsize=(10,6));plt.barh(importance.head(15).feature.iloc[::-1],importance.head(15).gain_share.iloc[::-1]);plt.xlabel('Share of gain');plt.title('LightGBM feature importance');savefig('07_feature_importance')
display(Markdown('本次增益重要性前三项为：**'+'、'.join(importance.head(3).feature)+'**。该排序只描述这一梯度提升模型。'))
del importance_model,hold_model;gc.collect()
''')
M('''## 10. 最终训练与测试预测
使用开发阶段锁定的算法与参数，重新拟合 2016-01-01—2017-08-15 的全部训练标签。更早历史用于 EDA 和安全滞后，最终训练窗口不临时改变。预测 2017-08-16—08-31，按样例 ID 顺序输出，保留小数，不把预测解释为商品单价。
''')
C('''final_fit=base.loc[base.sales.notna()&base.date.le('2017-08-15')]
future=base.loc[base.date.between('2017-08-16','2017-08-31')].copy()
assert future.date.max()-pd.Timedelta(days=28)<=final_fit.date.max()
future_pred,final_model,final_seconds=fit_predict(selected_algorithm,final_fit,future,selected_params)
pred_map=pd.Series(future_pred,index=future.id)
submission=sample[['id']].copy();submission['sales']=submission.id.map(pred_map)
assert list(submission.columns)==['id','sales'] and len(submission)==len(test)==28512
assert submission.id.equals(sample.id) and submission.id.is_unique
assert set(submission.id)==set(test.id)
assert np.isfinite(submission.sales).all() and submission.sales.ge(0).all()
submission.to_csv('submission.csv',index=False);display(submission.head(10))
print('最终训练标签数：',len(final_fit),'预测数：',len(submission),'耗时：',round(final_seconds,2),'秒')
print('提交格式、ID 顺序、唯一性、行数、有限值和非负值检查通过。')
fig,ax=plt.subplots(figsize=(12,4))
train.loc[train.date.ge('2017-07-01')].groupby('date').sales.sum().plot(ax=ax,label='Historical')
pd.Series(future_pred,index=future.date).groupby(level=0).sum().plot(ax=ax,label='Forecast')
ax.set(title='Recent historical sales and 16-day forecast',ylabel='Sales');ax.legend();savefig('08_final_forecast')
run_summary={'selected_model':selected_name,'holdout_metrics':hold_scores,'submission_rows':len(submission),'final_training_rows':len(final_fit),'final_seconds':final_seconds,'kaggle_public_score':None}
(ART/'run_summary.json').write_text(json.dumps(run_summary,ensure_ascii=False,indent=2))
''')
M('''## 11. 项目总结与不足
''')
C('''display(Markdown(f'本项目完成了商店销量预测的全流程。开发阶段选择 **{selected_name}**，最后保留窗口 RMSLE 为 **{hold_scores["RMSLE"]:.5f}**、MAE 为 **{hold_scores["MAE"]:.3f}**、RMSE 为 **{hold_scores["RMSE"]:.3f}**、R² 为 **{hold_scores["R2"]:.4f}**。已生成 **{len(submission):,}** 条正式预测。'))
''')
M('''**不足与改进方向：**
1. 两个开发窗口与一个保留窗口未覆盖所有季节，可增加跨月及节假日回测。
2. 最短 28 天滞后保证信息安全，但牺牲近期信息；可研究按预测起点构造的多步模型或严格递归预测。
3. 为控制 CPU 耗时，训练从 2016 年起、随机森林规模有限、梯度提升仅新增三组参数，不能代表各算法最优水平。
4. 本版没有使用未来实际油价与交易笔数；历史外生特征能否改进结果，应通过可用性一致的消融实验检验。
5. 门店、品类存在异质性和间歇需求，后续可研究分组与层级预测。
6. 没有代用户提交 Kaggle，Public Score 待实际上传后填写；本地回测分数不等同于线上分数。

## 12. 参考资料
- [Kaggle 竞赛与数据说明](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview)
- [Kaggle 评价规则](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview/evaluation)
- [Scikit-learn](https://scikit-learn.org/stable/)、[LightGBM](https://lightgbm.readthedocs.io/)、[Pandas](https://pandas.pydata.org/docs/)
- 用户提供的 House Prices 作业用于参考结构；本报告的数值由商店数据独立运行生成。

**复现说明：** 见 README.md 和 requirements.txt。Notebook 所在目录启动内核，数据置于 data/，重启并依次执行全部单元格。HTML 为同一份运行后 Notebook 导出，图表嵌入文件，可离线查看。不同机器的耗时和小数位可能不同。
''')
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
nb.write(n,ROOT/'Store_Sales.ipynb')
print('生成单元格数：',len(cells))
