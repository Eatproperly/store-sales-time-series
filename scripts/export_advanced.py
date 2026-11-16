"""Independently audit saved experiment predictions and export the advanced report."""
from pathlib import Path
import os,sys,json,hashlib,zipfile,gzip,shutil
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
import numpy as np
import pandas as pd
import nbformat as nb
from nbclient import NotebookClient
from nbconvert import HTMLExporter
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--output-dir',default=str(ROOT/'outputs/advanced'));args=parser.parse_args()
OUT=Path(args.output_dir).resolve();metrics=pd.read_csv(OUT/'metrics.csv');sel=json.loads((OUT/'selection.json').read_text())
dev=['2016-08-16','2017-04-16','2017-06-15','2017-07-15']
def read_oof(f):return pd.read_csv(f if f.exists() else Path(str(f)+'.gz'))
# Recompute each saved score; verify labels by joining against the original data.
actual=pd.read_csv(ROOT/'data/train.csv',usecols=['date','store_nbr','family','sales'])
checks=[]
for r in metrics.itertuples(index=False):
 suffix=str((pd.Timestamp(r.date)-pd.Timedelta(days=1)).date());f=OUT/f'oof_{r.model}_{suffix}.csv';d=read_oof(f)
 assert not d.duplicated(['origin','store_nbr','family','h','target_date']).any()
 j=d.merge(actual,left_on=['target_date','store_nbr','family'],right_on=['date','store_nbr','family'],validate='one_to_one')
 assert len(j)==28512
 assert np.allclose(j.actual_log,np.log1p(j.sales),atol=1e-6)
 v=float(np.sqrt(np.mean((j.actual_log-j.pred_log)**2)))
 assert abs(v-r.RMSLE)<1e-6,(r.model,v,r.RMSLE)
 checks.append({'model':r.model,'date':r.date,'rmsle_recomputed':v,'labels_match':True})
base=pd.read_csv(ROOT/'data/sample_submission.csv')
for kind in ['final','tree','network']:
 d=pd.read_csv(OUT/f'submission_{kind}.csv')
 assert d.shape==(28512,2) and d.columns.tolist()==['id','sales'] and d.id.equals(base.id)
 assert d.id.is_unique and np.isfinite(d.sales).all() and d.sales.ge(0).all()
(OUT/'independent_verification.json').write_text(json.dumps({'verified_experiments':checks,'submission_rows':28512,'submission_id_order_matches':True,'labels_matched_original':True,'kaggle_public_score':None},indent=2))
manifest={'source_sha256':{},'data_sha256':{},'execution_commit':os.popen('git rev-parse HEAD').read().strip()}
for p in [*Path('src').rglob('*.py'),Path('scripts/run_advanced.py'),Path('configs/advanced.yaml')]:manifest['source_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
for p in (ROOT/'data').glob('*.csv'):manifest['data_sha256'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
C=[]
def md(s):C.append(nb.v4.new_markdown_cell(s))
def code(s):C.append(nb.v4.new_code_cell(s))
md('''# Store Sales 进阶版：基于预测起点的共享神经网络与模型融合
姓名：________　学号：________

本报告在基础课程作业上增加起点建样本、最新历史统计、未来促销、共享嵌入、MLP/Mixer、季节残差、消融、多种子和融合实验。

**执行范围说明：** 当前环境为 CPU。本轮测得网络训练速度后，采用宽度32、每轮随机4096条序列窗口、最多8轮的受限实验；内部比较2/8轮再重训。保留112天历史、16天输出、六组结构、四个外层窗口和多个种子。它是实际完成的 CPU 进阶实验，不代表计划中的更大GPU训练已完成。原固定窗口已经观察过，属于回归检查，不称新盲测。Kaggle尚无真实提交回执。

耗时、参数量与分数均由保存的实际实验读取。默认报告加载已运行产物；要重新计算，执行下一单元格中的开关。''')
code('''from pathlib import Path
import sys,json,subprocess,inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
ROOT=Path.cwd();OUT=ROOT/'outputs/advanced'
sys.path.insert(0,str(ROOT/'src'))
RUN_EXPERIMENTS=False
if RUN_EXPERIMENTS:subprocess.run([sys.executable,'scripts/run_advanced.py'],check=True)
metrics=pd.read_csv(OUT/'metrics.csv');selection=json.loads((OUT/'selection.json').read_text())
config=json.loads((OUT/'execution_config.json').read_text())
dev=['2016-08-16','2017-04-16','2017-06-15','2017-07-15']
leader=pd.read_csv(OUT/'leaderboard.csv')
plt.rcParams.update({'figure.dpi':110,'axes.grid':True,'grid.alpha':.2})
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
def fig(name):
 plt.tight_layout();plt.savefig(FIG/(name+'.png'),bbox_inches='tight');plt.show();plt.close()
def oof(name,date):
 suffix=str((pd.Timestamp(date)-pd.Timedelta(days=1)).date());p=OUT/f'oof_{name}_{suffix}.csv'
 return pd.read_csv(p if p.exists() else Path(str(p)+'.gz'))
display(pd.Series(config).to_frame('实际运行配置'))
''')
md('''## 1. 研究目标、数据与验证协议
54家商店、33个品类，共1782条日销量序列。每个起点T输入最近112天历史，使用未来16天已知促销与日历，预测T+1…T+16。历史销量、交易笔数及油价只取截至T的记录，地震等不可预知事件不作为已知未来输入。

四个开发窗口：2016-08-16—08-31、2017-04-16—05-01、2017-06-15—06-30、2017-07-15—07-30。最后用2017-07-31—08-15回归检查。内层窗口用于选择轮数，外层标签不参与训练或早停。主指标为逐记录RMSLE，辅助报告MAE、RMSE、R²。''')
code('''tests=json.loads((OUT/'data_tests.json').read_text());display(pd.Series(tests).to_frame('检查'))
verified=json.loads((OUT/'independent_verification.json').read_text())
display(pd.DataFrame(verified['verified_experiments']).head(10))
old=metrics.loc[(metrics.model=='B0')&metrics.date.isin(dev),'RMSLE']
best=leader.iloc[0]
gain=1-best.pooled_RMSLE/np.sqrt(np.mean(old**2))
r=metrics.loc[(metrics.model=='FINAL')&(metrics.date=='2017-07-31')].iloc[0]
display(Markdown(f'**实际结果摘要：** 开发阶段选择 **{selection["winner"]}**，四窗合并RMSLE为 **{best.pooled_RMSLE:.5f}**，相对原模型改善 **{gain:.2%}**。锁定后固定回归窗口RMSLE为 **{r.RMSLE:.5f}**。这些均为本地结果，不是Kaggle线上分数。'))
''')
md('''## 2. 预测起点样本与增强树
树模型把每个起点展开为16个预测步长，使用截至T的最新1/7/14/28/56天信息、均值、波动、零销量比例、门店和品类总量、历史同星期锚点，以及未来已知促销、日历和364天前参照。训练标签末日必须不晚于该折截止日。

B0是冻结的基础模型；B1使用2015年以来的起点样本；B1_2016单独检验训练窗；B2增加截至起点的油价和交易信息。两组近期窗口用于选择树配置，只有选中的树扩展到完整四窗。''')
code('''table=metrics.loc[metrics.date.isin(dev)].pivot(index='model',columns='date',values='RMSLE')
table['window_count']=table.notna().sum(axis=1)
table['available_windows_pooled']=np.sqrt((table[dev]**2).mean(axis=1))
display(table.round(5))
full=table.loc[table.window_count==4].sort_values('available_windows_pooled')
full['available_windows_pooled'].plot.barh(figsize=(9,5));plt.xlabel('RMSLE');plt.title('Models evaluated on all four development windows');fig('01_full_window_comparison')
''')
md('''## 3. 共享多步网络设计
历史包含对数销量、促销、门店与品类总量、历史外生量及观测掩码。上下文归一化只看截至起点的数据，标签保持原log1p尺度。门店、品类和组合嵌入分别为8、8、16维，商店类型及cluster提供静态补充。

Mixer沿时间与特征维混合，带LayerNorm、GELU、Dropout和残差；时间投影将112步历史转换为16步状态，与未来促销/日历、静态及预测步长嵌入一起解码。模型预测季节锚点上的修正量，输出log销量，最后截断非负并expm1还原。MLP使用相同输入作为稠密编码对照。

训练采用log销量MSE、AdamW、梯度裁剪。以下直接展示本轮执行的网络源码。''')
code('''from store_sales.models.mixer import Block,ForecastNet
for cls in [Block,ForecastNet]:display(Markdown('```python\\n'+inspect.getsource(cls)+'\\n```'))
from matplotlib.patches import FancyBboxPatch
f,ax=plt.subplots(figsize=(12,5));ax.axis('off')
boxes=[(.02,.65,'112-day history + masks'),(.02,.3,'Static IDs / future promotion'),(.36,.65,'MLP / residual Mixer'),(.36,.3,'16-step shared decoder'),(.72,.48,'Seasonal anchor + delta'),(.72,.12,'16-day sales forecast')]
for x,y,label in boxes:
 ax.add_patch(FancyBboxPatch((x,y),.26,.16,boxstyle='round,pad=.015',facecolor='#e6eff9',edgecolor='#4274a2'));ax.text(x+.13,y+.08,label,ha='center',va='center',fontsize=10)
for a,b in [((.28,.73),(.36,.73)),((.28,.38),(.36,.38)),((.49,.65),(.49,.46)),((.62,.38),(.72,.56)),((.85,.48),(.85,.28))]:ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->'})
fig('02_architecture')
''')
md('''## 4. 训练曲线、复杂度与消融
六组结构为MLP、两层Mixer、三层Mixer、去组合嵌入、去未来促销、去季节残差。为保证单因素比较，后三项消融均与**两层Mixer**对比。所有配置采用相同CPU抽样及轮数候选预算；图中不是全数据遍历epoch，准确含义是每轮随机4096条起点—序列窗口。

基础组内层2/8轮、加长三层组2/16轮选择后，回填至外层起点重训；加长组每轮8192样本，其余4096，不将训练量收益归为架构收益；没有用外层真实值提前停止。''')
code('''kinds=['mlp','mixer2','mixer3','no_pair','no_promo','no_anchor']
recent=metrics.loc[metrics.date.isin(dev[-2:])&metrics.model.isin(kinds)]
abl=recent.groupby('model').RMSLE.apply(lambda x:np.sqrt(np.mean(x*x))).to_frame('recent_two_windows_RMSLE')
abl['difference_vs_mixer2']=abl.recent_two_windows_RMSLE-abl.loc['mixer2','recent_two_windows_RMSLE']
display(abl.round(5));abl.recent_two_windows_RMSLE.plot.barh(figsize=(9,4));plt.xlabel('RMSLE');plt.title('Controlled two-window architecture comparison');fig('03_ablation')
for extra in ['mixer3_long','mixer3_scale']:
 if (OUT/f'training_{extra}_{dev[-1]}.json').exists():kinds=kinds+[extra]
complexity=[]
f,ax=plt.subplots(figsize=(9,4))
for k in kinds:
 data=json.loads((OUT/f'training_{k}_{dev[-1]}.json').read_text())
 curve=pd.DataFrame(data['training_curve']);ax.plot(curve.epoch,curve.train_log_mse,marker='o',label=k)
 complexity.append({'model':k,'parameters':data['parameters'],'selected_epochs':data['selected_epochs'],'train_seconds':curve.seconds.iloc[-1],'windows_per_epoch':data.get('windows_per_epoch',4096)})
ax.set(xlabel='CPU sampled epoch',ylabel='Training log MSE',title='Refit training curves');ax.legend();fig('04_learning_curves')
display(pd.DataFrame(complexity))
''')
md('''## 5. 多种子检验与模型融合
最终神经候选在两个近期窗口用42、2026、3407三个种子复验。仅报告真实波动，不将微小差异视为显著收益。融合在log1p空间进行，使用同一组四窗外层预测，在0、0.25、0.5、0.75、1这五个全局权重中选择，包含单模型端点。''')
code('''nk=selection['network_kind']
seed_rows=[]
for seed,name in [(42,nk),(2026,nk+'_seed2026'),(3407,nk+'_seed3407')]:
 rr=metrics.loc[(metrics.model==name)&metrics.date.isin(dev[-2:])]
 seed_rows.append({'seed':seed,'two_window_RMSLE':np.sqrt(np.mean(rr.RMSLE**2))})
seed_table=pd.DataFrame(seed_rows);display(seed_table)
print('三种子均值与标准差：',seed_table.two_window_RMSLE.mean(),seed_table.two_window_RMSLE.std())
display(leader.round(5))
blend=leader.loc[leader.model.str.startswith('blend_')].copy();blend['tree_weight']=blend.model.str.replace('blend_','',regex=False).astype(float)
blend.sort_values('tree_weight').plot(x='tree_weight',y='pooled_RMSLE',marker='o',legend=False,figsize=(8,4));plt.title('OOF log-space blend');plt.ylabel('RMSLE');fig('05_fusion')
''')
md('''## 6. 锁定后的固定回归窗口
此窗口在基础版已经观察过，只用于新版锁定后的检查。报告逐品类、逐步长和总量曲线，不利用该窗口回流重新选参数。''')
code('''reg=metrics.loc[metrics.date=='2017-07-31'];display(reg.round(5))
d=oof('FINAL','2017-07-31');d['actual']=np.expm1(d.actual_log);d['prediction']=np.expm1(d.pred_log);d['sqlog']=(d.actual_log-d.pred_log)**2
f,ax=plt.subplots(1,2,figsize=(14,4))
d.groupby('target_date')[['actual','prediction']].sum().plot(ax=ax[0]);ax[0].set(title='Regression window total sales',xlabel='Date')
d.groupby('h').sqlog.mean().pow(.5).plot(ax=ax[1],marker='o');ax[1].set(title='RMSLE by horizon',xlabel='Days ahead');fig('06_regression')
family=d.groupby('family').sqlog.mean().pow(.5).sort_values(ascending=False);display(family.head(12).to_frame('RMSLE'))
family.head(12).sort_values().plot.barh(figsize=(10,5));plt.xlabel('RMSLE');plt.title('Largest family errors');fig('07_family_errors')
''')
md('''## 7. 正式测试预测与核验
最终训练截至2017-08-15，只使用已锁定方案。分别导出树、神经和最终预测CSV，每份28512行，ID按官方样例顺序对齐。基础版文件原样保留，进阶版提交文件在outputs/advanced/中。

完整实验的OOF标签已与原始训练数据逐行连接核对，全部指标独立复算。无Kaggle登录提交回执，因此不填写Public Score。''')
code('''sub=pd.read_csv(OUT/'submission_final.csv');display(sub.head(10))
assert sub.shape==(28512,2) and sub.id.is_unique and np.isfinite(sub.sales).all() and sub.sales.ge(0).all()
print('预测行数：',len(sub),'；线上分数：待真实提交')
sub.sales.describe().to_frame().T.pipe(display)
plt.figure(figsize=(9,4));plt.hist(np.log1p(sub.sales),bins=50);plt.xlabel('log1p(predicted sales)');plt.title('Final prediction distribution');fig('08_submission')
display(Markdown(f'**结论：** 本轮四窗开发RMSLE相对B0改善 **{gain:.2%}**，是否达到5%目标：**{gain>=.05}**。固定回归窗口RMSLE为 **{r.RMSLE:.5f}**。最终选择 **{selection["winner"]}**。模型复杂度增加并不必然提升精度，最终提交依据同一协议下的实际结果。'))
''')
md('''## 8. 局限与后续GPU实验
本轮是有明确CPU预算的受限研究（另增加同预算mixer3_scale，将上下文均值与尺度提供给解码器，与mixer3_long比较）：宽度32而非计划64，训练使用每轮4096随机窗口而非全窗口遍历，轮数候选2/8而非50轮耐心早停。所有结果只适用于这些实际设置，不能据此断言某网络家族的充分训练上限。

MLP/Mixer之间按同一数据、信息和训练预算对比；消融参考两层Mixer。部分模型只在两个窗口初筛，不能直接和四窗指标当作同一排名。四窗用于开发，固定窗口已观察，真实泛化仍需线上评分及更多月份回测。

GPU训练可使用同一模块扩大宽度、样本量与epoch；需要重新生成OOF和锁定选择，而非沿用CPU权重结论。未覆盖的计划项目包括完整GPU预算实验、不同种子预测的集成提交、真实Kaggle回执和按patience=6逐epoch的早停；不能标记为已完成。

参考：[TSMixer](https://research.google/pubs/tsmixer-an-all-mlp-architecture-for-time-series-forecasting/)、[TiDE](https://research.google/pubs/long-horizon-forecasting-with-tide-time-series-dense-encoder/)、[Kaggle数据与评价](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview)。本网络为任务适配实现，不宣称新的基础架构。
''')
actual_config=json.loads((OUT/'execution_config.json').read_text())
for cell in C:
 if cell.cell_type=='code':cell.source=cell.source.replace("ROOT/'outputs/advanced'","ROOT/"+repr(str(OUT.relative_to(ROOT))))
 if cell.cell_type=='markdown':
  cell.source=cell.source.replace('每轮随机4096条序列窗口、最多8轮的受限实验','基础组每轮随机4096条窗口、最多8轮，追加三层Mixer每轮8192条、最多16轮的受限实验')
  cell.source=cell.source.replace('内部比较2/8轮再重训','基础组内部比较2/8轮、加长组比较2/16轮后再重训')
  cell.source=cell.source.replace('统一采用相同CPU抽样及轮数候选预算','基础结构组采用相同CPU抽样及轮数候选预算')
  cell.source=cell.source.replace('完整GPU预算实验、不同种子预测的集成提交、真实Kaggle回执和按patience=6逐epoch的早停','完整GPU预算实测、不同种子预测的集成提交和真实Kaggle回执；GPU配置已实现patience=6，但尚无GPU运行证据')
  if actual_config['device']=='cuda':
   if cell.source.startswith('# Store Sales 进阶版'):cell.source='# Store Sales GPU进阶实验报告\n姓名：________　学号：________\n本报告加载实际GPU实验结果；设备、训练预算和指标见运行配置。未来16天输出，历史112天输入。Kaggle线上分数待真实提交。'
   if cell.source.startswith('## 8.'):cell.source='## 8. 局限与复现\n本报告按实际运行配置记录GPU训练结果。四个窗口用于开发，固定回归窗口已观察过，不能称为全新盲测。线上成绩必须由真实提交回执证明。不同环境可能有小数位差异。来源与原计划见仓库docs。'
n=nb.v4.new_notebook(cells=C,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
nb.write(n,ROOT/'Store_Sales_Advanced.ipynb')
client=NotebookClient(n,timeout=600,resources={'metadata':{'path':str(ROOT)}})
try:client.execute()
finally:nb.write(n,ROOT/'Store_Sales_Advanced.ipynb')
assert not [o for c in n.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
exporter=HTMLExporter();exporter.embed_images=True;html,_=exporter.from_notebook_node(n)
(ROOT/'Store_Sales_Advanced.html').write_text(html)
assert html.count('data:image/png;base64,')>=8
# Keep OOF manageable in Git; readers accept either CSV or gzip.
for f in OUT.glob('oof_*.csv'):
 with f.open('rb') as a,gzip.open(str(f)+'.gz','wb') as b:shutil.copyfileobj(a,b)
 f.unlink()
readme=f'''# Store Sales 进阶版实际执行说明

完整执行了CPU受限版本：起点建样本、增强树、6组网络/消融、4窗确认、3种子复验和对数融合。

最终开发选择：{sel['winner']}；四窗开发RMSLE：{sel['development_rmsle']:.6f}。
固定回归窗口：{metrics.loc[metrics.model.eq('FINAL'),'RMSLE'].iloc[0]:.6f}。线上分数待实际提交。

报告：Store_Sales_Advanced.ipynb、Store_Sales_Advanced.html。
最终提交：outputs/advanced/submission_final.csv，另提供tree/network对照。

## 运行
将原始七个CSV放入data/；安装requirements-advanced.txt；在仓库目录运行：

```bash
python -m unittest discover -s tests -v
python scripts/run_advanced.py
python scripts/export_advanced.py
```

报告默认读取已保存的实测结果，RUN_EXPERIMENTS=True会重新执行实验。大规模重新训练应选择新输出目录，避免混淆旧配置。OOF已gzip压缩，指标保留CSV与JSON。

本轮CPU调整：width32、每轮4096随机窗口、基础epoch候选2/8，追加三层网络为8192样本/轮、2/16轮。没有宣称完整GPU计划或Kaggle提交已经完成。基础版Notebook作为冻结预处理/估计器来源随包保留。权重未作为部署模型交付，复现需重新训练。
'''
(ROOT/'README_ADVANCED.md').write_text(readme)
archive=ROOT/'Store_Sales_Advanced_Assignment.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for name in ['Store_Sales_Advanced.ipynb','Store_Sales_Advanced.html','Store_Sales.ipynb','README_ADVANCED.md','requirements-advanced.txt','requirements-advanced-gpu.txt','requirements.txt','Store_Sales_Advanced_GPU.ipynb','LICENSE']:
  z.write(ROOT/name,name)
 for folder in ['src','configs','tests',str(OUT.relative_to(ROOT))]:
  for f in (ROOT/folder).rglob('*'):
   if f.is_file() and '__pycache__' not in f.parts and '/artifacts/' not in str(f):z.write(f,f.relative_to(ROOT))
 for f in ['scripts/run_advanced.py','scripts/export_advanced.py','data/README.md']:z.write(ROOT/f,f)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print('ADVANCED_EXPORT_VERIFIED',archive.stat().st_size,flush=True)
