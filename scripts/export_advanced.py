"""Independently audit saved experiment predictions and export the advanced report."""

from pathlib import Path
import os, sys, json, hashlib, zipfile, gzip, shutil

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
import numpy as np
import pandas as pd
import nbformat as nb
from nbclient import NotebookClient
from nbconvert import HTMLExporter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default=str(ROOT / "outputs/advanced"))
args = parser.parse_args()
OUT = Path(args.output_dir).resolve()
model_audit = json.loads((OUT / "saved_model_verification.json").read_text())
assert model_audit["weights_reproduce_saved_predictions"]
for name, digest in model_audit["model_file_sha256"].items():
    assert (
        hashlib.sha256((OUT / "checkpoints" / name).read_bytes()).hexdigest() == digest
    )
metrics = pd.read_csv(OUT / "metrics.csv")
sel = json.loads((OUT / "selection.json").read_text())
final_description = (
    "B0"
    if sel["winner"] == "B0"
    else sel["tree_model"]
    if sel["weight_tree"] == 1
    else sel["network_kind"]
    if sel["weight_tree"] == 0
    else sel["winner"]
)
dev = ["2016-08-16", "2017-04-16", "2017-06-15", "2017-07-15"]


def read_oof(f):
    return pd.read_csv(f if f.exists() else Path(str(f) + ".gz"))


# Recompute each saved score; verify labels by joining against the original data.
actual = pd.read_csv(
    ROOT / "data/train.csv",
    usecols=["date", "store_nbr", "family", "sales", "onpromotion"],
)
checks = []
for r in metrics.itertuples(index=False):
    suffix = str((pd.Timestamp(r.date) - pd.Timedelta(days=1)).date())
    f = OUT / f"oof_{r.model}_{suffix}.csv"
    d = read_oof(f)
    assert not d.duplicated(["origin", "store_nbr", "family", "h", "target_date"]).any()
    j = d.merge(
        actual,
        left_on=["target_date", "store_nbr", "family"],
        right_on=["date", "store_nbr", "family"],
        validate="one_to_one",
    )
    assert len(j) == 28512
    assert float(np.max(np.abs(j.actual_log - np.log1p(j.sales)))) < 2e-6
    v = float(np.sqrt(np.mean((np.log1p(j.sales) - j.pred_log) ** 2)))
    assert abs(v - r.RMSLE) < 1e-6, (r.model, v, r.RMSLE)
    checks.append(
        {"model": r.model, "date": r.date, "rmsle_recomputed": v, "labels_match": True}
    )
base = pd.read_csv(ROOT / "data/sample_submission.csv")
for kind in ["final", "tree", "network"]:
    d = pd.read_csv(OUT / f"submission_{kind}.csv")
    assert (
        d.shape == (28512, 2)
        and d.columns.tolist() == ["id", "sales"]
        and d.id.equals(base.id)
    )
    assert d.id.is_unique and np.isfinite(d.sales).all() and d.sales.ge(0).all()
(OUT / "independent_verification.json").write_text(
    json.dumps(
        {
            "verified_experiments": checks,
            "submission_rows": 28512,
            "submission_id_order_matches": True,
            "labels_matched_original": True,
            "kaggle_public_score": None,
        },
        indent=2,
    )
)
# Diagnostics use frozen OOF predictions; they do not change model selection.
group_rows = []
window_rows = []
for date in dev + ["2017-07-31"]:
    suffix = str((pd.Timestamp(date) - pd.Timedelta(days=1)).date())
    candidate = "FINAL" if date == "2017-07-31" else sel["tree_model"]
    for name in ["B0", candidate]:
        d = read_oof(OUT / f"oof_{name}_{suffix}.csv")
        d = d.merge(
            actual.loc[
                actual.date.between(
                    date, str((pd.Timestamp(date) + pd.Timedelta(days=15)).date())
                )
            ],
            left_on=["target_date", "store_nbr", "family"],
            right_on=["date", "store_nbr", "family"],
            validate="one_to_one",
        )
        d["squared_log_error"] = (np.log1p(d.sales) - d.pred_log) ** 2
        d["zero_sales"] = np.where(d.sales.eq(0), "zero", "positive")
        d["promotion"] = np.where(d.onpromotion.gt(0), "promotion", "no_promotion")
        window_rows.append(
            {
                "date": date,
                "model": name,
                "RMSLE": float(np.sqrt(d.squared_log_error.mean())),
            }
        )
        for group in ["store_nbr", "family", "zero_sales", "promotion", "h"]:
            for value, rows in d.groupby(group):
                group_rows.append(
                    {
                        "date": date,
                        "model": name,
                        "group": group,
                        "value": str(value),
                        "count": len(rows),
                        "RMSLE": float(np.sqrt(rows.squared_log_error.mean())),
                    }
                )
pd.DataFrame(group_rows).to_csv(OUT / "group_diagnostics.csv", index=False)
pd.DataFrame(window_rows).to_csv(OUT / "window_diagnostics.csv", index=False)
seed_ensemble_rows = []
key = ["origin", "store_nbr", "family", "h", "target_date"]
for date in dev[-2:]:
    suffix = str((pd.Timestamp(date) - pd.Timedelta(days=1)).date())
    frames = [
        read_oof(OUT / f"oof_{sel['network_kind']}{tag}_{suffix}.csv").set_index(key)
        for tag in ["", "_seed2026", "_seed3407"]
    ]
    reference = frames[0]
    assert all(reference.index.equals(f.index) for f in frames)
    prediction = np.mean([f.pred_log.to_numpy() for f in frames], axis=0)
    seed_ensemble_rows.append(
        {
            "date": date,
            "single_seed_RMSLE": float(
                np.sqrt(np.mean((reference.actual_log - reference.pred_log) ** 2))
            ),
            "three_seed_mean_RMSLE": float(
                np.sqrt(np.mean((reference.actual_log - prediction) ** 2))
            ),
        }
    )
pd.DataFrame(seed_ensemble_rows).to_csv(
    OUT / "seed_ensemble_diagnostics.csv", index=False
)
manifest = {
    "source_sha256": {},
    "data_sha256": {},
    "source_snapshot_commit": os.popen("git rev-parse HEAD").read().strip(),
    "baseline_notebook_code_sha256": hashlib.sha256(
        "\n".join(
            c.source
            for c in nb.read(ROOT / "Store_Sales.ipynb", as_version=4).cells
            if c.cell_type == "code"
        ).encode()
    ).hexdigest(),
}
for p in [
    *Path("src").rglob("*.py"),
    Path("scripts/run_advanced.py"),
    Path("configs/advanced.yaml"),
]:
    manifest["source_sha256"][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
for p in (ROOT / "data").glob("*.csv"):
    manifest["data_sha256"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
C = []


def md(s):
    C.append(nb.v4.new_markdown_cell(s))


def code(s):
    C.append(nb.v4.new_code_cell(s))


md("""# Store Sales 进阶版：基于预测起点的共享神经网络与模型融合
姓名：________　学号：________

本报告在基础课程作业上增加起点建样本、最新历史统计、未来促销、共享嵌入、MLP/Mixer、季节残差、消融、多种子和融合实验。

**执行范围说明：** 当前环境为 CPU。本轮测得网络训练速度后，采用宽度32、每轮随机4096条序列窗口、最多8轮的受限实验；内部比较2/8轮再重训。保留112天历史、16天输出、六组结构、四个外层窗口和多个种子。它是实际完成的 CPU 进阶实验，不代表计划中的更大GPU训练已完成。原固定窗口已经观察过，属于回归检查，不称新盲测。Kaggle尚无真实提交回执。

耗时、参数量与分数均由保存的实际实验读取。默认报告加载已运行产物；要重新计算，执行下一单元格中的开关。""")
code("""from pathlib import Path
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
display_name=('B0' if selection['winner']=='B0' else selection['tree_model'] if selection['weight_tree']==1 else selection['network_kind'] if selection['weight_tree']==0 else selection['winner'])
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
""")
md("""## 1. 研究目标、数据与验证协议
54家商店、33个品类，共1782条日销量序列。每个起点T输入最近112天历史，使用未来16天已知促销与日历，预测T+1…T+16。历史销量、交易笔数及油价只取截至T的记录，地震等不可预知事件不作为已知未来输入。

四个开发窗口：2016-08-16—08-31、2017-04-16—05-01、2017-06-15—06-30、2017-07-15—07-30。最后用2017-07-31—08-15回归检查。内层窗口用于选择轮数，外层标签不参与训练或早停。主指标为逐记录RMSLE，辅助报告MAE、RMSE、R²。""")
code("""tests=json.loads((OUT/'data_tests.json').read_text());display(pd.Series(tests).to_frame('检查'))
verified=json.loads((OUT/'independent_verification.json').read_text())
display(pd.DataFrame(verified['verified_experiments']).head(10))
old=metrics.loc[(metrics.model=='B0')&metrics.date.isin(dev),'RMSLE']
best=leader.iloc[0]
gain=1-best.pooled_RMSLE/np.sqrt(np.mean(old**2))
r=metrics.loc[(metrics.model=='FINAL')&(metrics.date=='2017-07-31')].iloc[0]
display(Markdown(f'**实际结果摘要：** 开发阶段选择 **{display_name}**，四窗合并RMSLE为 **{best.pooled_RMSLE:.5f}**，相对原模型改善 **{gain:.2%}**。锁定后固定回归窗口RMSLE为 **{r.RMSLE:.5f}**。这些均为本地结果，不是Kaggle线上分数。'))
""")
md("""## 2. 预测起点样本与增强树
树模型把每个起点展开为16个预测步长，使用截至T的最新1/7/14/28/56天信息、均值、波动、零销量比例、门店和品类总量、历史同星期锚点，以及未来已知促销、日历和364天前参照。训练标签末日必须不晚于该折截止日。

B0是冻结的基础模型；B1使用2015年以来的起点样本；B1_2016单独检验训练窗；B2增加截至起点的油价和交易信息；B3比较600轮、63直方图bins、全列采样的高效参数组。两组近期窗口用于初筛树配置，将候选B1和B3扩展到完整四窗，按四窗结果锁定最终模型。""")
code("""table=metrics.loc[metrics.date.isin(dev)].pivot(index='model',columns='date',values='RMSLE')
table['window_count']=table.notna().sum(axis=1)
table['available_windows_pooled']=np.sqrt((table[dev]**2).mean(axis=1))
display(table.round(5))
full=table.loc[table.window_count==4].sort_values('available_windows_pooled')
full['available_windows_pooled'].plot.barh(figsize=(9,5));plt.xlabel('RMSLE');plt.title('Models evaluated on all four development windows');fig('01_full_window_comparison')
""")
md("""## 3. 共享多步网络设计
历史包含对数销量、促销、门店与品类总量、历史外生量及观测掩码。上下文归一化只看截至起点的数据，标签保持原log1p尺度。门店、品类和组合嵌入分别为8、8、16维，商店类型及cluster提供静态补充。

Mixer沿时间与特征维混合，带LayerNorm、GELU、Dropout和残差；时间投影将112步历史转换为16步状态，与未来促销/日历、静态及预测步长嵌入一起解码。模型预测季节锚点上的修正量，输出log销量，最后截断非负并expm1还原。基础MLP使用相同输入作为稠密编码对照。额外mixer3_scale分支保留归一化前的均值、尺度与起点原始水平，与mixer3_long按相同训练预算比较；该收益不全归因于混合层。

训练采用log销量MSE、AdamW、梯度裁剪。以下直接展示本轮执行的网络源码。组合嵌入消融采用置零关闭，注册参数量仍包含该表；比较的是信息使用而非模型压缩。CPU耗时受并行阶段资源共享影响，不作为专用硬件速度基准。""")
code("""from store_sales.models.mixer import Block,ForecastNet
for cls in [Block,ForecastNet]:display(Markdown('```python'+chr(10)+inspect.getsource(cls)+chr(10)+'```'))
from matplotlib.patches import FancyBboxPatch
f,ax=plt.subplots(figsize=(12,5));ax.axis('off')
boxes=[(.02,.65,'112-day history + masks'),(.02,.3,'Static IDs / future promotion'),(.36,.65,'MLP / residual Mixer'),(.36,.3,'16-step shared decoder'),(.72,.48,'Seasonal anchor + delta'),(.72,.12,'16-day sales forecast')]
for x,y,label in boxes:
 ax.add_patch(FancyBboxPatch((x,y),.26,.16,boxstyle='round,pad=.015',facecolor='#e6eff9',edgecolor='#4274a2'));ax.text(x+.13,y+.08,label,ha='center',va='center',fontsize=10)
for a,b in [((.28,.73),(.36,.73)),((.28,.38),(.36,.38)),((.49,.65),(.49,.46)),((.62,.38),(.72,.56)),((.85,.48),(.85,.28))]:ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->'})
fig('02_architecture')
""")
md("""## 4. 训练曲线、复杂度与消融
六组结构为MLP、两层Mixer、三层Mixer、去组合嵌入、去未来促销、去季节残差。为保证单因素比较，后三项消融均与**两层Mixer**对比。基础六组采用相同CPU抽样及轮数候选预算；额外long与scale组单独报告加长预算；图中不是全数据遍历epoch，准确含义是每轮随机4096条起点—序列窗口。

基础组内层2/8轮、加长三层组2/16轮选择后，回填至外层起点重训；加长组每轮8192样本，其余4096，不将训练量收益归为架构收益；没有用外层真实值提前停止。""")
code("""kinds=['mlp','mixer2','mixer3','no_pair','no_promo','no_anchor']
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
""")
md("""## 5. 多种子检验与模型融合
最终神经候选在两个近期窗口用42、2026、3407三个种子复验，并对三份对数预测均值进行同窗检验；该诊断不改变已锁定的最终提交方案。权重为1的融合端点等同纯树，不能声称神经网络贡献了融合增益。仅报告真实波动，不将微小差异视为显著收益。融合在log1p空间进行，使用同一组四窗外层预测，在0、0.25、0.5、0.75、1这五个全局权重中选择，包含单模型端点。""")
code("""nk=selection['network_kind']
seed_rows=[]
for seed,name in [(42,nk),(2026,nk+'_seed2026'),(3407,nk+'_seed3407')]:
 rr=metrics.loc[(metrics.model==name)&metrics.date.isin(dev[-2:])]
 seed_rows.append({'seed':seed,'two_window_RMSLE':np.sqrt(np.mean(rr.RMSLE**2))})
seed_table=pd.DataFrame(seed_rows);display(seed_table)
print('三种子均值与标准差：',seed_table.two_window_RMSLE.mean(),seed_table.two_window_RMSLE.std())
ensemble=pd.read_csv(OUT/'seed_ensemble_diagnostics.csv');display(ensemble)
print('两窗合并：单种子',np.sqrt(np.mean(ensemble.single_seed_RMSLE**2)),'；三种子预测均值',np.sqrt(np.mean(ensemble.three_seed_mean_RMSLE**2)))
display(leader.round(5))
blend=leader.loc[leader.model.str.startswith('blend_')].copy();blend['tree_weight']=blend.model.str.replace('blend_','',regex=False).astype(float)
blend.sort_values('tree_weight').plot(x='tree_weight',y='pooled_RMSLE',marker='o',legend=False,figsize=(8,4));plt.title('OOF log-space blend');plt.ylabel('RMSLE');fig('05_fusion')
""")
md("""## 6. 锁定后的固定回归窗口
此窗口在基础版已经观察过，只用于新版锁定后的检查。报告逐品类、逐步长和总量曲线，不利用该窗口回流重新选参数。""")
code("""reg=metrics.loc[metrics.date=='2017-07-31'];display(reg.round(5))
d=oof('FINAL','2017-07-31');d['actual']=np.expm1(d.actual_log);d['prediction']=np.expm1(d.pred_log);d['sqlog']=(d.actual_log-d.pred_log)**2
f,ax=plt.subplots(1,2,figsize=(14,4))
d.groupby('target_date')[['actual','prediction']].sum().plot(ax=ax[0]);ax[0].set(title='Regression window total sales',xlabel='Date')
d.groupby('h').sqlog.mean().pow(.5).plot(ax=ax[1],marker='o');ax[1].set(title='RMSLE by horizon',xlabel='Days ahead');fig('06_regression')
family=d.groupby('family').sqlog.mean().pow(.5).sort_values(ascending=False);display(family.head(12).to_frame('RMSLE'))
family.head(12).sort_values().plot.barh(figsize=(10,5));plt.xlabel('RMSLE');plt.title('Largest family errors');fig('07_family_errors')
windows=pd.read_csv(OUT/'window_diagnostics.csv')
comparison=windows.loc[windows.date.isin(dev)].pivot(index='date',columns='model',values='RMSLE')
comparison['relative_improvement']=1-comparison[selection['tree_model']]/comparison.B0
display(comparison)
print('四窗改善数：',int(comparison.relative_improvement.gt(0).sum()),'；最差窗口相对变化（正数为改善）：',comparison.relative_improvement.min())
groups=pd.read_csv(OUT/'group_diagnostics.csv')
display(groups.loc[groups.date.eq('2017-07-31') & groups['group'].isin(['zero_sales','promotion'])])
display(groups.loc[groups.date.eq('2017-07-31') & groups.model.eq('FINAL') & groups['group'].eq('store_nbr')].sort_values('RMSLE',ascending=False).head(10))
""")
md("""## 7. 正式测试预测与核验
最终训练截至2017-08-15，只使用已锁定方案。分别导出树、神经和最终预测CSV，每份28512行，ID按官方样例顺序对齐。基础版文件原样保留，进阶版提交文件在outputs/advanced/中。

完整实验的OOF标签已与原始训练数据逐行连接核对，全部指标独立复算。无Kaggle登录提交回执，因此不填写Public Score。""")
code("""sub=pd.read_csv(OUT/'submission_final.csv');display(sub.head(10))
assert sub.shape==(28512,2) and sub.id.is_unique and np.isfinite(sub.sales).all() and sub.sales.ge(0).all()
print('预测行数：',len(sub),'；线上分数：待真实提交')
sub.sales.describe().to_frame().T.pipe(display)
plt.figure(figsize=(9,4));plt.hist(np.log1p(sub.sales),bins=50);plt.xlabel('log1p(predicted sales)');plt.title('Final prediction distribution');fig('08_submission')
display(Markdown(f'**结论：** 本轮四窗开发RMSLE相对B0改善 **{gain:.2%}**，是否达到5%目标：**{gain>=.05}**，是否达到固定窗≤0.40挑战线：**{r.RMSLE<=.40}**。固定回归窗口RMSLE为 **{r.RMSLE:.5f}**。最终选择 **{display_name}**。模型复杂度增加并不必然提升精度，最终提交依据同一协议下的实际结果。'))
""")
md("""## 8. 局限与后续GPU实验
本轮是有明确CPU预算的受限研究（另增加同预算mixer3_scale，将上下文均值、尺度与最新原始水平提供给解码器，与mixer3_long比较）：宽度32而非计划64，基础组每轮4096窗口、2/8轮候选，追加组8192窗口、2/16轮，均非全样本epoch，未采用完整50轮GPU预算。所有结果只适用于这些实际设置，不能据此断言某网络家族的充分训练上限。

MLP/Mixer之间按同一数据、信息和训练预算对比；消融参考两层Mixer。部分模型只在两个窗口初筛，不能直接和四窗指标当作同一排名。四窗用于开发，固定窗口已观察，真实泛化仍需线上评分及更多月份回测。

GPU训练可使用同一模块扩大宽度、样本量与epoch；需要重新生成OOF和锁定选择，而非沿用CPU权重结论。未覆盖的计划项目包括完整GPU预算实验、不同种子预测的集成提交、真实Kaggle回执和按patience=6逐epoch的早停；不能标记为已完成。

参考：[TSMixer](https://research.google/pubs/tsmixer-an-all-mlp-architecture-for-time-series-forecasting/)、[TiDE](https://research.google/pubs/long-horizon-forecasting-with-tide-time-series-dense-encoder/)、[Kaggle数据与评价](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview)。本网络为任务适配实现，不宣称新的基础架构。
""")
audit_index = next(
    i
    for i, c in enumerate(C)
    if c.cell_type == "markdown" and c.source.startswith("## 8.")
)
C[audit_index:audit_index] = [
    nb.v4.new_markdown_cell(
        "## 8. 最终树模型解释与权重复核\n增益重要性基于最终训练模型，仅反映模型内部使用情况，不代表因果作用。模型已保存，下面验证重新加载权重后可复现三份提交预测。"
    ),
    nb.v4.new_code_cell(
        "importance=pd.read_csv(OUT/'feature_importance.csv');display(importance.head(12))\nq=importance.head(12).copy();q['gain_share']=q.gain/importance.gain.sum()\nq.set_index('feature').gain_share.sort_values().plot.barh(figsize=(10,5));plt.xlabel('Share of total gain');plt.title('Final tree model feature importance');fig('09_feature_importance')\naudit=json.loads((OUT/'saved_model_verification.json').read_text());assert audit['weights_reproduce_saved_predictions']\ndisplay(pd.Series(audit['max_abs_log_prediction_difference']).to_frame('max log prediction difference'))"
    ),
]
C[audit_index + 2].source = C[audit_index + 2].source.replace("## 8.", "## 9.", 1)
actual_config = json.loads((OUT / "execution_config.json").read_text())
for cell in C:
    if cell.cell_type == "code":
        cell.source = cell.source.replace(
            "ROOT/'outputs/advanced'", "ROOT/" + repr(str(OUT.relative_to(ROOT)))
        )
    if cell.cell_type == "markdown":
        cell.source = cell.source.replace(
            "每轮随机4096条序列窗口、最多8轮的受限实验",
            "基础组每轮随机4096条窗口、最多8轮，追加三层Mixer每轮8192条、最多16轮的受限实验",
        )
        cell.source = cell.source.replace(
            "内部比较2/8轮再重训", "基础组内部比较2/8轮、加长组比较2/16轮后再重训"
        )
        cell.source = cell.source.replace(
            "统一采用相同CPU抽样及轮数候选预算",
            "基础结构组采用相同CPU抽样及轮数候选预算",
        )
        cell.source = cell.source.replace(
            "完整GPU预算实验、不同种子预测的集成提交、真实Kaggle回执和按patience=6逐epoch的早停",
            "完整GPU预算实测、不同种子预测的集成提交和真实Kaggle回执；GPU配置已实现patience=6，但尚无GPU运行证据",
        )
        if actual_config["device"] == "cuda":
            if cell.source.startswith("# Store Sales 进阶版"):
                cell.source = "# Store Sales GPU进阶实验报告\n姓名：________　学号：________\n本报告加载实际GPU实验结果；设备、训练预算和指标见运行配置。未来16天输出，历史112天输入。Kaggle线上分数待真实提交。"
            if cell.source.startswith("## 9."):
                cell.source = "## 9. 局限与复现\n本报告按实际运行配置记录GPU训练结果。四个窗口用于开发，固定回归窗口已观察过，不能称为全新盲测。线上成绩必须由真实提交回执证明。不同环境可能有小数位差异。来源与原计划见仓库docs。"
n = nb.v4.new_notebook(
    cells=C,
    metadata={
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        }
    },
)
for index, cell in enumerate(n.cells):
    if cell.cell_type == "code":
        compile(cell.source, f"report_cell_{index}", "exec")
nb.write(n, ROOT / "Store_Sales_Advanced.ipynb")
client = NotebookClient(n, timeout=600, resources={"metadata": {"path": str(ROOT)}})
try:
    client.execute()
finally:
    nb.write(n, ROOT / "Store_Sales_Advanced.ipynb")
assert not [
    o
    for c in n.cells
    if c.cell_type == "code"
    for o in c.outputs
    if o.output_type == "error"
]
exporter = HTMLExporter()
exporter.embed_images = True
html, _ = exporter.from_notebook_node(n)
(ROOT / "Store_Sales_Advanced.html").write_text(html)
assert html.count("data:image/png;base64,") >= 9
# Keep OOF manageable in Git; readers accept either CSV or gzip.
for f in OUT.glob("oof_*.csv"):
    with f.open("rb") as a, gzip.open(str(f) + ".gz", "wb") as b:
        shutil.copyfileobj(a, b)
    f.unlink()
readme = f"""# Store Sales 进阶版实际执行说明

完整执行了CPU受限版本：起点建样本、增强树、6组网络/消融、4窗确认、3种子复验和对数融合。

最终开发选择：{final_description}（网格标记{sel["winner"]}）；四窗开发RMSLE：{sel["development_rmsle"]:.6f}。
固定回归窗口：{metrics.loc[metrics.model.eq("FINAL"), "RMSLE"].iloc[0]:.6f}。线上分数待实际提交。

报告：Store_Sales_Advanced.ipynb、Store_Sales_Advanced.html。
最终提交：outputs/advanced/submission_final.csv，另提供tree/network对照。

## 运行
将原始七个CSV放入data/；安装requirements-advanced.txt；在仓库目录运行：

```bash
python -m unittest discover -s tests -v
python scripts/run_advanced.py
python scripts/verify_saved_models.py
python scripts/export_advanced.py
```

提交课程作业时建议提供完整压缩包；Notebook复现依赖随包src、scripts与outputs，原始CSV另放data/。报告默认读取已保存的实测结果，RUN_EXPERIMENTS=True会重新执行实验。大规模重新训练应选择新输出目录，避免混淆旧配置。OOF已gzip压缩，指标保留CSV与JSON。

本轮CPU调整：width32、每轮4096随机窗口、基础epoch候选2/8，追加三层网络为8192样本/轮、2/16轮。没有宣称完整GPU计划或Kaggle提交已经完成。基础版Notebook作为冻结预处理/估计器来源随包保留。压缩包包含checkpoints权重，可直接加载复现三份预测；从原始数据重新训练仍需完整运行。
"""
readme = readme.replace("outputs/advanced", str(OUT.relative_to(ROOT)))
readme += "\n## 完整GPU实验入口\n保持Kaggle预装CUDA版PyTorch，使用requirements-advanced-gpu.txt。打开Store_Sales_Advanced_GPU.ipynb，启用GPU并挂载原始数据后运行。该配置width64、全窗口、最多50轮、内层patience6；当前CPU报告不包含这一GPU实测。\n"
if actual_config["device"] == "cuda":
    readme = readme.replace("CPU受限版本", "GPU配置版本").replace(
        "本轮CPU调整：width32、每轮4096随机窗口、基础epoch候选2/8，追加三层网络为8192样本/轮、2/16轮。没有宣称完整GPU计划或Kaggle提交已经完成。",
        f"实际GPU配置：width={actual_config['width']}，每轮窗口数={actual_config['windows_per_epoch']}（0为全量），max_epochs={actual_config['max_epochs']}，inner_mode={actual_config['inner_mode']}。线上分数待真实提交。",
    )
word_report = ROOT / "Store_Sales_Data_Analysis_Report.docx"
if word_report.exists():
    readme += "\n## Word数据分析报告\n[下载Word报告](Store_Sales_Data_Analysis_Report.docx) · [PDF预览](Store_Sales_Data_Analysis_Report.pdf)。按House Prices示例的摘要、八章正文和参考资料组织，含真实图表与结果。封面个人信息留空。\n\n修改后可安装requirements-report.txt并运行 `python scripts/build_word_report.py --pdf` 重新生成；PDF还需要LibreOffice和Noto CJK字体。该命令读取已保存实验，不重新训练模型。\n"
(ROOT / "README_ADVANCED.md").write_text(readme)
rootreadme = ROOT / "README.md"
base_text = rootreadme.read_text()
start = "<!-- ADVANCED_RESULTS_START -->"
end = "<!-- ADVANCED_RESULTS_END -->"
if start in base_text:
    base_text = (
        base_text[: base_text.index(start)]
        + base_text[base_text.index(end) + len(end) :]
    )
block = f"{start}\n进阶版：[{final_description} 实际实验报告](README_ADVANCED.md) · [下载进阶作业包](https://github.com/Eatproperly/store-sales-time-series/raw/refs/heads/main/Store_Sales_Advanced_Assignment.zip) · [GPU训练入口](Store_Sales_Advanced_GPU.ipynb)。本轮设备={actual_config['device']}；Kaggle线上成绩待真实提交。\n{end}\n"
if word_report.exists():
    block = block.replace(
        end,
        "[Word数据分析报告](Store_Sales_Data_Analysis_Report.docx) · [PDF预览](Store_Sales_Data_Analysis_Report.pdf)\n"
        + end,
    )
base_text = base_text.replace(
    "# Store Sales：商店销量时间序列预测\n",
    "# Store Sales：商店销量时间序列预测\n\n" + block,
    1,
)
rootreadme.write_text(base_text)
execution = ROOT / "docs/advanced-execution.md"
execution.write_text(
    f"# 进阶版执行记录\n\n四窗开发选择：{final_description}，RMSLE={sel['development_rmsle']:.6f}。\n\n已完成：起点建样本、历史缺失掩码修正与回归测试、增强树及参数对照、基础6组神经结构与2组扩展、四窗评估、最终神经候选3种子复验、粗网格对数融合、固定回归窗口、最终训练与3份CSV、模型权重加载复现、独立指标审计和Notebook/HTML。\n\n实际CPU调整：width32，基础每轮4096条窗口/2或8轮，扩展每轮8192条/2或16轮。CPU推理和图表耗时可能包括资源调度等待。\n\n未完成：完整GPU预算实测、真实Kaggle提交回执，以及多种子集成正式测试提交。GPU配置、CUDA入口与patience机制已实现；当前环境没有CUDA硬件。不能用CPU数据冒充GPU实验或线上分数。\n\n原树特征中的缺失占位统计已修正，并重跑相关模型；最终排名不使用未修正结果。固定回归窗此前已观察过，不称新盲测。\n\n完整源码、数据哈希、训练预算和模型指标见outputs/advanced/；权重包含于进阶ZIP的checkpoints目录。"
)
if actual_config["device"] == "cuda":
    execution.write_text(
        f"# 进阶GPU执行记录\n\n实际设备={actual_config['device']}，结构宽度={actual_config['width']}，采样上限={actual_config['windows_per_epoch']}（0代表全部窗口），最大轮数={actual_config['max_epochs']}，内层选择={actual_config['inner_mode']}。\n\nOOF、分数、权重和提交文件经过脚本核验；真实Kaggle线上提交仍待执行。固定回归窗口此前已观察，不能称新盲测。\n\n最终选择={final_description}。完整配置及训练曲线见本次outputs目录。"
    )
archive = ROOT / "Store_Sales_Advanced_Assignment.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for name in [
        "Store_Sales_Advanced.ipynb",
        "Store_Sales_Advanced.html",
        "Store_Sales.ipynb",
        "README_ADVANCED.md",
        "requirements-advanced.txt",
        "requirements-advanced-gpu.txt",
        "requirements.txt",
        "Store_Sales_Advanced_GPU.ipynb",
        "LICENSE",
    ]:
        z.write(ROOT / name, name)
    for name in [
        "Store_Sales_Data_Analysis_Report.docx",
        "Store_Sales_Data_Analysis_Report.pdf",
        "Store_Sales_Data_Analysis_Report.md",
        "requirements-report.txt",
        "scripts/build_word_report.py",
        "artifacts/data_summary.csv",
        "artifacts/figures/01_distribution_trend.png",
        "artifacts/figures/02_seasonality_stores_families.png",
    ]:
        if (ROOT / name).exists():
            z.write(ROOT / name, name)
    for folder in ["src", "configs", "tests", str(OUT.relative_to(ROOT))]:
        for f in (ROOT / folder).rglob("*"):
            if (
                f.is_file()
                and "__pycache__" not in f.parts
                and "/artifacts/" not in str(f)
                and "tree_repaired" not in f.parts
                and "scale_trial" not in f.parts
                and "scale_seeds" not in f.parts
                and "tree_capacity" not in f.parts
            ):
                z.write(f, f.relative_to(ROOT))
    for f in [
        "scripts/run_advanced.py",
        "scripts/export_advanced.py",
        "scripts/verify_saved_models.py",
        "docs/advanced-execution.md",
        "data/README.md",
    ]:
        z.write(ROOT / f, f)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
print("ADVANCED_EXPORT_VERIFIED", archive.stat().st_size, flush=True)
