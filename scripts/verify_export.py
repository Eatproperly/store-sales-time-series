"""独立检查运行产物、导出 HTML、构建可下载的成果包。"""
from pathlib import Path
from html.parser import HTMLParser
import json,zipfile,os
import numpy as np
import pandas as pd
import nbformat
from nbconvert import HTMLExporter

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
DATA=Path(os.environ.get('STORE_SALES_DATA_DIR','data'))
path=ROOT/'Store_Sales.ipynb'
n=nbformat.read(path,as_version=4)
# 保持重复导出幂等；这些摘要由已经核验的运行结果生成。
n.cells=[c for c in n.cells if not(c.cell_type=='markdown' and c.source.startswith('## 实验摘要'))]
nbformat.validate(n)
code=[c for c in n.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in code),'存在未执行的代码单元格'
assert not [o for c in code for o in c.outputs if o.output_type=='error'],'Notebook 包含执行错误'
run=json.loads((ROOT/'artifacts/run_summary.json').read_text())
sample=pd.read_csv(DATA/'sample_submission.csv')
sub=pd.read_csv(ROOT/'submission.csv')
assert sub.shape==(28512,2) and sub.columns.tolist()==['id','sales']
assert sub.id.equals(sample.id) and sub.id.is_unique
assert np.isfinite(sub.sales).all() and sub.sales.ge(0).all()
hold=pd.read_csv(ROOT/'artifacts/holdout_predictions.csv')
original=pd.read_csv(DATA/'train.csv',usecols=['date','store_nbr','family','sales'])
original=original.loc[original.date.between('2017-07-31','2017-08-15')]
j=hold.merge(original,on=['date','store_nbr','family'],validate='one_to_one',suffixes=('_report','_original'))
assert len(j)==28512
assert np.allclose(j.sales_report,j.sales_original,rtol=0,atol=1e-9)
rmsle=float(np.sqrt(np.mean((np.log1p(j.prediction)-np.log1p(j.sales_original))**2)))
assert abs(rmsle-run['holdout_metrics']['RMSLE'])<1e-12
dev=pd.read_csv(ROOT/'artifacts/development_leaderboard.csv')
assert dev.iloc[0].model==run['selected_model']
assert len(pd.read_csv(ROOT/'artifacts/development_fold_metrics.csv'))==10
assert len(pd.read_csv(ROOT/'artifacts/search_fold_metrics.csv'))==6
score=run['holdout_metrics']
abstract=f'''## 实验摘要
本项目比较周期基线与四种监督回归方法，在两个开发窗口进行选型与小范围参数优化，锁定 **{run['selected_model']}**。在未参与选择的最后 16 天保留窗口中，取得 **RMSLE={score['RMSLE']:.5f}、MAE={score['MAE']:.3f}、RMSE={score['RMSE']:.3f}、R²={score['R2']:.4f}**。按既定近年窗口重新训练后生成 **28,512** 条预测。数字均来自本次实际运行，Kaggle 线上分数待实际提交后补充。

关键词：销量预测；时间序列；梯度提升；特征工程；滚动验证。
'''
n.cells.insert(1,nbformat.v4.new_markdown_cell(abstract))
nbformat.validate(n);nbformat.write(n,path)
exporter=HTMLExporter();exporter.embed_images=True
body,_=exporter.from_notebook_node(n)
body=body.replace('</head>','''<style>
body{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif;line-height:1.65}
.jp-Notebook{max-width:1180px;margin:auto;padding:30px}
.jp-RenderedHTMLCommon table{font-size:14px}
@media print{.jp-Notebook{max-width:none;padding:0}.jp-InputArea{break-inside:avoid}}
</style></head>''')
(ROOT/'Store_Sales.html').write_text(body,encoding='utf-8')
class Inspect(HTMLParser):
    def __init__(self):super().__init__();self.images=0;self.embedded=0;self.tables=0
    def handle_starttag(self,t,attrs):
        a=dict(attrs)
        if t=='img':self.images+=1;self.embedded+=int(a.get('src','').startswith('data:image/'))
        if t=='table':self.tables+=1
p=Inspect();p.feed(body)
assert p.images>=8 and p.images==p.embedded
assert p.tables>=8
verification={'notebook_code_cells':len(code),'notebook_error_outputs':0,'embedded_plot_images':p.embedded,'html_tables':p.tables,'submission_rows':len(sub),'independently_recomputed_holdout_rmsle':rmsle,'holdout_labels_match_original':True,'submission_id_order_matches_template':True}
(ROOT/'artifacts/verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2))
readme=ROOT/'README.md'
text=readme.read_text()
text=text.split('\n## 本次实际运行结果')[0]
text+=f'''\n## 本次实际运行结果

最终模型：**{run['selected_model']}**。最后保留窗口：

| RMSLE | MAE | RMSE | R² |
|---:|---:|---:|---:|
| {score['RMSLE']:.5f} | {score['MAE']:.3f} | {score['RMSE']:.3f} | {score['R2']:.4f} |

全部 {len(code)} 个代码单元格执行成功；HTML 含 {p.embedded} 组嵌入图表。28,512 行提交的 ID 顺序、唯一性、有限值和非负值均已检查。保留窗口标签与原始数据逐行对齐，RMSLE 已独立复算。Kaggle 线上评分尚未提交。
'''
readme.write_text(text)
archive=ROOT/'Store_Sales_Assignment.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name in ['Store_Sales.ipynb','Store_Sales.html','submission.csv','requirements.txt','LICENSE']:
        z.write(ROOT/name,name)
    baseline_readme=ROOT/'README_BASELINE.md'
    z.write(baseline_readme if baseline_readme.exists() else ROOT/'README.md','README.md')
    for name in ['README_BASELINE.md','Store_Sales_Baseline_Data_Analysis_Report.docx','Store_Sales_Baseline_Data_Analysis_Report.pdf','Store_Sales_Baseline_Data_Analysis_Report.md','requirements-report.txt']:
        if (ROOT/name).exists():z.write(ROOT/name,name)
    for folder in ['artifacts']:
        for f in sorted((ROOT/folder).rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts:z.write(f,f.relative_to(ROOT))
    for name in ['build_notebook.py','execute_report.py','verify_export.py','build_baseline_word_report.py']:
        if (ROOT/'scripts'/name).exists():z.write(ROOT/'scripts'/name,'scripts/'+name)
    z.write(ROOT/'data/README.md','data/README.md')
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print(json.dumps({'run':run,'verification':verification,'archive_bytes':archive.stat().st_size},ensure_ascii=False,indent=2))
