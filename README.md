# Store Sales：商店销量时间序列预测

<!-- ADVANCED_RESULTS_START -->
进阶版：[B3 实际实验报告](README_ADVANCED.md) · [下载进阶作业包](https://github.com/Eatproperly/store-sales-time-series/raw/refs/heads/main/Store_Sales_Advanced_Assignment.zip) · [GPU训练入口](Store_Sales_Advanced_GPU.ipynb)。本轮设备=cpu；Kaggle线上成绩待真实提交。
[Word数据分析报告](Store_Sales_Data_Analysis_Report.docx) · [PDF预览](Store_Sales_Data_Analysis_Report.pdf)
<!-- ADVANCED_RESULTS_END -->



进阶方案：[网络设计与实施计划](docs/superpowers/plans/2026-11-16-store-sales-advanced-plan.md)（设计阶段，收益待实际实验验证）。

参照 House Prices 课程作业的组织形式，完成商店未来 16 天销量预测。包含中文报告、实际运行输出、8 组图表、周期基线与四种回归算法、参数比较、时间回测和提交文件。

## 下载成果

- [完整作业包（ZIP）](https://github.com/Eatproperly/store-sales-time-series/raw/refs/heads/main/Store_Sales_Assignment.zip)
- [Notebook](Store_Sales.ipynb)：代码、中文说明和实际运行输出。
- [HTML 报告](Store_Sales.html)：下载后用浏览器打开，图表已嵌入。
- [预测文件](submission.csv)：28,512 行，id、sales 两列。
- [实验指标与核验记录](artifacts/)。

姓名与学号在 Notebook 首页留空，提交课程作业前填写。未代用户提交 Kaggle，线上分数待真实提交后补充。

## 实验方法

- 54 家商店、33 个品类，共 1,782 条日粒度销量序列。
- 使用 2016 年以来的训练标签，2013 年以来历史用于 EDA 和滞后特征。
- 开发窗口：2017-06-29—07-14、2017-07-15—07-30；每次完整预测 16 天。
- 最后保留窗口：2017-07-31—08-15，不参与选型调参。
- 销量特征至少滞后 28 天，避免窗口内实际销量泄漏。
- 周期基线、Ridge、决策树、随机森林、LightGBM 比较；梯度提升再比较三组候选参数。
- 主指标 RMSLE，辅助 MAE、RMSE、R²；预测保留小数并截断至非负。

## 复现

建议 Python 3.11/3.12、CPU、8 GB 以上内存。原始七个 CSV 不重复入库。将用户提供的 store-sales-time-series-forecasting.zip 解压至 data/，或设置 STORE_SALES_DATA_DIR。

```bash
python -m venv .venv
# 激活环境后安装：
python -m pip install -r requirements.txt
python scripts/build_notebook.py
python scripts/execute_report.py
python scripts/verify_export.py
```

也可安装 notebook，在仓库目录启动 Jupyter，打开 Store_Sales.ipynb，重启内核并运行所有单元格。build_notebook.py 会重新生成 Notebook，复现已有报告时可直接从 execute_report.py 开始。不同机器的耗时及小数位可能不同。

scripts/verify_export.py 独立复算保留窗口误差、核对真实标签与原始数据、检查提交 ID 与数值、导出 HTML 并打包。原始数据下载页面：[Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)。

## 本次实际运行结果

最终模型：**LightGBM_tuned_3**。最后保留窗口：

| RMSLE | MAE | RMSE | R² |
|---:|---:|---:|---:|
| 0.42611 | 76.180 | 267.466 | 0.9541 |

全部 29 个代码单元格执行成功；HTML 含 8 组嵌入图表。28,512 行提交的 ID 顺序、唯一性、有限值和非负值均已检查。保留窗口标签与原始数据逐行对齐，RMSLE 已独立复算。Kaggle 线上评分尚未提交。
