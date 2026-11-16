# Store Sales：商店销量时间序列预测

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
