# Store Sales 基础版课程作业

本目录对应基础版实验：周期基线、Ridge、决策树、随机森林、LightGBM和三组LightGBM参数比较。Word参考House Prices示例，包含摘要、八章正文、参考资料和本项目真实图表；封面姓名、学号留空。

- [基础版Word报告](Store_Sales_Baseline_Data_Analysis_Report.docx)
- [PDF预览](Store_Sales_Baseline_Data_Analysis_Report.pdf)
- [完整基础版作业包](https://github.com/Eatproperly/store-sales-time-series/raw/refs/heads/main/Store_Sales_Assignment.zip)
- [Notebook](Store_Sales.ipynb) / [HTML](Store_Sales.html) / [提交预测](submission.csv)

最终模型：LightGBM_tuned_3。两个开发窗口为2017-06-29—07-14、2017-07-15—07-30，合并RMSLE=0.39144。最后保留窗口为2017-07-31—08-15，RMSLE=0.42611、MAE=76.18、RMSE=267.47、R²=0.9541。正式测试预测28,512行，Kaggle Public Score待真实提交。

报告仅整理基础版既有实验，没有重新训练；不引用进阶版网络、四窗回测或增强树的结果。最后保留窗口在基础版当时未参与选型；本次引用历史结果，不称新盲测。

## 复现

将原始七个CSV放入data/或设置STORE_SALES_DATA_DIR，然后安装requirements.txt。原基础版Notebook可直接重新运行，或执行：

```bash
python scripts/execute_report.py
python scripts/verify_export.py
```

补写或更新Word只读取已保存实验，不训练模型：

```bash
python -m pip install -r requirements-report.txt
python scripts/build_baseline_word_report.py --pdf
```

PDF转换需要LibreOffice以及Noto Serif CJK SC、Noto Sans CJK SC字体；省略`--pdf`可仅生成可编辑DOCX和Markdown。报告核验记录位于artifacts/baseline_word_report_verification.json，包括独立复算RMSLE、原始标签匹配、报告及实验文件哈希。
