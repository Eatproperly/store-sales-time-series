# Store Sales 进阶版实际执行说明

完整执行了CPU受限版本：起点建样本、增强树、6组网络/消融、4窗确认、3种子复验和对数融合。

最终开发选择：B3（网格标记blend_1.0）；四窗开发RMSLE：0.501776。
固定回归窗口：0.406031。线上分数待实际提交。

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

## 完整GPU实验入口
保持Kaggle预装CUDA版PyTorch，使用requirements-advanced-gpu.txt。打开Store_Sales_Advanced_GPU.ipynb，启用GPU并挂载原始数据后运行。该配置width64、全窗口、最多50轮、内层patience6；当前CPU报告不包含这一GPU实测。

## Word数据分析报告
[下载Word报告](Store_Sales_Data_Analysis_Report.docx) · [PDF预览](Store_Sales_Data_Analysis_Report.pdf)。按新版House Prices示例改为正文为主的连续排版，仅保留1张模型结果表；详细图表在Notebook中。Word中文宋体、英文Times New Roman，正文12磅、主标题14磅、一级标题22磅、二级标题16磅，姓名学号留空。

正文源稿为docs/advanced_word_report.md，指标占位符读取已保存实验。修改后安装requirements-report.txt，运行 `python scripts/build_word_report.py --pdf` 重新生成。PDF还需LibreOffice和Noto CJK字体；仅在临时渲染副本中替换本机缺少的字体，交付Word保留宋体/Times New Roman。该命令不重新训练模型。
