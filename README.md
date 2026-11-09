# 交叉口车型组成预测

利用 D1、D2 历史观测和 D3 的 5 个已知交叉口，预测 D3 其余 13 个交叉口在 24 小时内的 8 类车型占比，共 312 条记录。

## 项目结构

- `作业一.md`：作业要求。
- `作业思路.md`：建模思路与初步实验。
- `作业一/01_data_analysis.py`：数据核验、前期分析和绘图。
- `作业一/02_prediction.py`：D1 历史基线、时空动态校准及预测文件导出。
- `作业一/03_validation.py`：内部验证、参数搜索及误差分析。
- `作业一/course_data.csv`：公开观测数据。
- `作业一/submission_template.xlsx`：提交模板。
- `作业一/figures/`：分析和验证图件。
- `作业一/outputs/`：预测、提交文件及验证指标。

## 运行

建议使用 Python 3.12。已有虚拟环境可直接使用；新机器可在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe 作业一/01_data_analysis.py
.\.venv\Scripts\python.exe 作业一/03_validation.py
.\.venv\Scripts\python.exe 作业一/02_prediction.py
```

脚本按自身所在目录定位数据，输出到 `作业一/figures/` 和 `作业一/outputs/`。图件优先使用 Microsoft YaHei 或 SimHei；其他系统如无这些字体，需安装中文字体或调整脚本字体设置。

## 模型与验证

基线直接使用 D1 同交叉口、同小时的车型占比。最终模型利用 D3 已知点相对 D1 的占比变化，按相对距离和 D1 历史组成相似性加权传播，以校准系数缩放后修正基线，并进行非负截断和归一化。

内部验证在 D1、D2 上保留与 D3 相同的 5 个已知点，隐藏其余 13 个点，比较 MAE、RMSE 和自然对数定义的 JSD。参数搜索与汇报使用相同的跨日验证集，指标属于内部调参结果，并非独立测试结果。D1 与 D2 日类型不同，不能据此保证 D3 的真实误差或将其视为误差上界。

当前实现使用无量纲相对坐标构建空间权重；道路连接图为参考资料，模型未使用道路拓扑。D3 未知点没有公开真值，输出约束检查不等同于预测精度检验。

虚拟环境和本地凭据不纳入版本控制；数据、图件及预测结果随项目保存。
