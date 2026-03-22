# Progress

- `project`: 当前仓库已具备本地 synthetic 数据上的监督训练闭环，下一步补全 Part 3 与 Part 4，并保持本地与 DCC 工作流一致。
- `data`: 已有 synthetic 数据生成和 MIMIC manifest 预处理；正在扩展为 supervised、SimCLR、embedding 共用的数据接口。
- `part2`: 已实现 Lightning 二分类训练、验证、测试、checkpoint、预测导出和基础曲线绘图。
- `part3`: 正在补 Grad-CAM 解释模块与批量导出脚本。
- `part4`: 正在补 SimCLR 预训练、冻结 encoder 的线性评估、embedding 提取与 t-SNE 可视化。
- `dcc`: 已有基础 Slurm 训练脚本；正在补 DCC 虚拟环境初始化脚本和 Part 3/4 对应 job。
