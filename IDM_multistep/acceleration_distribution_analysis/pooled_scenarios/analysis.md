# R3C7：按场景合并的加速度分布

本版按用户要求合并 Dec/Dis、全部 CAV 和子系统规模。每个场景单独归一化为概率密度，不因样本数不同改变曲线面积。

- HighD：合并四种规模的训练输入；使用当前轨迹级划分及种子42。每条轨迹的每个有效输入时间点只计一次，不重复计入重叠窗口。
- 复现：使用 Python 运行上一级目录中的 pool_scenarios.py；该脚本独立读取 highd_training_inputs.npz 和原始实验记录，生成本目录中的图片与统计。HighD 输入文件是保留的必要分析数据。
- Tracking：正文 Table III 对应的 Dec 和 Dis 两次实验，合并全部5辆CAV，共22080个实际输入。
- Braking：正文 Table III 对应的 Dec 和 Dis 两次实验，合并全部5辆CAV，共9330个实际输入。
- Cut-in：Comment 4 中15组参数×4种制动强度（-3、-3.5、-4、-4.5 m/s²）×6个切入时刻，共360次 distributed HK-RPC 实验，合并全部5辆CAV，共599400个实际输入。该组实验只有 distributed HK-RPC；不混入 nominal 或 all-HDVs，也不加入未纳入 Comment 4 的 -5 m/s² 结果。
- 各实验排除前50步初始化和没有对应实际输入的最终状态；cut-in使用共同333步控制区间。
- 所有360次cut-in均已完成，全部15组参数各有24次实验，各配置贡献相同的输入样本数。每个轨迹的加速度均与速度增量/采样时间相符。
- 绘图采用共同0.02 m/s²分箱及0.06 m/s²高斯平滑带宽；左图为整体密度，右图为同一密度的对数纵轴，用于观察尾部。统计量直接由原始输入计算，未使用平滑后的曲线。

| 分布 | 输入数 | 最小值 | 最大值 | 标准差 | 超出合并HighD观测范围 | 超出合并HighD中央99%区间 |
|---|---:|---:|---:|---:|---:|---:|
| HighD training | 1741438 | -4.4767 | 4.6700 | 0.4134 | 0.0000% | 0.9908% |
| Tracking | 22080 | -1.5775 | 1.5379 | 0.2256 | 0.0000% | 0.0453% |
| Braking | 9330 | -4.6101 | 4.0003 | 0.5841 | 0.0429% | 3.3762% |
| Cut-in | 599400 | -6.0005 | 2.1317 | 0.8653 | 0.4701% | 10.3790% |

合并HighD训练输入的中央99%区间为 [-1.4567, 1.1900] m/s²，观测范围为 [-4.4767, 4.6700] m/s²。

这些曲线用于描述不同场景下总体控制输入与自然驾驶输入的差异。合并后的覆盖比例不等于逐模型覆盖率；分布重叠也不能建立联合状态—历史—输入覆盖或无条件安全保证。尾部差异及其对外推的含义仍应在回复中说明。

## 建议回复信中的实验口径

To compare the naturalistic training inputs with the CAV interventions, we pooled the HighD training accelerations across subsystem sizes and pooled the actually applied accelerations of all CAVs separately for each evaluation scenario. For tracking and emergency braking, the decentralized and distributed HK-RPC results reported in Table III were combined within each scenario. For cut-in, we pooled all 360 distributed HK-RPC runs reported in our response to Comment 4, comprising 15 HDV parameter sets, four braking intensities, and six cut-in timings. Each scenario is represented by one normalized acceleration distribution. The initial data-collection period and terminal states without an applied input were excluded.

图和表尚未插入回复信或正文。数据来源见 source_cases.csv；统计见 pooled_scenario_statistics.csv。
