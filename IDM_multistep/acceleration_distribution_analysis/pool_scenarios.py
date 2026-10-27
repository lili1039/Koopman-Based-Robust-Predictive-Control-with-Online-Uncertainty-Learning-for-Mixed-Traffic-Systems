"""Reproduce the pooled HighD/tracking/braking/cut-in comparison for R3C7.

Run this script with Python, numpy, pandas and matplotlib. The retained HighD
input archive contains the pooled training inputs; evaluation inputs are read
from the original experimental records. No first-version script is required.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
HIGH_D_INPUTS = OUT / 'highd_training_inputs.npz'
RUNS = {
    ('Tracking', 'Dec'): 'tracking/Decentralized_MPC',
    ('Tracking', 'Dis'): 'tracking/Distributed_MPC',
    ('Braking', 'Dec'): 'braking/Decentralized_MPC',
    ('Braking', 'Dis'): 'braking/Distributed_MPC',
}
CAVS = (1, 4, 8, 10, 15)
NPST = 50

DEST = OUT / 'pooled_scenarios'
CUTIN = ROOT / 'IDM_multistep/result_cutin_robustness_15profiles'


def summary(a):
    q = np.quantile(a, [0, .005, .01, .025, .5, .975, .99, .995, 1])
    names = ['min', 'q005', 'q01', 'q025', 'median', 'q975', 'q99', 'q995', 'max']
    return {'n': len(a), 'mean': float(a.mean()), 'std': float(a.std()),
            **dict(zip(names, map(float, q))),
            **{f'abs_gt_{v}_pct': float((np.abs(a) > v).mean()*100) for v in (1, 2, 3)}}


def main():
    DEST.mkdir(exist_ok=True)
    with np.load(HIGH_D_INPUTS) as data:
        training_inputs = data['acceleration']
        assert len(training_inputs) == int(data['counts'].sum()) == 1741438
    arrays = {'HighD training': training_inputs}
    provenance = []
    for scenario in ('Tracking', 'Braking'):
        values = []
        for (s, method), rel in RUNS.items():
            if s != scenario:
                continue
            folder = ROOT / 'IDM_multistep/result' / rel
            for cav in CAVS:
                frame = pd.read_csv(folder / 'vehicle_data' / f'veh_{cav}.csv')
                a = frame.acceleration.to_numpy()[NPST:-1]
                assert np.allclose(np.diff(frame.velocity.to_numpy()[NPST:])/.12, a, atol=1e-8)
                values.append(a)
            provenance.append({'scenario': scenario, 'method': method, 'case_id': folder.name,
                               'path': str(folder), 'applied_steps': len(a), 'cavs': 5})
        arrays[scenario] = np.concatenate(values)

    manifest = pd.read_csv(CUTIN / 'manifest.csv')
    # These are exactly the four braking accelerations reported in Comment 4.
    cases = manifest[(manifest.method == 'distributed') & manifest.brake_mps2.isin([3, 3.5, 4, 4.5])]
    assert len(cases) == 15 * 4 * 6
    assert cases.groupby('profile').size().to_dict() == {i: 24 for i in range(1, 16)}
    values = []
    for case in cases.to_dict('records'):
        folder = CUTIN / 'cases' / case['case_id']
        status = json.loads((folder / 'case_metrics.json').read_text())
        assert status['completed'] == 1
        with np.load(folder / 'trajectory.npz') as npz:
            warmup = int(npz['history_steps'])
            dt = float(npz['dt'])
            states = npz['states'][warmup:warmup+334]
            indices = npz['nominal_cav_indices']
        assert states.shape == (334, 18, 3)
        assert np.array_equal(indices, list(CAVS))
        a = states[:-1, indices, 2]
        assert a.shape == (333, 5)
        assert np.isfinite(a).all()
        assert np.allclose(np.diff(states[:, indices, 1], axis=0)/dt, a, atol=1e-8)
        values.append(a.reshape(-1))
        provenance.append({'scenario': 'Cut-in', 'method': 'Distributed HK-RPC',
                           'case_id': case['case_id'], 'path': str(folder),
                           'applied_steps': 333, 'cavs': 5,
                           **{k: case[k] for k in ('profile', 'parameter_seed', 'brake_mps2', 'cutin_time_s')}})
    arrays['Cut-in'] = np.concatenate(values)
    assert len(arrays['Cut-in']) == 599400
    train = arrays['HighD training']
    q = np.quantile(train, [.005, .025, .975, .995])
    rows = []
    for label, a in arrays.items():
        rows.append({'scenario': label, **summary(a),
                     'outside_pooled_training_range_pct': ((a<train.min())|(a>train.max())).mean()*100,
                     'outside_pooled_training_99_pct': ((a<q[0])|(a>q[3])).mean()*100,
                     'outside_pooled_training_95_pct': ((a<q[1])|(a>q[2])).mean()*100})
    stats = pd.DataFrame(rows)
    stats.to_csv(DEST / 'pooled_scenario_statistics.csv', index=False)
    pd.DataFrame(provenance).to_csv(DEST / 'source_cases.csv', index=False)

    # All curves use the same bin width and Gaussian smoothing bandwidth.
    # This is a smoothed empirical histogram, not a Gaussian fit.
    width, bandwidth = .02, .06
    edges = np.arange(-7, 6+width/2, width)
    centers = (edges[:-1]+edges[1:])/2
    offset = np.arange(-int(np.ceil(4*bandwidth/width)), int(np.ceil(4*bandwidth/width))+1)
    kernel = np.exp(-.5*(offset*width/bandwidth)**2)
    kernel /= kernel.sum()
    colors = {'HighD training': '#333333', 'Tracking': '#2774ad', 'Braking': '#d35c35', 'Cut-in': '#389469'}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    density_rows = {'acceleration_mps2': centers}
    for label, a in arrays.items():
        counts, _ = np.histogram(a, bins=edges)
        assert counts.sum() == len(a)
        density = np.convolve(counts/(len(a)*width), kernel, mode='same')
        assert abs(density.sum()*width-1) < 1e-8
        density_rows[label] = density
        for ax in axes:
            ax.plot(centers, density, label=label, color=colors[label], lw=1.8)
    for ax in axes:
        ax.set(xlim=(-6.3, 5), xlabel='Acceleration (m/s²)', ylabel='Probability density')
        ax.grid(alpha=.2, which='major')
        ax.legend(frameon=True, fontsize=9)
    axes[0].set_title('(a) Acceleration distributions', fontsize=12)
    axes[0].set_ylim(bottom=0)
    axes[1].set(title='(b) Acceleration distributions with a logarithmic scale', yscale='log', ylim=(1e-5,20))
    pd.DataFrame(density_rows).to_csv(DEST / 'plotted_densities.csv', index=False)
    fig.savefig(DEST / 'scenario_acceleration_distributions.png', dpi=200)
    fig.savefig(DEST / 'scenario_acceleration_distributions.pdf')
    plt.close(fig)

    lines = [
        '# R3C7：按场景合并的加速度分布',
        '',
        '本版按用户要求合并 Dec/Dis、全部 CAV 和子系统规模。每个场景单独归一化为概率密度，不因样本数不同改变曲线面积。',
        '',
        '- HighD：合并四种规模的训练输入；使用当前轨迹级划分及种子42。每条轨迹的每个有效输入时间点只计一次，不重复计入重叠窗口。',
        '- 复现：使用 Python 运行上一级目录中的 pool_scenarios.py；该脚本独立读取 highd_training_inputs.npz 和原始实验记录，生成本目录中的图片与统计。HighD 输入文件是保留的必要分析数据。',
        '- Tracking：正文 Table III 对应的 Dec 和 Dis 两次实验，合并全部5辆CAV，共22080个实际输入。',
        '- Braking：正文 Table III 对应的 Dec 和 Dis 两次实验，合并全部5辆CAV，共9330个实际输入。',
        '- Cut-in：Comment 4 中15组参数×4种制动强度（-3、-3.5、-4、-4.5 m/s²）×6个切入时刻，共360次 distributed HK-RPC 实验，合并全部5辆CAV，共599400个实际输入。该组实验只有 distributed HK-RPC；不混入 nominal 或 all-HDVs，也不加入未纳入 Comment 4 的 -5 m/s² 结果。',
        '- 各实验排除前50步初始化和没有对应实际输入的最终状态；cut-in使用共同333步控制区间。',
        '- 所有360次cut-in均已完成，全部15组参数各有24次实验，各配置贡献相同的输入样本数。每个轨迹的加速度均与速度增量/采样时间相符。',
        '- 绘图采用共同0.02 m/s²分箱及0.06 m/s²高斯平滑带宽；左图为整体密度，右图为同一密度的对数纵轴，用于观察尾部。统计量直接由原始输入计算，未使用平滑后的曲线。',
        '',
        '| 分布 | 输入数 | 最小值 | 最大值 | 标准差 | 超出合并HighD观测范围 | 超出合并HighD中央99%区间 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(f"| {row['scenario']} | {row['n']} | {row['min']:.4f} | {row['max']:.4f} | {row['std']:.4f} | {row['outside_pooled_training_range_pct']:.4f}% | {row['outside_pooled_training_99_pct']:.4f}% |")
    lines += [
        '',
        f'合并HighD训练输入的中央99%区间为 [{q[0]:.4f}, {q[3]:.4f}] m/s²，观测范围为 [{train.min():.4f}, {train.max():.4f}] m/s²。',
        '',
        '这些曲线用于描述不同场景下总体控制输入与自然驾驶输入的差异。合并后的覆盖比例不等于逐模型覆盖率；分布重叠也不能建立联合状态—历史—输入覆盖或无条件安全保证。尾部差异及其对外推的含义仍应在回复中说明。',
        '',
        '## 建议回复信中的实验口径',
        '',
        'To compare the naturalistic training inputs with the CAV interventions, we pooled the HighD training accelerations across subsystem sizes and pooled the actually applied accelerations of all CAVs separately for each evaluation scenario. For tracking and emergency braking, the decentralized and distributed HK-RPC results reported in Table III were combined within each scenario. For cut-in, we pooled all 360 distributed HK-RPC runs reported in our response to Comment 4, comprising 15 HDV parameter sets, four braking intensities, and six cut-in timings. Each scenario is represented by one normalized acceleration distribution. The initial data-collection period and terminal states without an applied input were excluded.',
        '',
        '图和表尚未插入回复信或正文。数据来源见 source_cases.csv；统计见 pooled_scenario_statistics.csv。',
    ]
    (DEST / 'analysis.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(stats.to_string(index=False))
    print(f'Included {len(cases)} completed cut-in HK-RPC runs from all 15 parameter profiles.')


if __name__ == '__main__':
    main()
