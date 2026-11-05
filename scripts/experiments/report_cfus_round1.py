"""Independent numerical audit and Chinese report for the locked CFUS screen."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/csi300_aligned/models/cfus_round1_20261022'


def md(frame):
    # No optional tabulate dependency.
    frame=frame.astype(str)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |',
        '| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(row)+' |' for row in frame.to_numpy().tolist()])


def main():
    cfg=json.loads((OUT/'locked_config.json').read_text(encoding='utf-8'))
    split=pd.read_csv(OUT/'split_episodes.csv')
    grid=pd.read_csv(OUT/'selection_grid_daily_horizon.csv')
    steps=pd.read_csv(OUT/'selected_steps.csv')
    raw=pd.read_csv(OUT/'evaluation_daily_horizon.csv')
    metrics=pd.read_csv(OUT/'metrics.csv')
    paired=pd.read_csv(OUT/'paired_block10_comparisons.csv')
    half=pd.read_csv(OUT/'half_metrics.csv')
    checks={}
    for earlier,later in [('fit','selection'),('selection','evaluation')]:
        assert split[split.stage.eq(earlier)].query_label_end.max()<split[split.stage.eq(later)].query_date.min()
    assert (split.support_label_end.eq(split.update_date)).all()
    assert (split.support_label_end<split.query_date).all()
    assert (split.support_origin<split.update_date).all()
    checks['split_maturity_and_next_day_activation']=True
    assert len(raw)==2*2*7*86*10
    assert not raw.duplicated(['checkpoint','protocol','method','query_index','horizon']).any()
    counts=raw.groupby(['checkpoint','protocol','method']).size()
    assert counts.eq(860).all()
    checks['all_methods_identical_86_query_dates_and_10_horizons']=True
    calc=raw.groupby(['checkpoint','protocol','method'])[['mse','mae']].mean().sort_index()
    saved=metrics.set_index(['checkpoint','protocol','method'])[['mse','mae']].sort_index()
    np.testing.assert_allclose(calc,saved,rtol=1e-12,atol=1e-15)
    checks['row_level_recomputed_metrics']=True
    for row in steps.itertuples():
        g=grid[(grid.checkpoint==row.checkpoint)&(grid.protocol==row.protocol)&(grid.method==row.method)]
        ranked=g.groupby('alpha').mse.mean().reset_index().sort_values(['mse','alpha'])
        assert float(ranked.iloc[0].alpha)==row.alpha
    checks['all_selected_steps_minimize_selection_mse_only']=True
    support_tables=[]
    for ck in ['Prediction','RankIC']:
        table=pd.read_csv(OUT/f'{ck}_support_events.csv')
        assert not table.local_row.duplicated().any()
        assert table.groupby('query_date').size().eq(32).all()
        assert len(table)==146*32
        assert table.support_label_end.eq(table.update_date).all()
        assert (table.support_label_end<table.query_date).all()
        support_tables.append(table)
    pd.testing.assert_frame_equal(*support_tables)
    checks['continuous_supervision_unique_causal_and_same_between_checkpoints']=True
    refs=[]
    for sub in ['prediction_adapter','adapter_drankic_dynamic_static_20261013']:
        ref=pd.read_csv(ROOT/f'experiments/csi300_aligned/models/{sub}/predictions.csv').sort_values(['as_of_date','stock']).reset_index(drop=True)
        refs.append(ref)
    pd.testing.assert_frame_equal(refs[0][['as_of_date','stock']],refs[1][['as_of_date','stock']])
    cols=[f'actual_return_{k}' for k in range(1,11)]
    label_delta=float(abs(refs[0][cols].to_numpy()-refs[1][cols].to_numpy()).max())
    assert label_delta<5e-8
    checks['archived_label_max_abs_difference_between_checkpoints']=label_delta
    checks['evaluation_stock_origin_rows_per_method']=int(refs[0].as_of_date.isin(split[split.stage.eq('evaluation')].query_date).sum())
    checks['evaluation_daily_metric_weight']='equal dates, equal horizons; within date equal available stocks'
    dump=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    dump(OUT/'independent_recalculation.json',checks)

    rows=[]
    for ck in ['Prediction','RankIC']:
        for pr in ['one_step','continuous']:
            a=paired[(paired.checkpoint==ck)&(paired.protocol==pr)].set_index('baseline')
            rows.append({'起点':ck,'协议':pr,
                '相对Frozen MSE改善':f'{a.loc["Frozen"].cfus_mse_reduction_pct:+.4f}%',
                '相对PCA MSE改善':f'{a.loc["TargetPCA"].cfus_mse_reduction_pct:+.4f}%',
                '相对Full MSE改善':f'{a.loc["Full"].cfus_mse_reduction_pct:+.4f}%',
                '相对Full的95%区间':f'[{a.loc["Full"].ci95_low:+.4f}%, {a.loc["Full"].ci95_high:+.4f}%]'})
    comparison=md(pd.DataFrame(rows))
    continuous=metrics[metrics.protocol.eq('continuous')].pivot(index='method',columns='checkpoint',values='mse').reindex(cfg['methods'])
    continuous=continuous.map(lambda x:f'{x:.9f}').reset_index()
    fit=pd.read_csv(OUT/'fit_selected_one_step_daily_horizon.csv')
    selected_grid=grid.merge(steps[['checkpoint','protocol','method','alpha']],on=['checkpoint','protocol','method','alpha'])
    stage_rows=[]
    for label,d in [('拟合期（样本内）',fit),('步长选择期',selected_grid[selected_grid.protocol.eq('one_step')]),('后续评估期',raw[raw.protocol.eq('one_step')])]:
        m=d.groupby(['checkpoint','method']).mse.mean().unstack()
        for ck in ['Prediction','RankIC']:
            stage_rows.append({'起点':ck,'时间段':label,
                'CFUS相对Frozen':f'{100*(1-m.loc[ck,"CFUS"]/m.loc[ck,"Frozen"]):+.5f}%',
                'PCA相对Frozen':f'{100*(1-m.loc[ck,"TargetPCA"]/m.loc[ck,"Frozen"]):+.5f}%'})
    text=f'''# CFUS 第一轮真实数据验证

结论：**未通过预先锁定的继续条件，不建议扩大该版本的实验或叠加 OA-CFUS。** 在本次 CSI300、H=10、固定秩2、冻结表示、普通SGD、完整成熟反馈设置下，CFUS 没有同时稳定超过目标PCA和调过步长的全空间更新。这个结论限定于当前版本和数据，不是对所有效用子空间方法的否定。

本轮是探索性筛选：原226日测试区间已经被多轮查看。当前按时间隔离出的后段也不能称为独立确认集。实验不支持提高论文中稿率的推断。

## 锁定设置与数据

- 基拟合：2025-07-16—2025-10-14，59个查询日期；对应全部查询标签在步长选择开始前成熟。
- 步长选择：2025-10-29—2026-01-08，50个查询日期；对应全部标签在评估开始前成熟。
- 后续评估：2026-01-23—2026-06-05，86个日期、25,210个股票起点、252,100个期限标签/方法。
- 两个离线起点：Prediction、RankIC，分别在各自冻结起点内比较；不把两者的离线差异归因于在线方法。
- 保留 HCA 形式；冻结 Kronos、Adapter、原预测头，只更新新增系数头。子空间方法均为秩2、1,026个参数；Full 为10维、5,130个参数。无Adam、权衰减、动量、裁剪或dropout。
- 基线：Frozen、目标PCA、残差PCA、DCT、固定随机正交基、CFUS、Full。目标PCA使用原训练期保存基；残差PCA和CFUS只用本轮拟合段。CFUS固定取前2个方向，没有正特征值门控。
- 步长候选：alpha ∈ {{0, .001, .003, .01, .03, .1, .3, 1}}。eta=alpha×10/拟合期支持特征平均平方范数。每个方法只依据选择期MSE定步长。CFUS每个正步长在拟合期计算相应基，候选数相同。
- 每次支持更新取前一预测日刚完整成熟的一条起点路径，固定哈希挑32只股票，消费320个标签标量。所有方法监督键相同。查询最早在更新后的下一预测日发布；不修订旧预测。
- 单次协议每个episode从零系数头开始；连续协议在选择期开始时置零，随后每天更新，跨越不评分的边界间隔并持续到评估末尾。连续流共146次更新机会、46,720个不重复监督标量，包含评估前60次；alpha=0时参数不变。
- 核心指标为日期等权MSE/MAE，日期内股票与期限等权。95%区间使用配对循环日期块bootstrap（块长10、4,000次）；区间描述当前窗口，未做多重比较校正。

## 主结果

下表改善率=(基线MSE−CFUS MSE)/基线MSE；正值表示CFUS更好。

{comparison}

连续在线MSE全表（越低越好）：

{md(continuous)}

Prediction连续协议的PCA、Full、残差PCA、DCT在选择期均选择alpha=0，因此该列中它们与Frozen完全相同。CFUS选alpha=.3，只有0.1834%的改善；相对Random仅改善0.0828%，95%区间[-0.0099%, 0.1689%]。这是一条小的正向信号，不能据此声称已经获得强子空间优势。

RankIC连续协议中，CFUS较Frozen改善1.0677%，但目标PCA改善1.0968%、残差PCA改善1.1163%、DCT改善2.7224%、Full改善2.7718%。CFUS较Full的MAE也恶化1.8797%。因此“在线更新有效”与“CFUS优于普通更新”是两件需要分别验证的事；本轮只支持前者在部分设置成立。

## 为什么没有通过

CFUS在拟合期最大化经验单次效用，这个代数性质已核对；它没有保证效用方向在后续市场仍占优。使用各自在选择期定下的步长，单次协议的跨时间表现如下：

{md(pd.DataFrame(stage_rows))}

Prediction中，CFUS在拟合期优于PCA，到了选择期和评估期却反向，评估期比PCA差0.2657%。RankIC中，拟合期CFUS相对Frozen的0.54378%略高于PCA的0.52364%，但后续不再优于PCA。这与跨时间效用估计不稳定一致；还不能仅凭本轮区分有限样本噪声、分布变化与目标错配各占多少。

单次目标也没有为连续更新带来稳定优势。RankIC连续流中，CFUS与PCA都选alpha=.03，CFUS仍略差，Full选择alpha=.1后显著更好。不能把连续流中与Full的全部差距只归因于基方向，因为两者选定步长不同；单次协议的同一步长比较已经没有显示CFUS稳定胜出。

时间稳定性同样不够：Prediction单次相对PCA在前43日和后43日分别恶化0.2361%、0.2895%；RankIC连续相对Full分别恶化2.1933%、1.3887%。不是只由一个后段平均值掩盖了稳定优势。

本轮预先设置的项目筛选线为：相对PCA和调优Full均至少2% MSE改善、MAE下降、前后半段同向、远期8—10步整体不严重反转。4个“起点×协议”组合全部未通过。2%是本项目投入筛选线，并非ICDE规则；即使不采用2%线，也没有跨两个起点稳定优于基线的证据。

## 正确性、成本和限制

缓存缺少meta.csv及hidden_val.npy，hidden_train.npy实际长度不满足头部声明的形状，未使用其训练特征做效能实验。使用保存的测试预测CSV按日期和股票重建对齐；全部66,103×10冻结预测与原归档比对，最大误差Prediction为6.50e-8、RankIC为5.96e-8。隐藏缓存和模型检查点SHA256均吻合历史记录。

普通SGD实际支持样本的autograd一步与闭式响应误差小于1.1e-19；正交基旋转不变性与效用trace恒等式均通过。将未来查询标签置NaN不影响该支持更新。另从逐日逐期限CSV独立重算全部均值、选步长最优性、监督不重复、查询日期一致性和切分成熟边界，检查通过。

两个历史预测CSV的实际标签存在至多{label_delta:.3e}的浮点差异；每个起点内部所有方法使用完全相同的标签。其量级不足以解释本轮方法间差距，但后续独立确认应使用统一canonical标签缓存。

本轮运行主体约10.2秒、CPU，包含冻结特征计算和候选重放，不含Python导入、结果bootstrap与报告，也不含早期Kronos隐藏缓存生成。该数字只说明本地小验证成本低，不能作为端到端系统吞吐结论。

仍未覆盖部分标签、额外延迟、不规则释放、其他预测长度、非金融数据和多种子。RankIC单次协议有多个方法选到步长网格上界1，不能称为全局最优学习率；但CFUS并未优于同一步长PCA/DCT，因此无需扩大网格来维持当前继续条件。固定随机基仅一个种子；本轮也没有独立的meta-preconditioner近邻基线，不能用于证明方法原创性。

## 研究决策

停止把当前固定CFUS作为已经验证有效的论文主创新；不进入OA-CFUS，不恢复MOG，不用交易收益选择性覆盖预测误差结果。保留HCA及本轮失败对照，它揭示“目标路径压缩”和“未来更新效用”都需要跨时间验证，而效用谱的源期最优本身不足以支撑主贡献。

若未来重新启动该方向，应先在新的校准数据上证明一个具体、可复现的失败来源及修正必要性，再形成新版本和新锁定方案。当前数据不足以支持追加模块、大规模证明实验或“提高中稿率”的论断。

复现：在仓库根目录运行 `.runtime/doubleadapt_env/Scripts/python.exe scripts/experiments/run_cfus_round1.py prepare`，然后 `... run`，最后运行 `scripts/experiments/report_cfus_round1.py`。prepare拒绝覆盖已有目录；复现实验前需选择新的OUT目录。原始结果、锁定配置、选步长表、逐日期逐期限误差、基矩阵、审计和完整比较均保存在本目录。
'''
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='artifact_hashes.json'}
    dump(OUT/'artifact_hashes.json',files)
    print(json.dumps(checks,ensure_ascii=False,indent=2))
    print(str(OUT/'REPORT.md'))


if __name__=='__main__':
    main()
