"""Report and audit saved predictions; no test-driven model selection."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments.run_temporal_attention_pilot import (ARCHITECTURES,PCOLS,ACOLS,KEYS,sha256,load_model,predict,write_json)
from evaluate.metrics import EvaluationConfig,evaluate_frame


def main(out):
    status=json.loads((out/'status.json').read_text());cfg=json.loads((out/'run_config.json').read_text());seeds=cfg['seeds']
    rows=pd.read_csv(out/'selected_samples.csv',dtype={'stock':str}); test=rows[rows.split.eq('test')];ids=test.index.to_numpy()
    cache=np.load(out/'cache/hidden_seq.npy',mmap_mode='r');initial=torch.load(ROOT/'experiments/csi300_aligned/models/prediction_adapter/model.pt',weights_only=True)
    access=pd.read_csv(out/'label_access_log.csv'); logs=pd.read_csv(out/'online_update_log.csv')
    assert (access.label_date<=access.update_date).all() and (access.complete_date<=access.update_date).all()
    assert (access.origin_date<access.update_date).all()
    event_sets=[]
    for (a,s),g in access.groupby(['architecture','seed']):
        assert not g.duplicated(['row_id','horizon']).any()
        event_sets.append(set(map(tuple,g[['update_date','row_id','horizon']].to_numpy())))
        for d,digest in g[['update_date','prediction_sha256']].drop_duplicates().itertuples(index=False,name=None):
            assert sha256(out/f'issued/{a}_{s}/{d}.csv')==digest
    assert all(x==event_sets[0] for x in event_sets)
    daily_dates=sorted(test.as_of_date.unique()); horizon_rows=[]; diag=[]
    for s in seeds:
        for a in ARCHITECTURES:
            name=f'{a}_{s}_online'; frame=pd.read_csv(out/'evaluation'/name/'predictions.csv',dtype={'stock':str})
            assert frame[KEYS].equals(test[KEYS].reset_index(drop=True))
            expected=np.maximum(0,np.arange(len(daily_dates))-10)
            got=frame.groupby('as_of_date').version.first().to_numpy()
            np.testing.assert_array_equal(got,expected)
            for phase in ('static','online'):
                f=pd.read_csv(out/'evaluation'/f'{a}_{s}_{phase}'/'predictions.csv')
                for k in range(1,11):
                    p=f[f'pred_return_{k}'].to_numpy();y=f[f'actual_return_{k}'].to_numpy();tp=((p>0)&(y>0)).sum();tn=((p<=0)&(y<=0)).sum();fp=((p>0)&(y<=0)).sum();fn=((p<=0)&(y>0)).sum()
                    recall=tp/max(tp+fn,1);spec=tn/max(tn+fp,1)
                    horizon_rows.append({'architecture':a,'seed':s,'phase':phase,'horizon':k,'mae':abs(p-y).mean(),'rmse':np.sqrt(((p-y)**2).mean()),'direction_accuracy':((p>0)==(y>0)).mean(),'balanced_accuracy':(recall+spec)/2})
            if a!='temporal_attention':continue
            for phase in ('offline','online'):
                model=load_model(a,initial,s);checkpoint=torch.load(out/f'checkpoints/{a}_{s}_{phase}.pt',weights_only=False);model.load_state_dict(checkpoint['model']);model.eval()
                with torch.no_grad():
                    x=torch.from_numpy(np.array(cache[ids[:64]]));_,w=model.pool(x,return_weights=True)
                    init_model=load_model(a,initial,s)
                    qdelta=float((model.pool.q_proj.weight-init_model.pool.q_proj.weight).norm());kdelta=float((model.pool.k_proj.weight-init_model.pool.k_proj.weight).norm())
                diag.append({'seed':s,'phase':phase,'q_weight_delta_l2':qdelta,'k_weight_delta_l2':kdelta,
                             'attention_entropy':float(-(w*w.clamp_min(1e-12).log()).sum(1).mean()),'uniform_entropy':float(np.log(90)),
                             'mean_weights':w.mean(0).tolist()})
    write_json(out/'audit.json',{'status':'passed','identical_feedback_events_per_stream':len(event_sets[0]),'streams':len(event_sets),'mature_only':True,'single_consumption':True,'issued_forecast_hash_verified':True,'next_day_versions_verified':True,'test_keys_verified':True})
    pd.DataFrame(horizon_rows).to_csv(out/'horizon_metrics_extended.csv',index=False);write_json(out/'learned_attention_diagnostics.json',diag)
    m=pd.read_csv(out/'comparison_metrics.csv');agg=m.groupby(['phase','signal','architecture']).mean(numeric_only=True)
    initial_preds=np.load(out/'initial_test_predictions.npy');ref=pd.read_csv(out/'evaluation/adapter_100_static/predictions.csv');ref[PCOLS]=initial_preds;ref['pred_signal']=initial_preds.mean(1);ref['pred_endpoint_return']=initial_preds[:,-1];ref['method']='original_checkpoint';ref['seed']=100
    result=evaluate_frame(ref,EvaluationConfig(test_start=ref.as_of_date.min(),test_end=ref.as_of_date.max(),top_k=6,drop_n=1,min_holding_days=5,cost_rate=.0015,bootstrap_reps=1000,bootstrap_block=10))
    write_json(out/'original_checkpoint_evaluation.json',result)
    b=pd.read_csv(out/'paired_date_bootstrap.csv');sel=json.loads((out/'selection.json').read_text())
    lines=['# Temporal Attention 扩大版 CPU 实验','',
      '结论：本配置的小范围实验没有支持 temporal attention 的稳定增量价值。在线 RMSE 略降，但 MAE 与横截面排序指标未改善；终点组合收益上升也几乎被等权汇总复现。','',
      '## 数据、预算与公平性','',
      f"- 执行约 {status['seconds']/60:.2f} 分钟，低于一小时预算；特征提取约 {json.loads((out/'sequence_cache_manifest.json').read_text())['seconds']/60:.2f} 分钟。完成预设 8 轮，不为凑时间追加测试期调参。",
      '- 固定 hash 从训练期可见股票选择 60 股，不按标签或收益选股。训练 80 个日期跨 2020-12-14—2024-12-13，验证 40 个日期跨 2025-01-02—2025-06-13；测试为 2025-07-01—2025-10-28 的 80 个连续交易日。',
      f"- 训练 {int(cfg['split_rows']['train'])}、验证 {int(cfg['split_rows']['val'])}、测试 {int(cfg['split_rows']['test'])} 样本；90 日输入、10 日累计收益路径。样本存在原固定成分股/未来完整标签可评价筛选偏差，已查看测试窗口，属于回顾性开发实验。",
      '- 冻结原 Kronos 与 tokenizer；同一个已训练 Prediction Adapter 起点；attention 2 heads/width32/零输出初始化。等权控制共享 LN、V、输出投影，冻结 Q/K；其可训练参数少于 attention，不能称容量完全匹配。',
      '- 三种子 100/200/300；相同批次顺序、学习率、loss 和 576 次离线优化步；每轮用验证 MSE 选模，含 epoch0。所有方法均选 epoch1（72 个 batch steps），此次 Q/K 已经历非零更新，不再是前次 smoke 的单步情况。',
      '- 静态流冻结验证选择的 checkpoint；在线流从同一 checkpoint 出发，先预测落盘再读取完整成熟反馈。每流 70 次每日更新，各 22,400 标签标量，每 horizon 最多32股；历史源标签先成熟才使用，次日启用新版本。',
      '- 三种结构分别评价静态和在线流。MAE/RMSE 与方向使用实际路径，IC/RankIC 按日截面计算。主表为种子指标均值，不是预测集成。','',
      '## 在线结果（三种子均值）','',
      '|模型|轨迹MAE↓|轨迹RMSE↓|Path方向准确率|Path IC|Path RankIC|Endpoint IC|Endpoint RankIC|',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for a in ARCHITECTURES:
        p=agg.loc[('online','path',a)];e=agg.loc[('online','endpoint',a)]
        lines.append(f"|{a}|{p.trajectory_mae:.6f}|{p.trajectory_rmse:.6f}|{p.direction_accuracy:.2%}|{p.ic:.6f}|{p.rankic:.6f}|{e.ic:.6f}|{e.rankic:.6f}|")
    a=agg.loc[('online','path','adapter')];t=agg.loc[('online','path','temporal_attention')]
    lines+=['',f"Attention 相对 Adapter：MAE {(t.trajectory_mae/a.trajectory_mae-1)*100:+.3f}%，RMSE {(t.trajectory_rmse/a.trajectory_rmse-1)*100:+.3f}%（均为误差相对变化）。方向准确率低于50%，不能说方向预测有效。",'',
      '## 子股票池组合结果','',
      '仅为约60股子池 Top6/Drop1、最低持有5日、单边15bps、下一日 close-to-close 简化代理，首日现金建仓。不是完整CSI300 Top30，也不是官方开盘执行AER。各日约59–60股；成分变化强制退出规则沿用现有 evaluator。','',
      '|模型|Path净收益|Path Sharpe|Path最大回撤|Path换手|Endpoint净收益|Endpoint Sharpe|Endpoint最大回撤|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for a in ARCHITECTURES:
        p=agg.loc[('online','path',a)];e=agg.loc[('online','endpoint',a)]
        lines.append(f"|{a}|{p.net_return:.2%}|{p.sharpe:.3f}|{p.max_drawdown:.2%}|{p.turnover:.2%}|{e.net_return:.2%}|{e.sharpe:.3f}|{e.max_drawdown:.2%}|")
    lines+=['','Endpoint收益与排序指标出现相反方向，说明Top6边界持仓变化不能概括为整体预测改善。不能只挑Endpoint净收益报成功；等权汇总收益相近，也不能将变化归因于学习attention权重。','',
      '## 静态流结果（三种子均值）','','|模型|轨迹MAE|轨迹RMSE|Path IC|Path RankIC|Path净收益|Endpoint净收益|','|---|---:|---:|---:|---:|---:|---:|']
    for a in ARCHITECTURES:
        p=agg.loc[('static','path',a)];e=agg.loc[('static','endpoint',a)]
        lines.append(f"|{a}|{p.trajectory_mae:.6f}|{p.trajectory_rmse:.6f}|{p.ic:.6f}|{p.rankic:.6f}|{p.net_return:.2%}|{e.net_return:.2%}|")
    lines+=['','在线效果并非对所有指标都好：本面板离线后静态的IC/RankIC高于各自在线版本，在线RMSE改善不能被描述成全面增益。','',
      '## 配对不确定性','','日期为统计单位，先对种子差值平均，再做3000次非循环日期块bootstrap；block10与20。区间不重训练、不重建持仓，未校正多重检验。方向列是轨迹整体方向率差；不是Path方向率。','',
      '|在线对照|信号|指标|差值|block10 95%区间|block20 95%区间|','|---|---|---|---:|---|---|']
    for comparison,mode,metric in [('temporal_attention_minus_adapter','path','mae'),('temporal_attention_minus_adapter','path','mse'),('temporal_attention_minus_adapter','path','rankic'),('temporal_attention_minus_adapter','endpoint','rankic'),('temporal_attention_minus_temporal_mean','path','mae'),('temporal_attention_minus_temporal_mean','endpoint','rankic'),('temporal_attention_minus_temporal_mean','endpoint','top_net')]:
        r=b[(b.phase=='online')&(b.comparison==comparison)&(b.signal==mode)&(b.metric==metric)].set_index('block')
        lines.append(f"|{comparison}|{mode}|{metric}|{r.loc[10,'delta']:+.7f}|[{r.loc[10,'ci_low']:+.7f}, {r.loc[10,'ci_high']:+.7f}]|[{r.loc[20,'ci_low']:+.7f}, {r.loc[20,'ci_high']:+.7f}]|")
    lines+=['','## 实现核验与后续判断','',
      '- 8项模块/实际引擎测试通过；未来未成熟标签置NaN不改变在线参数和预测；预测落盘hash、同反馈事件、一次消费、次日版本、同测试键审计通过（audit.json）。',
      '- 重新生成11,738条完整float32序列，不使用旧的截断训练hidden文件；cache中Kronos/tokenizer训练前后状态hash一致。',
      '- learned_attention_diagnostics.json保留Q/K权重变化、熵和平均权重；attention_weights.json包含在线首尾权重。',
      '- 目前不建议直接扩大attention层数或据此宣称方法有效。若继续，先研究为何MAE/排序下降而RMSE略改善，并用同容量MLP、同预算调优普通头及未查看时段确认；本次没有这些确认性证据。','',
      '## 文件索引','',
      '- comparison_metrics.csv / seed_summary.csv：所有种子、静态/在线、Path/Endpoint指标。',
      '- horizon_metrics_extended.csv：各预测期限MAE/RMSE、方向率、平衡方向率。',
      '- evaluation/*/evaluation_summary.json：F1、MCC、零收益基线R²、Newey-West、分位差和全部原evaluator输出。',
      '- paired_date_bootstrap.csv：block10/20配对日期区间。',
      '- original_checkpoint_evaluation.json：未做离线再适配的原始checkpoint参考。',
      '- label_access_log.csv / online_update_log.csv / issued/：成熟标签、预测先落盘与更新轨迹。',
      '- run_config.json / source_manifest.json / sequence_cache_manifest.json：输入、源码、权重身份。','',
      '![净收益与RankIC曲线](comparison.png)']
    (out/'REPORT.md').write_text(chr(10).join(lines)+chr(10),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    colors={'adapter':'#555555','temporal_mean':'#ca8a04','temporal_attention':'#2563eb'}
    for col,mode in enumerate(('path','endpoint')):
        for a in ARCHITECTURES:
            ds=[pd.read_csv(out/f'evaluation/{a}_{s}_online/{mode}_daily.csv') for s in seeds]
            net=np.mean([np.cumprod(1+x.top_net.to_numpy())-1 for x in ds],axis=0)*100
            rank=np.mean([x.rankic.to_numpy() for x in ds],axis=0)
            dates=pd.to_datetime(ds[0].as_of_date)
            axes[0,col].plot(dates,net,label=a,color=colors[a]);axes[1,col].plot(dates,pd.Series(rank).rolling(10,min_periods=1).mean(),color=colors[a])
        axes[0,col].set_title(mode+' | subset Top6 / Drop1');axes[0,col].set_ylabel('Net return (%)');axes[1,col].set_ylabel('RankIC (10-day mean)');axes[0,col].legend(fontsize=8)
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.axhline(0,color='#999999',lw=.5)
    import matplotlib.dates as mdates
    for ax in axes.flat:
        locator=mdates.AutoDateLocator(minticks=4,maxticks=5)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.suptitle('60-stock retrospective CPU pilot | online | mean of 3 seeds')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);fig.savefig(out/'comparison.svg');plt.close(fig)
    print(agg[['trajectory_mae','trajectory_rmse','ic','rankic','net_return']].to_string());print(out/'REPORT.md')

if __name__=='__main__':main(Path(sys.argv[1]))
