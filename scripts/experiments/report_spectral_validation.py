"""Independent validation and report for the spectral screen."""
import base64,hashlib,html,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/csi300_aligned/models/spectral_validation_20261028'
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
 cfg=json.loads((OUT/'locked_config.json').read_text());audit=json.loads((OUT/'audit.json').read_text());assert audit['status']=='passed'
 df=pd.read_csv(OUT/'evaluation_daily_horizon.csv');metrics=pd.read_csv(OUT/'metrics.csv');comp=pd.read_csv(OUT/'paired_comparisons.csv')
 assert len(metrics)==16 and len(df)==16*86*10 and not df.duplicated(['checkpoint','method','query_index','horizon']).any()
 recompute=df.groupby(['checkpoint','method'])[['mse','mae']].mean();saved=metrics.set_index(['checkpoint','method'])
 np.testing.assert_allclose(recompute.loc[saved.index].to_numpy(),saved[['mse','mae']].to_numpy(),atol=1e-15)
 windows=pd.read_csv(ROOT/'experiments/csi300_aligned/reference/date_windows.csv',dtype=str);windows=windows[windows.split.eq('test')].set_index('as_of_date')
 assert all(windows.loc[cfg['dates'][q],'label_end_date']<cfg['dates'][80] for q in cfg['fit_indices'])
 assert all(windows.loc[cfg['dates'][q],'label_end_date']<cfg['dates'][140] for q in cfg['selection_indices'])
 for ck in ['Prediction','RankIC']:
  events=pd.read_csv(ROOT/f'experiments/csi300_aligned/models/cfus_round1_20261022/{ck}_support_events.csv')
  assert len(events)==4672 and not events.local_row.duplicated().any()
  assert all(windows.loc[o,'label_end_date']==d and d<q for o,d,q in zip(events.support_origin,events.update_date,events.query_date))
  power=pd.read_csv(OUT/f'{ck}_frequency_daily.csv');tot=power.groupby(['method','query_index']).error_energy.sum()
  assert np.max(abs(tot-power.groupby(['method','query_index']).total_energy.first()))<1e-14
 D=np.load(OUT/'dct_basis.npy');cov=np.minimum.outer(np.arange(1,11),np.arange(1,11));energy=np.diag(D.T@cov@D)
 null=dict(description='independent mean-zero unit-variance additive increments; cumulative covariance min(i,j); explanatory reference, not market model or significance test',low2_pct=float(100*energy[:2].sum()/energy.sum()),low4_pct=float(100*energy[:4].sum()/energy.sum()),frequency_energy_share=(energy/energy.sum()).tolist())
 dump(OUT/'cumulative_noise_reference.json',null)
 drift=pd.read_csv(OUT/'Prediction_basis_drift.csv');gap=100*(drift.optimal2_coverage-drift.PCA2_covariance_coverage)
 drift_stats=dict(max_angle_deg=float(drift.max_subspace_angle_deg.max()),median_angle_deg=float(drift.max_subspace_angle_deg.median()),old_coverage_min_pct=float(100*drift.PCA2_covariance_coverage.min()),old_coverage_max_pct=float(100*drift.PCA2_covariance_coverage.max()),max_optimal_coverage_gain_pp=float(gap.max()),median_optimal_coverage_gain_pp=float(gap.median()))
 dump(OUT/'drift_summary.json',drift_stats)
 checks=dict(status='passed',metric_rows=16,evaluation_horizon_rows=len(df),evaluation_days=86,unique_update_opportunities=146,label_accesses_per_stream=46720,zero_step_has_no_effective_update=True,chronological_embargo=True,metric_recomputation=True,frequency_aggregation=True,support_maturity_verified_against_calendar=True,historical_reproduction=json.loads((OUT/'historical_reproduction.json').read_text())['status'])
 dump(OUT/'final_validation.json',checks)
 plt.rcParams.update({'font.size':10});fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
 for ck in ['Prediction','RankIC']:
  d=pd.read_csv(OUT/f'{ck}_frequency_daily.csv');d=d[(d.method=='HCA')&d.query_index.isin(cfg['evaluation_indices'])];p=d.groupby('frequency').error_energy.mean();axes[0].plot(np.arange(10),100*p/p.sum(),marker='o',label=ck+' HCA')
 axes[0].plot(np.arange(10),100*np.array(null['frequency_energy_share']),ls='--',color='black',label='Cumulative iid noise reference');axes[0].set(xlabel='DCT frequency index',ylabel='Squared-error energy (%)',title='Error spectrum across 10 horizons');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
 cc=comp[(comp.checkpoint=='RankIC')&(comp.baseline=='Full')&comp.method.isin(['DCT2','DCT4','SpectralShrink'])].set_index('method').loc[['DCT2','DCT4','SpectralShrink']]
 axes[1].errorbar(cc.mse_reduction_pct,np.arange(3),xerr=np.array([cc.mse_reduction_pct-cc.ci95_low,cc.ci95_high-cc.mse_reduction_pct]),fmt='o',capsize=4,color='#087e8b');axes[1].axvline(0,color='gray',ls='--');axes[1].set_yticks(np.arange(3),cc.index);axes[1].set(xlabel='MSE reduction vs tuned Full (%)',title='RankIC: paired block-10 95% intervals');axes[1].grid(axis='x',alpha=.2);fig.savefig(OUT/'spectral_results.png',dpi=170);plt.close(fig)
 conclusion='本轮没有验证出谱约束优于调优普通全维更新。低频修正在RankIC起点有效，但不足以证明独立方法优势；Prediction起点的主要候选均选择零步长。'
 setup='冻结Kronos、Adapter和原预测头，仅更新零初始化线性修正；采用普通SGD和半均方误差以隔离期限方向，不是完整HCA的联合训练/AdamW实验。每个非Frozen候选均有8个步长（包含0），只按选择期日期等权MSE选择。DCT2取频率0、1；DCT4取0—3；HighDCT2取8、9；SpectralShrink使用1/(1+(k/2)^2)作为固定频率更新权重，归一化均值为1；Full为普通全维线性修正。'
 timing='拟合/缩放段2025-07-16—2025-10-14（59日）；步长选择2025-10-29—2026-01-08（50日）；后续评估2026-01-23—2026-06-05（86日、25,210个股票起点）。前阶段查询标签在后阶段前完整成熟。连续流从选择期开始，跨不评分间隔保留状态，共146次更新机会，每次32股×10期限，共46,720个不重复标量标签；零步长没有有效参数更新。更新使用查询日前一日已完整成熟的支持路径；所有方法使用相同监督事件。'
 findings=[
 '当前完整HCA已发布预测的前2个DCT频段包含88.47%（Prediction）/88.73%（RankIC）误差能量，前4个约95.32%/95.38%；前后43日都集中。误差能量包含均值偏置，未中心化。该诊断不用于选方法。',
 f'独立单日零均值噪声累加的理论参考，也有前2频段{null["low2_pct"]:.2f}%、前4频段{null["low4_pct"]:.2f}%能量。低频集中可能主要来自累计目标的积分结构，不能推出低频误差可预测或动态门控有效。参考是加性随机游走解释，不是实际复利收益模型或显著性检验。',
 'RankIC中DCT2/DCT4相对PCA降低MSE约1.64%/1.68%，前后半段同向，MAE改善；但相对调优Full分别恶化约0.0508%/0.0189%。差异很小，描述性95%区间仍全部低于零，不能声称超过Full。',
 '软谱收缩较RankIC PCA降低MSE约1.56%，但较Full恶化约0.1348%。DCT2、DCT4、软收缩与Full的单次更新有害日比例均为32.56%，没有显示减少有害更新。该指标隔离当天新一步对下一查询的影响，不代表长期总效用。',
 'Prediction的PCA、DCT2、DCT4、软收缩、Full均在选择期取零步长，没有产生增益；不能把同样结果描述成鲁棒。高频负对照只有极小变化，单个随机基也没有跨起点优势。',
 f'每5日观察一次40个成熟起点的日期等权协方差：旧PCA覆盖近期方差{drift_stats["old_coverage_min_pct"]:.2f}%—{drift_stats["old_coverage_max_pct"]:.2f}%；新旧二维空间最大夹角中位数{drift_stats["median_angle_deg"]:.2f}°、最大{drift_stats["max_angle_deg"]:.2f}°；重估最优二维基最多增加{drift_stats["max_optimal_coverage_gain_pp"]:.3f}个百分点覆盖率。当前窗口未显示严重的目标主子空间失配；这不衡量更新效用，也不是动态B效果实验。',
 '完整DCT与单位矩阵全维SGD更新数值等价，证明只做正交频率坐标变换不会自动增加表达能力或带来优势。']
 limitations='已反复查看的单市场、单随机种子历史窗口，只能作探索性机制筛选；固定股票面板有潜在成分前视/幸存者偏差。配对循环日期块bootstrap块长10、4,000次，重采样已实现预测损失，不重训、未校正多重比较。秩4、全维和软收缩容量不同；没有测试动态频段门控、在线换基、部分期限反馈、日历周期识别，也没有新增交易收益回测。'
 action='不把PCA换DCT或动态谱权重作为已验证创新，暂不扩展动态门控。保留DCT2/DCT4作为可解释强基线。若继续，先在新校准流检验低频残差能否由股票特征预测，并证明频段效用存在可重复、可提前识别的变化；相同反馈预算下超过调优Full后，再接回完整HCA。'
 table=metrics.pivot(index='method',columns='checkpoint',values='mse').loc[cfg['methods']]
 main_table='| 方法 | Prediction MSE | RankIC MSE |\n|---|---:|---:|\n'+''.join(f'| {m} | {table.loc[m,"Prediction"]:.10f} | {table.loc[m,"RankIC"]:.10f} |\n' for m in table.index)
 compare_table='| RankIC方法 | 相对PCA改善 | 相对Full改善 | 对Full的95%区间 |\n|---|---:|---:|---:|\n'
 for m in ['DCT2','DCT4','SpectralShrink']:
  a=comp[(comp.checkpoint=='RankIC')&(comp.method==m)&(comp.baseline=='TargetPCA2')].iloc[0];b=cc.loc[m];compare_table+=f'| {m} | {a.mse_reduction_pct:+.4f}% | {b.mse_reduction_pct:+.4f}% | [{b.ci95_low:+.4f}%, {b.ci95_high:+.4f}%] |\n'
 validation='历史5方法×2起点MSE复现误差小于1e-12；支持样本梯度与autograd一致；完整DCT与全维更新一致；未来标签置毒不改变训练；成熟时序、监督去重、指标聚合通过独立核验。主体运行约16.2秒CPU，不含原Kronos提取、bootstrap和报告。'
 reproduction='复现：仓库根目录运行 .runtime/doubleadapt_env/Scripts/python.exe scripts/experiments/run_spectral_validation.py prepare，然后 run；新运行须修改OUT，prepare拒绝覆盖。报告与独立校验：运行 scripts/experiments/report_spectral_validation.py。'
 report='# HCA谱分析验证实验\n\n'+conclusion+'\n\n## 方案与时序\n\n'+setup+'\n\n'+timing+'\n\n## 核心结果\n\n'+main_table+'\n'+compare_table+'\n正值为MSE相对降低比例，负值为恶化，不是收益百分点。\n\n## 机制诊断\n\n'+'\n\n'.join(f'{i+1}. {x}' for i,x in enumerate(findings))+'\n\n## 正确性与限制\n\n'+validation+'\n\n'+limitations+'\n\n## 决策\n\n'+action+'\n\n'+reproduction+'\n'
 (OUT/'REPORT.md').write_text(report,encoding='utf-8')
 def t(d):return d.to_html(index=False,border=0,float_format=lambda x:f'{x:.7f}')
 pic=base64.b64encode((OUT/'spectral_results.png').read_bytes()).decode();links=['REPORT.md','locked_config.json','metrics.csv','paired_comparisons.csv','frequency_summary.csv','cumulative_noise_reference.json','drift_summary.json','final_validation.json','audit.json']
 doc='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HCA谱分析验证</title><style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:1150px;margin:35px auto;padding:0 24px;line-height:1.85;color:#183448;background:#f5f8fa}h1,h2{color:#086d77}section{background:white;padding:22px 28px;border:1px solid #dce6ed;border-radius:10px;margin:18px 0}.lead{background:#e6f2f1;border-left:5px solid #087e8b;padding:18px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%;white-space:nowrap}th,td{padding:9px;border-bottom:1px solid #dce5ec;text-align:right}th{background:#eef4f7}img{width:100%;height:auto}a{color:#087e8b}li{margin:12px 0}.warn{background:#fff3df;padding:15px}@media print{body{background:white;margin:0}section{padding:10px;border:0}.scroll{overflow:visible}table{white-space:normal;font-size:8pt}tr{break-inside:avoid}}</style></head><body><h1>HCA：谱分析验证实验</h1><p>2026-10-28 · 2种初始化 × 8种方法 · 同监督预算 · 后段86日评估</p><p class="lead"><b>'+conclusion+'</b></p><section><h2>1. 实验方案</h2><p>'+setup+'</p><p>'+timing+'</p></section><section><h2>2. 频谱与效果</h2><img alt="误差频谱及RankIC相对Full的配对区间" src="data:image/png;base64,'+pic+'"><p class="warn">低频误差集中不等于可预测，也不等于谱更新优于普通更新。</p><div class="scroll">'+t(metrics[['checkpoint','method','mse','mae','alpha']])+'</div><h3>RankIC相对Full</h3><div class="scroll">'+t(cc.reset_index()[['method','mse_reduction_pct','ci95_low','ci95_high','mae_reduction_pct']])+'</div><p>正值表示误差降低比例，负值表示恶化。</p></section><section><h2>3. 机制诊断</h2><ol>'+''.join('<li>'+html.escape(x)+'</li>' for x in findings)+'</ol></section><section><h2>4. 当前决策与证据边界</h2><p>'+action+'</p><p>'+limitations+'</p></section><section><h2>5. 正确性与复核</h2><p>'+validation+'</p><p>'+' · '.join('<a href="'+x+'">'+x+'</a>' for x in links)+'</p></section></body></html>'
 (OUT/'spectral_validation.html').write_text(doc,encoding='utf-8')
 manifest={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in OUT.iterdir() if x.is_file() and x.suffix in ['.csv','.json','.md','.html','.png','.npy'] and x.name!='artifact_manifest.json'};dump(OUT/'artifact_manifest.json',manifest)
 print(json.dumps({'status':'passed','report':str(OUT/'spectral_validation.html'),'validation':checks},ensure_ascii=False))
if __name__=='__main__':main()
