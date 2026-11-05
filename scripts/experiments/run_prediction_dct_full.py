"""Prediction-only full HCA comparison: reproduce A/PCA, then replace B by DCT.

No learning-rate selection; retain the historical complete-feedback protocol.
"""
from __future__ import annotations
import copy, json, shutil, sys, time, hashlib, html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from scripts.experiments import run_hca_gain_attribution as runner
from evaluate.metrics import EvaluationConfig,evaluate_frame,clean_json
from src.doubleadapt.compare import return_stats,indices

OUT=ROOT/'experiments/csi300_aligned/models/prediction_dct_full_20261102'
HISTORY=runner.DEFAULT_OUT
PROGRESSION=ROOT/'experiments/csi300_aligned/evaluations/prediction_progression_20261102'

def save(path,obj):
    path.write_text(json.dumps(clean_json(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def train():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    config=dict(checkpoint='Prediction',variants=['A','PCA','DCT'],seed=100,rank=2,
        adapter_lr=1e-5,head_and_coeff_lr=1e-4,optimizer='AdamW wd0',loss='SmoothL1 beta1',
        clip_norm=1,dropout=False,updates=216,labels=69120,dates=226,rows=66103,
        dct='orthonormal DCT-II, frequencies k=0,1 over the 10 horizon positions; fixed, zero coefficient initialization',
        tuning='none; inherited historical hyperparameters; no selection by evaluation',
        evaluation='Top30 Drop3 minhold5 cost15bps annual252; close-to-close proxy; identical labels/keys',
        limitations='Previously inspected single-market/seed fixed panel; retrospective, possible survivor/constituent bias; no independent confirmation')
    save(OUT/'locked_config.json',config)
    shutil.copy2(__file__,OUT/'runner_snapshot.py')
    runner.old.seed_all(100);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    entry=runner.full.MODELS['Prediction']
    assert runner.old.sha256(entry['checkpoint'])==entry['checkpoint_sha256']
    bundle=runner.load_inputs_for_checkpoint(entry)
    pca,info=runner.hca.training_basis(bundle[0],bundle[3],rank=2)
    np.testing.assert_array_equal(pca,np.load(HISTORY/'Prediction_basis.npy'))
    dct=np.cos(np.pi*(np.arange(10)[:,None]+.5)*np.arange(2)[None,:]/10)
    dct=(dct/np.linalg.norm(dct,axis=0)).astype(np.float32)
    np.testing.assert_allclose(dct.T@dct,np.eye(2),atol=2e-7)
    np.save(OUT/'PCA_basis.npy',pca);np.save(OUT/'DCT_basis.npy',dct)
    reference=pd.read_csv(entry['reference'],dtype={'stock':str}).sort_values(runner.KEYS).reset_index(drop=True)
    save(OUT/'inputs.json',dict(checkpoint_sha256=runner.old.sha256(entry['checkpoint']),
        reference_sha256=runner.old.sha256(entry['reference']),hidden_sha256=bundle[11],pca_fit=info,
        engine_sha256=runner.old.sha256(Path(runner.__file__)),runner_sha256=runner.old.sha256(Path(__file__))))
    audits=[];start=time.perf_counter()
    for kind,basis in [('A',pca),('PCA',pca),('DCT',dct)]:
        variant='A' if kind=='A' else 'HCA'
        name='Prediction_Live_'+kind
        initial=copy.deepcopy(bundle[-1]) if kind=='A' else runner.hca.with_hca(bundle[-1],basis)
        model=copy.deepcopy(initial)
        # Zero initialization must preserve all forecasts prior to online training.
        with torch.no_grad():
            for date in bundle[4]:
                torch.testing.assert_close(initial(bundle[7][date]),bundle[-1](bundle[7][date]),rtol=0,atol=0)
        print('TRAINING '+name,flush=True)
        result=runner.run_stream('A',model,bundle[7],bundle[8],bundle[6],bundle[9],
            runner.pilot.ReleasedLabels(bundle[3],bundle[10]),OUT/name,
            head_lr=1e-4,adapter_lr=1e-5,variant=variant,basis=basis)
        history=HISTORY/('Prediction_Live_A' if kind=='A' else 'Prediction_Live_HCA')/'causal_predictions.csv' if kind!='DCT' else None
        audit=runner.audit_attribution(OUT,name,result,initial,model,'Live',variant,basis,bundle[4],reference,history)
        # Assert exactly the same supervision, even across distinct basis choices.
        cols=['update_date','row_id','horizon','origin_date','label_date','complete_date']
        if kind=='A':events=result[1][cols].reset_index(drop=True)
        else:pd.testing.assert_frame_equal(events,result[1][cols].reset_index(drop=True))
        runner.save_stream(OUT,name,result,None)
        runner.old.atomic_csv(OUT/name/'predictions.csv',runner.full.attach_targets(result[0],reference,name))
        audits.append(audit);save(OUT/'training_audits.json',audits)
        print('TRAINED '+json.dumps(audit),flush=True)
    save(OUT/'training_complete.json',dict(status='passed',seconds=time.perf_counter()-start,audits=audits))

def report():
    config=EvaluationConfig(top_k=30,drop_n=3,min_holding_days=5,cost_rate=.0015,annualization_days=252,bootstrap_reps=0)
    rows=[];daily={};direction_daily={};frames={}
    for kind in ['A','PCA','DCT']:
        name='Prediction_Live_'+kind
        f=pd.read_csv(OUT/name/'predictions.csv',dtype={'stock':str}).sort_values(runner.KEYS).reset_index(drop=True)
        result=evaluate_frame(f,config);save(OUT/(kind+'_evaluation.json'),result);frames[kind]=f
        if kind=='A':ref=f
        else:
            pd.testing.assert_frame_equal(f[runner.KEYS+runner.ACOLS],ref[runner.KEYS+runner.ACOLS])
        for mode in ['path','endpoint']:
            m=result['metrics'][mode];p=pd.DataFrame(result['daily'][mode]['portfolio']);cross=pd.DataFrame(result['daily'][mode]['cross_section'])
            q=p.merge(cross,on='as_of_date',validate='one_to_one');daily[kind,mode]=q;q.to_csv(OUT/(kind+'_'+mode+'_daily.csv'),index=False)
            rows.append(dict(stage='Prediction_'+kind,signal=mode,rows=len(f),trajectory_mae=result['metrics']['trajectory_pooled']['mae'],trajectory_rmse=result['metrics']['trajectory_pooled']['rmse'],
                mae=m['point']['mae'],rmse=m['point']['rmse'],direction_accuracy=m['direction']['accuracy'],balanced_accuracy=m['direction']['balanced_accuracy'],mcc=m['direction']['mcc'],
                ic=m['cross_section']['ic']['mean'],rankic=m['cross_section']['rankic']['mean'],turnover=m['portfolio']['turnover'],**{k:float(v) for k,v in return_stats(p.top_net.to_numpy()).items()}))
            pred=f[runner.PCOLS].mean(1) if mode=='path' else f[runner.PCOLS[-1]]
            actual=f[runner.ACOLS].mean(1) if mode=='path' else f[runner.ACOLS[-1]]
            count=pd.DataFrame({'date':f.as_of_date,'correct':(pred>0)==(actual>0)})
            direction_daily[kind,mode]=count.groupby('date').correct.agg(['sum','count'])
    table=pd.DataFrame(rows);table.to_csv(OUT/'metrics.csv',index=False)
    # Reproduced baselines must also reproduce previously published metrics.
    hist=pd.read_csv(HISTORY/'attribution_metrics.csv')
    for kind,hname in [('A','Prediction_Live_A'),('PCA','Prediction_Live_HCA')]:
        for mode in ['path','endpoint']:
            a=table[(table.stage=='Prediction_'+kind)&(table.signal==mode)].iloc[0]
            b=hist[(hist.name==hname)&(hist.signal==mode)].iloc[0]
            np.testing.assert_allclose([a.trajectory_rmse,a.ic,a.net_cumulative_return],[b.trajectory_rmse,b.ic,b.net_cumulative_return],rtol=0,atol=1e-12)
    ci=[];draws=indices(226,5000,10,20261013)
    for left,right in [('PCA','A'),('DCT','A'),('DCT','PCA')]:
        for mode in ['path','endpoint']:
            la,rb=daily[left,mode],daily[right,mode]
            for metric in ['net_cumulative_return','ic','rankic','direction_accuracy']:
                if metric=='net_cumulative_return':
                    a,b=la.top_net.to_numpy(),rb.top_net.to_numpy();delta=return_stats(a)[metric]-return_stats(b)[metric];sample=return_stats(a[draws])[metric]-return_stats(b[draws])[metric]
                elif metric=='direction_accuracy':
                    a,b=direction_daily[left,mode],direction_daily[right,mode];delta=a['sum'].sum()/a['count'].sum()-b['sum'].sum()/b['count'].sum()
                    sample=a['sum'].to_numpy()[draws].sum(1)/a['count'].to_numpy()[draws].sum(1)-b['sum'].to_numpy()[draws].sum(1)/b['count'].to_numpy()[draws].sum(1)
                else:
                    d=la[metric].to_numpy()-rb[metric].to_numpy();delta=d.mean();sample=d[draws].mean(1)
                lo,hi=np.quantile(sample,[.025,.975]);ci.append(dict(comparison=left+'_minus_'+right,signal=mode,metric=metric,delta=float(delta),ci_low=float(lo),ci_high=float(hi)))
    pd.DataFrame(ci).to_csv(OUT/'paired_intervals.csv',index=False)
    progression=pd.read_csv(PROGRESSION/'metrics.csv');progression=progression[progression.stage.isin(['Kronos_zero_shot','Prediction_static'])]
    combined=pd.concat([progression,table],ignore_index=True);combined.to_csv(OUT/'progression_metrics.csv',index=False)
    labels={'Kronos_zero_shot':'原 Kronos 零样本预测','Prediction_static':'① 静态 Prediction Adapter','Prediction_A':'② 加入在线更新（无残差）','Prediction_PCA':'③ 加入 PCA 残差','Prediction_DCT':'③ 替换为 DCT 残差'}
    lines=['# Prediction：PCA替换为DCT的完整在线对照','',
        '仅替换固定期限基，PCA与DCT均为10×2；Adapter、原头和系数头联合更新。零初始化、同checkpoint、同216次成熟反馈/69120标量标签、相同AdamW/学习率/裁剪；所有预测先保存再访问标签。未根据本窗口选择超参。','',
        '|阶段|轨迹RMSE ↓|Path方向准确率|第10日方向准确率|Path组合净收益|Endpoint组合净收益|',
        '|---|---:|---:|---:|---:|---:|']
    for stage,label in labels.items():
        a=combined[(combined.stage==stage)&(combined.signal=='path')].iloc[0];b=combined[(combined.stage==stage)&(combined.signal=='endpoint')].iloc[0]
        lines.append(f'|{label}|{a.trajectory_rmse:.6f}|{a.direction_accuracy:.3%}|{b.direction_accuracy:.3%}|{a.net_cumulative_return:.3%}|{b.net_cumulative_return:.3%}|')
    lines+=['','误差为累计收益尺度，不是价格元。Path为10期限累计收益预测均值；Endpoint为第10日累计收益。Top30/Drop3/最小持有5日/单向15bps，actual_return_1兑现日收益，252日年化，现金起始。方向return>0为正，零作负。','',
        '完整指标见metrics.csv；逐日回测、方向统计、IC/RankIC来自相同公共evaluate。A/PCA与历史预测复现误差≤1e-7、指标复现≤1e-12；逐键监督一致、标签成熟、重复访问、先保存预测与版本顺序均核验。','',
        '区间：配对非循环日期块bootstrap，块10、5000次、seed20261013；重采样已实现收益，不重训或重模拟持仓；未多重比较校正。单市场单种子、旧窗口已多次查看、固定成分潜在偏差；仅探索性比较。DCT固定前2频段，结论不涵盖其他秩/选频方案。','',
        '运行：python scripts/experiments/run_prediction_dct_full.py；新运行须换OUT（拒绝覆盖）。只重建报告可加report参数。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    summary_table=''
    for stage,label in labels.items():
        a=combined[(combined.stage==stage)&(combined.signal=='path')].iloc[0];b=combined[(combined.stage==stage)&(combined.signal=='endpoint')].iloc[0]
        summary_table+=f'<tr><td>{label}</td><td>{a.trajectory_rmse:.6f}</td><td>{a.direction_accuracy:.3%}</td><td>{b.direction_accuracy:.3%}</td><td>{a.net_cumulative_return:.3%}</td><td>{b.net_cumulative_return:.3%}</td></tr>'
    doc='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Prediction PCA与DCT完整对照</title><style>body{max-width:1250px;margin:38px auto;padding:0 24px;font-family:Segoe UI,Microsoft YaHei,sans-serif;color:#1e3040;line-height:1.8}h1,h2{color:#086d77}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}td,th{padding:15px 12px;border-bottom:1px solid #dfe5eb;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left;white-space:normal;min-width:190px}th{background:#f2f6f8}tbody tr:last-child{background:#eaf6f3;font-weight:700}.note{padding:18px;background:#f1f6fa;border-left:4px solid #087e8b}a{color:#087e8b}@media print{body{margin:0}table{font-size:9pt}td,th{padding:8px;white-space:normal}}</style></head><body><h1>Prediction：PCA 换成 DCT 后的指标比较</h1><p>2026-11-02 · 同一226日、66,103个预测样本 · 完整在线联合更新</p><div class="scroll"><table><thead><tr><th>阶段</th><th>轨迹 RMSE ↓</th><th>Path 方向准确率</th><th>第10日方向准确率</th><th>Path 组合净收益</th><th>Endpoint 组合净收益</th></tr></thead><tbody>'+summary_table+'</tbody></table></div><p class="note">仅替换期限基：PCA与DCT都是10×2，零初始化系数头；Adapter与两个头联合更新。DCT为固定前2个余弦频段。相同学习率、AdamW、成熟反馈预算与组合规则，无本窗口调参。</p><h2>完整指标</h2><div class="scroll">'+table.to_html(index=False,border=0,float_format=lambda x:f'{x:.7f}')+'</div><h2>配对差值与95%区间</h2><p>收益及方向准确率差值为小数，乘100转为百分点；IC差无量纲。</p><div class="scroll">'+pd.DataFrame(ci).to_html(index=False,border=0,float_format=lambda x:f'{x:.7f}')+'</div><p>点位误差为累计收益，非价格元；方向为收益正负。回测Top30/Drop3/至少5日/15bps，下一日收盘收益代理。旧窗口单市场单种子，区间不含重训、持仓重演或多重比较校正；结果不构成独立确认。</p><p><a href="REPORT.md">说明</a> · <a href="metrics.csv">完整指标</a> · <a href="paired_intervals.csv">配对区间</a> · <a href="training_audits.json">训练核验</a></p></body></html>'
    (OUT/'comparison.html').write_text(doc,encoding='utf-8')
    save(OUT/'final_validation.json',dict(status='passed',stages=3,rows_per_stage=66103,dates=226,same_supervision=True,historical_A_PCA_reproduced=True,metric_rows=len(table)))
    print(table.to_string(index=False),flush=True);print(pd.DataFrame(ci).to_string(index=False),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='report':report()
    else:train();report()
