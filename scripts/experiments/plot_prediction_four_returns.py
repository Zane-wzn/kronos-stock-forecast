"""Plot four saved Prediction strategies; no model training or portfolio replay."""
from pathlib import Path
import json
import hashlib
import html
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[2]
FULL=ROOT/'experiments/csi300_aligned/models/prediction_dct_full_20261102'
STATIC=ROOT/'experiments/csi300_aligned/evaluations/prediction_progression_20261102'
OUT=FULL/'four_mode_return_curves'
MODES=[('Static','静态 Prediction Adapter','#697586'),
       ('A','在线更新 · 无残差','#326dcc'),
       ('PCA','在线更新 · PCA 残差','#087e8b'),
       ('DCT','在线更新 · DCT 残差','#d87b28')]

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--with-index',action='store_true')
    args=parser.parse_args()
    source_dir=OUT
    if args.with_index:OUT=source_dir/'with_hs300'
    OUT.mkdir(exist_ok=True)
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none'})
    metrics=pd.read_csv(FULL/'metrics.csv')
    prior=pd.read_csv(STATIC/'metrics.csv')
    fig,axes=plt.subplots(2,1,figsize=(13.8,9.8),sharex=True)
    csv=[];summary=[];sources=[];calendar=None
    for ax,signal,title in zip(axes,['path','endpoint'],['Path：10个期限预测的平均值选股','Endpoint：第10日期限预测选股']):
        for key,label,color in MODES:
            path=STATIC/f'Prediction_static_{signal}_daily.csv' if key=='Static' else FULL/f'{key}_{signal}_daily.csv'
            d=pd.read_csv(path).sort_values('as_of_date').reset_index(drop=True)
            assert len(d)==226 and not d.as_of_date.duplicated().any()
            if calendar is None:calendar=d.as_of_date.tolist()
            assert calendar==d.as_of_date.tolist()
            np.testing.assert_allclose(d.top_net,d.top_raw-.0015*d.turnover,rtol=0,atol=1e-14)
            curve=np.cumprod(1+d.top_net.to_numpy())-1
            expected=(prior[(prior.stage=='Prediction_static')&(prior.signal==signal)] if key=='Static'
                      else metrics[(metrics.stage=='Prediction_'+key)&(metrics.signal==signal)]).iloc[0].net_cumulative_return
            assert abs(curve[-1]-expected)<1e-12
            dates=pd.to_datetime(d.as_of_date)
            ax.plot(dates,curve,color=color,lw=2.0 if key in ['PCA','DCT'] else 1.7,
                    linestyle='--' if key=='Static' else '-',label=f'{label}  {curve[-1]:.3%}')
            summary.append(dict(signal=signal,mode=key,label=label,net_cumulative_return=float(curve[-1])))
            csv.extend(dict(signal=signal,mode=key,as_of_date=date,net_daily_return=float(r),net_cumulative_return=float(c))
                       for date,r,c in zip(d.as_of_date,d.top_net,curve))
            sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if args.with_index:
            # Match each strategy's close(t)->close(next trading day) payoff.
            raw=json.loads((source_dir/'sina_hs300_raw.json').read_text(encoding='utf-8'))
            index=pd.DataFrame(raw).rename(columns={'day':'date'})
            index['close']=index.close.astype(float)
            check=pd.DataFrame(json.loads((source_dir/'tencent_hs300_raw.json').read_text(encoding='utf-8'))['data']['sh000300']['day'],
                columns=['date','open','close','high','low','volume'])
            check['close']=check.close.astype(float)
            cross=index[['date','close']].merge(check[['date','close']],on='date',suffixes=('_sina','_tencent'),validate='one_to_one')
            assert len(cross)==229
            max_error=float(abs(cross.close_sina-cross.close_tencent).max())
            assert max_error<=.00501
            index=index.sort_values('date').reset_index(drop=True)
            index['return_end_date']=index.date.shift(-1)
            index['next_close']=index.close.shift(-1)
            lookup=index.set_index('date')
            benchmark=lookup.loc[calendar,['close','next_close','return_end_date']].copy()
            benchmark.index.name='as_of_date'
            assert not benchmark.isna().any().any()
            windows=pd.read_csv(ROOT/'experiments/csi300_aligned/reference/date_windows.csv',dtype=str)
            windows=windows[windows.split.eq('test')]
            assert (windows.groupby('as_of_date').label_start_date.nunique()==1).all()
            release=windows.drop_duplicates('as_of_date').set_index('as_of_date').loc[calendar,'label_start_date']
            assert benchmark.return_end_date.tolist()==release.tolist()
            benchmark['index_daily_return']=benchmark.next_close/benchmark.close-1
            benchmark['index_cumulative_return']=(1+benchmark.index_daily_return).cumprod()-1
            expected=benchmark.next_close.iloc[-1]/benchmark.close.iloc[0]-1
            assert abs(benchmark.index_cumulative_return.iloc[-1]-expected)<1e-12
            benchmark.to_csv(OUT/'hs300_aligned_returns.csv')
            ax.plot(pd.to_datetime(calendar),benchmark.index_cumulative_return,color='#b74453',lw=2.2,
                linestyle='-.',label=f'沪深300价格指数  {expected:.3%}')
            summary.append(dict(signal=signal,mode='HS300',label='沪深300价格指数（未扣成本）',
                net_cumulative_return=None,index_cumulative_return=float(expected)))
            index_validation=dict(code='000300',source='Sina Finance daily historical index close, Tencent cross-check',
                base_date=calendar[0],base_close=float(benchmark.close.iloc[0]),
                end_date=str(benchmark.return_end_date.iloc[-1]),end_close=float(benchmark.next_close.iloc[-1]),
                cumulative_return=float(expected),dates=226,cross_check_max_abs_points=max_error,
                date_alignment='Each plotted forecast date t carries index close(next trading day)/close(t)-1, same return interval as strategy actual_return_1',
                dividend_and_cost='Price index; excludes dividend reinvestment, no fees. Strategy lines are after costs.')
            (OUT/'index_validation.json').write_text(json.dumps(index_validation,ensure_ascii=False,indent=2),encoding='utf-8')
            for filename in ['sina_hs300_raw.json','tencent_hs300_raw.json','hs300_sources.json']:
                path=source_dir/filename
                sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        ax.set_title(title,loc='left',fontsize=14,fontweight='bold',pad=13)
        ax.axhline(0,color='#8a96a3',lw=.8)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1,decimals=0))
        ax.set_ylabel('累计收益率' if args.with_index else '累计净收益率')
        ax.grid(alpha=.18)
        ax.legend(loc='upper left',ncol=2,fontsize=10,framealpha=.95)
        ax.margins(x=.015)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[-1].set_xlabel('预测起点日期（对应下一交易日收益）')
    fig.suptitle('Prediction：四种模式与沪深300指数' if args.with_index else 'Prediction 主线：四种模式累计净收益曲线',fontsize=19,fontweight='bold',y=.97)
    fig.text(.5,.925,'2025-07-01—2026-06-05 · 同一226日 · Top30 / Drop3 / 最少持有5日 · 单向成本15bps',ha='center',fontsize=10,color='#536475')
    footer='策略已扣成本；沪深300为价格指数涨跌幅（未扣成本、不含分红再投资）。均按相同下一交易日收益区间对齐。' if args.with_index else '累计净收益 = ∏(1 + 每日扣成本收益) − 1；图例为期末值。回顾性固定面板，单市场、单种子。'
    fig.text(.08,.025,footer,fontsize=10,color='#536475')
    fig.subplots_adjust(left=.085,right=.97,top=.875,bottom=.1,hspace=.28)
    for ext in ['png','svg']:
        fig.savefig(OUT/f'cumulative_net_returns.{ext}',dpi=180,facecolor='white')
    plt.close(fig)
    pd.DataFrame(csv).to_csv(OUT/'daily_return_curves.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'final_returns.csv',index=False)
    # Inline SVG keeps the viewing report fully self-contained and zoomable.
    svg=(OUT/'cumulative_net_returns.svg').read_text(encoding='utf-8')
    svg=svg[svg.index('<svg'):]
    index_note=('沪深300（000300）按相同的下一交易日收盘收益对齐，基准为2025-07-01收盘，最后兑现到2026-06-08收盘。指数为价格涨跌幅，不含分红再投资、未扣交易成本；不以股票池等权收益替代指数。<a href="hs300_aligned_returns.csv">指数逐日对齐数据</a> · <a href="index_validation.json">来源及核验</a>。' if args.with_index else '')
    comparison='../../comparison.html' if args.with_index else '../comparison.html'
    doc='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Prediction 四种模式累计净收益</title><style>body{max-width:1400px;margin:20px auto;padding:0 15px;font-family:Microsoft YaHei,Segoe UI,sans-serif;color:#1e3040}svg{width:100%;height:auto}p{line-height:1.8;color:#536475}a{color:#087e8b}@media print{body{margin:0}}</style></head><body>'+svg+'<p>四条策略曲线来自已保存的逐日回测，本次只绘图，未训练模型或重选参数。净值按预测起点日期标记，其日收益采用下一交易日实际收益。Path 与 Endpoint 使用不同选股信号，各自独立持仓；不表示持有10日收益的简单累加。</p><p>'+index_note+'</p><p><a href="cumulative_net_returns.png">PNG图片</a> · <a href="cumulative_net_returns.svg">SVG矢量图</a> · <a href="daily_return_curves.csv">策略逐日数据</a> · <a href="'+comparison+'">完整指标比较</a></p></body></html>'
    (OUT/'curves.html').write_text(doc,encoding='utf-8')
    (OUT/'validation.json').write_text(json.dumps(dict(status='passed',dates=226,curves=10 if args.with_index else 8,
        cost_identity_verified=True,final_values_reproduce_metrics=True,sources=sources),ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))
    print(OUT/'cumulative_net_returns.png')

if __name__=='__main__':main()
