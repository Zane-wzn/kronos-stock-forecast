"""Generate LaTeX comparison tables and the paired-return figure."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ..paths import OUTPUT as HERE

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', type=Path)
OUT = parser.parse_args().output
(OUT / 'tables').mkdir(parents=True, exist_ok=True)
(OUT / 'figures/revised_curves').mkdir(parents=True, exist_ok=True)
markets = ['csi300', 'nasdaq100', 'sp500']
titles = dict(zip(markets, ['CSI300', 'Nasdaq-100', r'S\&P 500']))
models = ['Static Adapter','Online Head Only','Full Model']
peer = ['Target only','Unprojected peers','Fixed PCA basis','Online PCA']
online = pd.read_csv(HERE/'table4_online_updates.csv').set_index(['market','model'])
peers = pd.read_csv(HERE/'table5_peer_correction.csv').set_index(['market','model'])
uci = pd.read_csv(HERE/'tables7_8_update_intervals.csv')
pci = pd.read_csv(HERE/'table5_paired_intervals.csv').query('metric == "annualized_return_pp"')
comparison_paths = [HERE / market / 'adjusted_comparisons.csv' for market in markets]
archived_comparison = HERE / 'paper_comparisons_by_seed.csv'
if all(path.is_file() for path in comparison_paths):
    comparisons = pd.concat([pd.read_csv(path) for path in comparison_paths], ignore_index=True)
else:
    comparisons = pd.read_csv(archived_comparison)
data = comparisons.groupby(['task','market','model']).mean(numeric_only=True)


def row(fields):
    return ' & '.join(map(str,fields)) + r' \\'


def table(name,key,caption,columns,header,body,padding=2):
    lines = [r'\begin{table}[!htbp]',rf'\caption{{{caption}}}\label{{{key}}}',
             r'\centering\small',rf'\setlength{{\tabcolsep}}{{{padding}pt}}',
             rf'\begin{{tabular*}}{{\linewidth}}{{@{{\extracolsep{{\fill}}}}{columns}@{{}}}}',
             r'\toprule',*header,r'\midrule',*body,r'\bottomrule',r'\end{tabular*}',r'\end{table}']
    (OUT/'tables'/f'{name}.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def note(name, content):
    (OUT/'tables'/f'{name}.tex').write_text(content+'\n', encoding='utf-8')


def interval(r, decimals=2):
    threshold = 1e-10
    lo = 0 if abs(r.low) < threshold else r.low
    return f'[{lo:.{decimals}f}, {r.high:.{decimals}f}]'


def cell(r, decimals=2):
    return f'{r.delta:+.{decimals}f} {interval(r, decimals)}'


def urow(market, reference, block=10, metric='annualized_return_pp', treatment='Full Model'):
    rows = uci.loc[(uci.market==market)&(uci.control==reference)&(uci.block_days==block)&
                   (uci.metric==metric)&(uci.treatment==treatment)]
    assert len(rows)==1
    return rows.iloc[0]


def span(labels,width):
    return row(['',*[rf'\multicolumn{{{width}}}{{c}}{{{titles[m]}}}' for m in labels]])


comparison_markets = ['csi300','nasdaq100']
names = ['Kronos (zero-shot)','Chronos-2','TimesFM 3.0','DoubleAdapt (GRU)','SSPT (ind-SCC)','MASTER (OHLCV)','Full Model']
assert [int(data.loc['external',m,'Full Model'].n) for m in comparison_markets]==[66103,23030]
assert [int(data.loc['offline',m,'Kronos (zero-shot)'].n) for m in ('csi300','sp500')]==[66103,117464]
for market in comparison_markets:
    for name in names:
        expected=1 if name in names[:3] else 3
        assert len(comparisons.query('market==@market and task=="external" and model==@name'))==expected
body=[]
for name in names:
    values=[]
    for m in comparison_markets:
        r=data.loc['external',m,name]
        values += [f'{r.rankic:.5f}', f'{100*r.annualized_return:.2f}',f'{r.sharpe:.3f}',f'{100*r.max_drawdown:.2f}']
    body.append(row([name,*values]))
table('external_comparison','tab:external',
      r'External forecasts evaluated on adjusted labels and with corporate actions in portfolio wealth. CSI300 and Nasdaq-100 share 66,103 and 23,030 scored stock dates, respectively. AR and signed maximum drawdown (MDD) are percentages.',
      'lrrrrrrrr',[span(comparison_markets,4),r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
                   row(['Model',*(['RankIC','AR','Sharpe','MDD']*2)])],body)
parts=[]
for market in comparison_markets:
    subset=data.loc['external',market]
    top_rank=subset.rankic.idxmax()
    top_ar=subset.annualized_return.idxmax()
    r=subset.loc['Full Model']
    parts.append(f"The full model's RankIC and after-cost annualized return on {titles[market]} are "
                 f"{r.rankic:.5f} and {100*r.annualized_return:.2f}\\%, respectively."
                 + (" It has the highest observed value on both metrics." if top_rank==top_ar=='Full Model' else
                    f" The highest RankIC is from {top_rank}, and the highest annualized return is from {top_ar}."))
note('external_findings',' '.join(parts))

two=['csi300','sp500']
body=[]
for name in ['Kronos (zero-shot)','Linear head','Adapter + head']:
    cells=[]
    for market in two:
        r=data.loc['offline',market,name]
        cells += [f'{r.rankic:.5f}',f'{r.annualized_return*100:.2f}']
    body.append(row([name,*cells]))
table('offline_adaptation','tab:offline-adaptation',
      r'Offline comparisons on 66,103 shared CSI300 stock dates and 117,464 shared S\&P 500 stock dates. The fitted models average three seeds. AR is after-cost annualized return.',
      'lrrrr',[span(two,2),r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
               row(['Predictor','RankIC',r'AR (\%)','RankIC',r'AR (\%)'])],body)
def ar(task,market,model):
    return f"{100*data.loc[task,market,model].annualized_return:.2f}\\%"
note('offline_findings',
     'The adapter plus head has the highest observed RankIC and annualized return on both panels. '
     'On CSI300, the linear head earns '+ar('offline','csi300','Linear head')+
     ' and the adapter plus head earns '+ar('offline','csi300','Adapter + head')+
     r'. On S\&P 500, the linear head earns '+ar('offline','sp500','Linear head')+
     ', zero-shot Kronos earns '+ar('offline','sp500','Kronos (zero-shot)')+
     ', and the adapter plus head earns '+ar('offline','sp500','Adapter + head')+'.')

body=[]
for name in models:
    means,stds=[],[]
    for market in markets:
        r=online.loc[market,name]
        means += [f'{r.rankic_mean:.5f}',f'{100*r.annualized_return_mean:.2f}',f'{r.sharpe_mean:.3f}']
        stds += [rf'{{\scriptsize ({r.rankic_std:.5f})}}',rf'{{\scriptsize ({100*r.annualized_return_std:.2f})}}','']
    body += [row([name,*means]),row(['',*stds])]
table('ablation','tab:ablation',
      r'Online updates under common corporate-action accounting. Parentheses give sample standard deviations across three seeds.',
      'lrrrrrrrrr',[span(markets,3),r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}',
                    row(['Variant',*(['RankIC',r'AR (\%)','Sharpe']*3)])],body,padding=1.15)
body=[]
for metric,mult,places,title in [('annualized_return_mean',100,2,r'(a) After-cost annualized return (\%)'),
                               ('rankic_mean',1,5,'(b) RankIC')]:
    body.append(rf'\multicolumn{{5}}{{@{{}}l}}{{\textit{{{title}}}}} \\')
    for m in markets:
        body.append(row([titles[m],*[f'{mult*peers.loc[m,p][metric]:.{places}f}' for p in peer]]))
    body.append(r'\addlinespace[2pt]')
body += [r'\bottomrule',r'\end{tabular*}',r'\smallskip',
         r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrr@{}}',
         r'\toprule',r'\multicolumn{4}{@{}l}{\textit{(c) Online PCA minus alternative in annualized return}} \\',
         row(['Market',r'vs.\ Target only',r'vs.\ Unprojected peers',r'vs.\ Fixed PCA basis']),r'\midrule']
for m in markets:
    cells=[cell(pci.loc[(pci.market==m)&(pci.control==p)].iloc[0]) for p in peer[:3]]
    body.append(row([titles[m],*cells]))
table('peer_pca','tab:peer-pca',
      r'Peer correction under shared joint updates. Panel (c) reports percentage points and ten-session 95\% paired intervals.',
      'lrrrr',[row(['Market',*peer])],body)
body=[row([titles[m] if model==models[0] else '', model, f'{r.rmse_mean:.5f}',f'{r.mae_mean:.5f}',f'{100*r.da_mean:.2f}'])
      for m in markets for model in models for r in [online.loc[m,model]]]
table('endpoint_errors','tab:endpoint_errors',
      r'Errors on the observed adjusted Endpoint targets. DA is direction accuracy.',
      'llrrr',[row(['Market','Variant',r'RMSE $\downarrow$',r'MAE $\downarrow$',r'DA (\%) $\uparrow$'])],body)
body=[row([titles[m],cell(urow(m,models[0])),cell(urow(m,models[1])),
           cell(urow(m,models[1],metric='rankic'),5)]) for m in markets]
table('portfolio_intervals','tab:portfolio_intervals',
      r'Joint update differences relative to matched policies. AR differences are percentage points. Brackets give 95\% ten-session paired intervals.',
      'lrrr',[row(['Market','AR vs. Static','AR vs. Head Only','RankIC vs. Head Only'])],body)
body=[]
for treatment, control, title in [(models[2],models[1],'Joint vs. head-only'),
                                  (models[1],models[0],'Head-only vs. static')]:
    if body:
        body.append(r'\addlinespace[2pt]')
    body.append(rf'\multicolumn{{5}}{{@{{}}l}}{{\textit{{{title}}}}} \\')
    for m in markets:
        values=[urow(m,control,d,treatment=treatment) for d in (10,20,40)]
        body.append(row([titles[m],f'{values[0].delta:+.2f}',*[interval(r) for r in values]]))
table('adapter_block_intervals','tab:adapter-blocks',
      r'Annualized return differences in percentage points across block lengths, with 95\% paired intervals.',
      'lrrrr',[row(['Market','Difference','10 sessions','20 sessions','40 sessions'])],body)

curve=pd.read_csv(HERE/'figure3_annualized_curves.csv',parse_dates=['as_of_date'])
fig=plt.figure(figsize=(6.6,3.55))
grid=fig.add_gridspec(2,3,height_ratios=[1,1.05],left=.127,right=.98,bottom=.20,top=.91,
                     hspace=1.05,wspace=.32)
ax=fig.add_subplot(grid[0,:])
ax.axvline(0,color='0.65',ls='--',lw=.8)
for i,m in enumerate(markets):
    r=urow(m,models[0])
    ax.errorbar(r.delta,2-i,xerr=[[r.delta-r.low],[r.high-r.delta]],fmt='o',ms=4,capsize=3,
                color='#0b6e66' if r.low>0 else '#56687b')
    ax.text(.99,2-i,cell(r),ha='right',va='center',transform=ax.get_yaxis_transform(),fontsize=7.5)
ax.set_yticks(range(3),['S&P 500','Nasdaq-100','CSI300'],fontsize=8)
ax.set_ylim(-.4,2.4)
ax.set_xlim(-48,133)
ax.set_xlabel('Annualized return difference (percentage points)',fontsize=8)
ax.set_title('(a) Joint updates minus Static Adapter',loc='left',fontsize=9,fontweight='bold')
axes=[]
for i,m in enumerate(markets):
    ax=fig.add_subplot(grid[1,i])
    axes.append(ax)
    for model,color,style in zip(models,['#56687b','#a44b21','#0b6e66'],[':', '--','-']):
        z=curve.loc[(curve.market==m)&(curve.model==model)]
        assert np.isclose(z.annualized_return.iloc[-1],online.loc[m,model].annualized_return_mean)
        ax.plot(z.as_of_date,100*z.annualized_return,color=color,ls=style,lw=1.15,label=model.replace('Full Model','Joint updates'))
    ax.set_title(titles[m].replace(r'\&','&'),fontsize=8)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[10,2,6]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.tick_params(labelsize=7)
    ax.grid(axis='y',lw=.45,color='.85')
    if i==0:
        ax.set_ylabel('Annualized return (%)',fontsize=8)
handles, legend = axes[1].get_legend_handles_labels()
fig.text(.127,.524,'(b) Annualized portfolio returns',fontsize=9,fontweight='bold')
fig.legend(handles,legend,loc='lower center',bbox_to_anchor=(.55,.005),ncol=3,frameon=False,fontsize=7)
for ax in fig.axes:
    for edge in ('top','right'):
        ax.spines[edge].set_visible(False)
fig.savefig(OUT/'figures/revised_curves/online_update_ablation.pdf')
comparisons.to_csv(OUT / 'paper_comparisons_by_seed.csv', index=False)
print(data[['rankic','annualized_return']].to_string())
