"""Render saved realized Top30 returns and Hedge weights; does not fit or tune."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(dynamic_dir):
    out=dynamic_dir/'evaluation'
    curves=pd.read_csv(out/'cumulative_net_return.csv',parse_dates=['as_of_date'])
    weights=pd.read_csv(dynamic_dir/'weight_log.csv',parse_dates=['update_date','next_test_date'])
    colors={'Static':'#365d85','Dynamic':'#176b67','Continual':'#b37722'}
    names={'Static':'D-RankIC Static','Dynamic':'Dynamic mix (initial alpha 0.3)','Continual':'Continual branch'}
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True,gridspec_kw={'height_ratios':[2.5,1,1]})
    for name in names:
        day=pd.read_csv(out/f'{name}_top30_daily_returns.csv',parse_dates=['as_of_date'])
        assert np.array_equal(day.as_of_date.to_numpy(),curves.as_of_date.to_numpy())
        wealth=np.cumprod(1+day.net_return.to_numpy())
        np.testing.assert_allclose(wealth-1,curves[name],atol=1e-12,rtol=0)
        axes[0].plot(curves.as_of_date,curves[name]*100,label=names[name],color=colors[name],lw=1.8,alpha=.95 if name!='Continual' else .7)
        axes[1].plot(curves.as_of_date,(wealth/np.maximum(1,np.maximum.accumulate(wealth))-1)*100,color=colors[name],lw=1.2)
    daily_alpha=pd.Series(.3,index=curves.as_of_date)
    for row in weights.itertuples():
        daily_alpha.loc[daily_alpha.index>=row.next_test_date]=row.alpha_after
    axes[2].step(daily_alpha.index,daily_alpha.to_numpy(),where='post',color=colors['Dynamic'],lw=1.8)
    axes[2].axhline(.3,color='#777',ls='--',lw=.7)
    axes[2].set_ylim(0,1)
    axes[2].text(.01,.82,f'Observed alpha: {daily_alpha.min():.4f} to {daily_alpha.max():.4f}',transform=axes[2].transAxes,fontsize=9,color='#555')
    fig.suptitle('D-RankIC: Top30/Drop3 cumulative net return',x=.08,y=.985,ha='left',weight='bold',fontsize=15)
    fig.text(.08,.952,'2025-07-01 to 2026-06-05 | min holding 5 days | 15 bps cost | seed 100',fontsize=9,color='#555')
    axes[0].set_ylabel('Cumulative net return (%)');axes[0].legend(loc='upper left',frameon=False)
    axes[1].set_ylabel('Drawdown (%)');axes[2].set_ylabel('Continual weight')
    for ax in axes:
        ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    fig.autofmt_xdate();fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(out/'cumulative_return_and_weights.png',dpi=180)
    fig.savefig(out/'cumulative_return_and_weights.svg')
    plt.close(fig)
    return out/'cumulative_return_and_weights.png'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dynamic_dir',type=Path)
    print(render(p.parse_args().dynamic_dir))
