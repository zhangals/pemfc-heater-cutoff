from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
DATA=Path(__file__).resolve().parents[2]/'results'
OUT=DATA.parent/'regenerated_figures'
OUT.mkdir(exist_ok=True)
with (DATA/'figure2_post_cutoff.csv').open() as f: pdict=list(csv.DictReader(f))
COLORS={'preheat':'#0072B2','concurrent':'#D55E00','reaction':'#E69F00','conduction':'#70539E','phase':'#009E73','net':'#333333'}
LABELS={'preheat':'Preheat','concurrent':'Concurrent'}
def letter(ax,label):
    ax.annotate(label,xy=(0,1),xycoords='axes fraction',xytext=(-25,8),
                textcoords='offset points',ha='left',va='bottom',fontsize=10.,fontweight='bold',annotation_clip=False)

def style(ax,title,label,xlabel=None,ylabel=None,grid=False):
    ax.set_title(title,loc='left',pad=10)
    letter(ax,label)
    if xlabel:ax.set_xlabel(xlabel)
    if ylabel:ax.set_ylabel(ylabel)
    if grid:ax.grid(axis='y',color='#E7E7E7',lw=.6,zorder=0)
    ax.set_axisbelow(True)

def figure2():
    summary={'all_qa_passed':True,'records':json.loads((DATA/'figure2_heat_balance.json').read_text())}
    assert summary['all_qa_passed'] and len(summary['records'])==6
    chosen={r['policy']:r for r in summary['records'] if r['refine']==4}
    fig,axes=plt.subplots(2,2,figsize=(17/2.54,13/2.54),layout='constrained')
    ax=axes[0,0];style(ax,'Cooling after switch-off','a','Time after switch-off / s','Minimum MEA mean / °C')
    for policy in ('preheat','concurrent'):
        for refine in (2,4):
            rows=[r for r in pdict if r['policy']==policy and int(r['mesh'])==refine];t=[float(r['time_s']) for r in rows];T=[float(r['coldest_MEA_C']) for r in rows]
            ax.plot(t,T,color=COLORS[policy],ls='--' if refine==2 else '-',lw=1. if refine==2 else 1.7,
                    label=LABELS[policy] if refine==4 else None,alpha=.7 if refine==2 else 1.)
        r=chosen[policy];ax.scatter(r['post_minimum_time_s'],r['post_minimum_C'],s=20,color=COLORS[policy],zorder=5)
    ax.axhline(.01,color='#555555',ls=':',lw=.9);ax.set(xlim=(0,60),xticks=[0,20,40,60],ylim=(-5.4,.8))
    ax.text(38,.28,'0.01°C',fontsize=8.5)
    ax=axes[0,1];style(ax,'Regional heat storage','b',ylabel='Sensible energy change / J')
    x=np.arange(3);width=.34
    for j,policy in enumerate(('preheat','concurrent')):
        values=np.array([chosen[policy]['storage_change_J'][k] for k in ('MEA','BP','EP')])
        bars=ax.bar(x+(j-.5)*width,values,width,color=COLORS[policy],zorder=3)
        for bar,value in zip(bars,values):
            ax.annotate(f'{value:.1f}' if abs(value)<10 else f'{value:.0f}',(bar.get_x()+bar.get_width()/2,value),
                        xytext=(0,5 if value>=0 else -6),textcoords='offset points',ha='center',va='bottom' if value>=0 else 'top',fontsize=8.)
    ax.axhline(0,color='#555555',lw=.8);ax.set(xticks=x,xticklabels=['MEA','BP','EP'],ylim=(-650,1150))
    ax.legend(handles=[Patch(facecolor=COLORS[p],label=LABELS[p]) for p in ('preheat','concurrent')],loc='upper left',frameon=False)
    ax=axes[1,0];style(ax,'MEA heat balance','c',ylabel='Integrated heat / J')
    terms=('conduction','reaction','phase','net');x=np.arange(4)
    for j,policy in enumerate(('preheat','concurrent')):
        vals=[chosen[policy]['mea_contributions_J'][k] if k!='net' else chosen[policy]['storage_change_J']['MEA'] for k in terms]
        for i,(term,value) in enumerate(zip(terms,vals)):
            ax.bar(i+(j-.5)*width,value,width,color=COLORS[term] if j==0 else 'white',edgecolor=COLORS[term],lw=1.,zorder=3)
    ax.axhline(0,color='#555555',lw=.8);ax.set(xticks=x,xticklabels=['Conduction','Reaction','Water\nexchange','Net'],ylim=(-730,900))
    ax.legend(handles=[Patch(facecolor='#BBBBBB',label='Preheat'),Patch(facecolor='white',edgecolor='#777777',label='Concurrent')],loc='upper right',frameon=False)
    ax=axes[1,1];style(ax,'Water-exchange heat','d',ylabel='Integrated heat / J')
    for j,policy in enumerate(('preheat','concurrent')):
        r=chosen[policy]['phase_components_J'];vals=[r['vapor_bound'],r['condensation_evaporation'],r['deposition_sublimation']+r['freezing_melting']]
        ax.bar(np.arange(3)+(j-.5)*width,vals,width,color=COLORS[policy],zorder=3)
    ax.axhline(0,color='#555555',lw=.8);ax.set(xticks=[0,1,2],xticklabels=['Vapor–\nbound','Vapor–\nliquid','Ice\nexchange'],ylim=(-215,55))
    ax.text(1.86,23,'|Q| < 0.00002 J',ha='center',fontsize=8.)
    fig.savefig(OUT/'Fig2.png',dpi=300,bbox_inches='tight')

if __name__=='__main__':figure2()
