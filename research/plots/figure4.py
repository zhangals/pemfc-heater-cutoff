"""Plot the constant-power capacity sweep and selected fine-grid temperature traces."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'regenerated_figures'
COLORS={0.5:'#0072B2',1.0:'#009E73',2.0:'#D55E00'}
def main():
 OUT.mkdir(exist_ok=True)
 with (ROOT/'results/figure4_capacity_scan.csv').open() as f: records=list(csv.DictReader(f))
 with (ROOT/'results/figure4_capacity_traces.csv').open() as f: traces=list(csv.DictReader(f))
 def row(raw):
  return dict(job={'factor':float(raw['capacity_factor']),'refine':int(raw['refine']),'cutoff':float(raw['cutoff_s'])},
              complete_180s=bool(raw['late_min_C']),dense_diagnostics={'late_window':{'minimum_mea_mean_C':float(raw['late_min_C']) if raw['late_min_C'] else None}})
 grid=[row(r) for r in records if r['stage']=='grid']
 selected={}
 for factor in COLORS:
  group=[r for r in records if float(r['capacity_factor'])==factor and r['stage']=='refinement' and r['refine']=='4' and r['late_min_C'] and float(r['late_min_C'])>=1]
  selected[factor]=row(min(group,key=lambda r:float(r['cutoff_s'])))
 fig,axs=plt.subplots(1,2,figsize=(6.6929133858,4.6456692913))
 fig.subplots_adjust(left=.105,right=.96,bottom=.33,top=.86,wspace=.40)
 a,b=axs
 for ax,title,label in zip(axs,['Heat-storage response','Recovered heating schedules'],['a','b']):
  ax.set_title(title,loc='left',pad=11)
  ax.annotate(label,xy=(0,1),xycoords='axes fraction',xytext=(-27,11),textcoords='offset points',ha='left',va='bottom',fontsize=10,fontweight='bold',annotation_clip=False)
  ax.set_axisbelow(True);ax.grid(axis='y',color='#E8E8E8',lw=.55)
 for factor,col in COLORS.items():
  ds=[d for d in grid if d['job']['factor']==factor]
  for r in [2,4]:
   cc=sorted([d for d in ds if d['job']['refine']==r and d['complete_180s']],key=lambda d:d['job']['cutoff'])
   xx=[.125*d['job']['cutoff'] for d in cc]
   yy=[d['dense_diagnostics']['late_window']['minimum_mea_mean_C'] for d in cc]
   if r==4:a.plot(xx,yy,color=col,lw=1.,alpha=.65)
   a.plot(xx,yy,ls='none',marker='o',markersize=5.3 if r==2 else 3.0,mfc='white' if r==2 else col,mec=col,mew=.9,zorder=3 if r==2 else 4)
  d=selected[factor]
  a.plot(.125*d['job']['cutoff'],d['dense_diagnostics']['late_window']['minimum_mea_mean_C'],marker='*',color=col,mec='white',mew=.4,markersize=9,zorder=5)
 a.axhline(1.,color='#555555',ls=(0,(4,3)),lw=.85)
 a.set(xlim=(1.8,12.),ylim=(-8,59),xlabel='Auxiliary energy (kJ)',ylabel='Late-window minimum (°C)')
 a.set_xticks([2.5,5,7.5,10]);a.set_yticks([0,15,30,45])
 # A separate nonnumerical status strip keeps incomplete records out of the temperature axis.
 position=a.get_position()
 status=fig.add_axes([position.x0,.17,position.width,.055])
 status.set_xlim(a.get_xlim());status.set_ylim(0,2);status.set_axis_off()
 failures=[d for d in grid if not d['complete_180s']]
 assert len(failures)==6 and all(d['dense_diagnostics']['late_window']['minimum_mea_mean_C'] is None for d in failures)
 for factor,row in [(1.,1.5),(2.,.35)]:
  for d in failures:
   if d['job']['factor']==factor:
    marker='x' if d['job']['refine']==4 else '+'
    status.plot(.125*d['job']['cutoff'],row,marker=marker,color=COLORS[factor],ls='none',ms=4.5,mew=.8)
  status.text(7.5,row,f'{factor:g}× domain stop',fontsize=8.,ha='left',va='center')
 times=np.array([float(d['time_s']) for d in traces])
 curves={factor:np.array([float(d[f'factor{factor:g}_coldest_cell_mean_C']) for d in traces]) for factor in COLORS}
 b.axvspan(120,180,color='#F1F1F1',lw=0,zorder=0)
 b.axhline(1.,color='#555555',ls=(0,(4,3)),lw=.85)
 for factor,d in selected.items():
  col=COLORS[factor];y=curves[factor];tau=d['job']['cutoff']
  b.plot(times,y,color=col,label=f'{factor:g}×; {tau:g} s')
  yy=np.interp(tau,times,curves[factor])
  b.plot(tau,yy,marker='s',color=col,mec='white',mew=.5,ms=4.8,zorder=4)
 b.set(xlim=(0,180),ylim=(-32,12),xlabel='Time (s)',ylabel='Coldest cell mean (°C)')
 b.set_xticks([0,60,120,180]);b.set_yticks([-30,-20,-10,0,10])
 b.legend(loc='lower left',bbox_to_anchor=(.01,.02),handlelength=1.2,labelspacing=.4)
 handles=[Line2D([0],[0],color=c,lw=1.5,label=f'{f:g}× EP heat capacity') for f,c in COLORS.items()]
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.04),ncol=3,columnspacing=1.15,handlelength=1.5,fontsize=8.5)
 b.text(.0,-.21,'○ r2   ● r4   ★ refined design\n■ heater cutoff; q = 1 W cm⁻²',transform=b.transAxes,ha='left',va='top',fontsize=8.0,linespacing=1.6)
 fig.canvas.draw()
 fonts=[t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
 assert min(fonts)>=8
 fig.savefig(OUT/'Fig4.png',dpi=300,facecolor='white')
 plt.close(fig)
if __name__=='__main__':main()
