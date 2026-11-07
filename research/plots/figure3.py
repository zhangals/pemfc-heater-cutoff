from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'regenerated_figures'
DATA=ROOT/'results'
def dense_minimum_trace(trace, times):
    segments, coefficients, orders = trace
    times = np.atleast_1d(np.asarray(times, dtype=float))
    if min(times)<segments[0,0] or max(times)>segments[-1,1]:
        raise ValueError("Time requested outside the dense output interval.")
    idx = np.searchsorted(segments[:,1], times, side="left").clip(0,len(segments)-1)
    z = (times-segments[idx,0])/(segments[idx,1]-segments[idx,0])
    cells = np.zeros((len(times),5))
    for degree in reversed(range(coefficients.shape[1])):
        active = degree<=orders[idx]
        cells *= z[:,None]
        cells[active] += coefficients[idx[active],degree,:]
    return np.min(cells,axis=1)


def main():
    with (DATA/'matched_cutoff.csv').open() as f: row=next(r for r in csv.DictReader(f) if int(r['refine'])==4)
    cutoffs=[float(row['first_cutoff_s']),float(row['sustained_cutoff_s'])]
    minima=[float(row['first_late_min_C']),float(row['sustained_min_C'])]
    min_times=[120.,120.]
    traces=[]
    for name in ['first_crossing','extended_heating']:
        with np.load(DATA/(name+'.npz'),allow_pickle=False) as z:
            traces.append(tuple(z[k] for k in ['dense_segment_times_s','dense_temperature_coefficients_C','dense_orders']))
    ART.mkdir(exist_ok=True)
    colors=["#0072B2","#C55A17"]
    styles=["-",(0,(4.1,2.1))]
    fig,ax=plt.subplots(1,2,figsize=(17/2.54,7.8/2.54),sharey=True)
    fig.subplots_adjust(left=.105,right=.983,top=.79,bottom=.18,wspace=.14)
    ax[1].axvspan(120,180,color="#E2EEE7",lw=0,zorder=0)
    ax[1].axvline(120,color="#799282",lw=.8,zorder=1)
    ax[1].text(151,6.18,"Assessment window",fontsize=7.8,ha="center",va="top",color="#365845")
    labels=("First 1 °C crossing","Extended heating")
    for panel,limits,xticks,head,tag in zip(ax,[(30,75),(90,180)],[[30,40,50,60,70],[90,120,150,180]],["Heater cutoff and recooling","Temperature under load"],("a","b")):
        panel.set(xlim=limits,ylim=(-5.6,6.5),xticks=xticks,yticks=[-4,-2,0,2,4,6],xlabel="Time (s)")
        panel.tick_params(length=3,width=.7)
        panel.text(-.035,1.044,tag,transform=panel.transAxes,fontsize=10.2,fontweight="bold",ha="right")
        panel.set_title(head,fontsize=9.4,loc="left",pad=6)
        panel.axhline(0,color="#D7DCE1",linewidth=.65,zorder=0)
        panel.axhline(1,color="#667077",linestyle=(0,(1.5,2)),linewidth=.9,zorder=1)
        for i,trace in enumerate(traces):
            points=np.linspace(*limits,1201)
            for special in [cutoffs[i],min_times[i]]:
                if limits[0]<=special<=limits[1]:
                    points=np.unique(np.r_[points,special])
            panel.plot(points,dense_minimum_trace(trace,points),c=colors[i],ls=styles[i],lw=1.6,zorder=2+i)
    ax[0].set_ylabel("Coldest cell-mean temperature (°C)")
    for i, (label_xy, txt_xy) in enumerate([((cutoffs[0],0.),(33.7,3.05)),((cutoffs[1],0.),(48.5,4.75))]):
        x=label_xy[0]; y=dense_minimum_trace(traces[i],x)[0]
        ax[0].scatter(x,y,s=25,marker="s",facecolors="white",edgecolors=colors[i],linewidths=1.0,zorder=4)
        ax[0].annotate(f"{x:.2f} s",xy=(x,y),xytext=txt_xy,fontsize=7.75,color=colors[i],
            ha="center",va="center",arrowprops={"arrowstyle":"-","lw":.7,"color":colors[i],"shrinkB":3.5},zorder=5)
    for i,(x,y,xytext) in enumerate(zip(min_times,minima,[(132.,-2.25),(115.0,3.55)])):
        ax[1].scatter(x,y,s=20,facecolors="white",edgecolors=colors[i],linewidths=1.0,zorder=4)
        label=f"{y:.3f} °C".replace("-","−")
        ax[1].annotate(label,xy=(x,y),xytext=xytext,ha="center",va="center",color=colors[i],fontsize=7.75,
            arrowprops={"arrowstyle":"-","lw":.7,"color":colors[i],"shrinkB":3.2},zorder=5)
    handles=[Line2D([],[],c=color,ls=ls,lw=1.65,label=lab) for color,ls,lab in zip(colors,styles,labels)]
    handles.append(Line2D([],[],c="#667077",ls=(0,(1.5,2)),lw=.9,label="1 °C requirement"))
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.55,1.0),ncol=3,frameon=False,
               fontsize=8.3,columnspacing=1.75,handlelength=2.5)
    ART.mkdir(parents=True,exist_ok=True)
    for ext in ("png","pdf","svg"):
        fig.savefig(ART/f"Fig3.{ext}",dpi=600,facecolor="white")
    plt.close(fig)

if __name__=='__main__':main()
