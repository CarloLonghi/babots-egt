import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import math

file = 'newtests/4bits/singleleader/res_singleleader'
data = np.load(file + '.npy')

nr = 2
nc = 5
fntsize=15

pSv=np.linspace(0.,1.,num=50)
deltaLv=[0, 1, 2, 4, 8]
f=0
betaF=1.
N = 9
eps = 0.01
eps1 = 1 - eps
rv=np.linspace(1,10,num=10)


fig,axs=plt.subplots(nrows=nr, ncols=nc, sharex='all', sharey='all', figsize=(10,5))
fig.subplots_adjust(hspace=0.4, wspace=0.2)
nticksY=6
nticksX=3

cmap = plt.get_cmap('viridis')


for idr, r in enumerate(rv):
    i = idr // nc
    j = idr % nc
    ax=axs[i,j]
    for iddl, deltaL in enumerate(deltaLv):
        deltaF = deltaL
        fs = 1/(1+np.exp(-deltaL))
        fw=1-fs
        res = np.zeros((pSv.shape[0]))
        for strat in range(16):
            s1=[strat%8%4%2//1, strat%8%4//2, strat%8//4, strat//8]
            for idps, pS in enumerate(pSv):
                pW = 1 - pS
                pF = np.zeros((2,2))

                pF[0,0] = 1/(1+np.exp(-betaF*(f)))
                pF[1,1] = 1/(1+np.exp(-betaF*(f)))
                pF[0,1] = 1/(1+np.exp(-betaF*(f+deltaF)))
                pF[1,0] = 1/(1+np.exp(-betaF*(f-deltaF)))
                
                benefit = 0
                cost = 0
                for Ns in range(N+1):
                    Nw = N - Ns
                    ns1 = Ns
                    nw1 = N - ns1
                    pNs = math.factorial(N)/(math.factorial(Ns)*math.factorial(Nw)) * pS**Ns * pW**Nw

                    Nwc = nw1*s1[2]
                    Nsc = ns1*s1[3]
                    Nwd = nw1*(1-s1[2])
                    Nsd = ns1*(1-s1[3])

                    N1wcl = nw1*s1[0]
                    N1wdl = nw1*(1-s1[0])
                    N1scl = ns1*s1[1]
                    N1sdl = ns1*(1-s1[1])

                    benefit_s = 0
                    benefit_w = 0
                    cost_s = 0
                    cost_w = 0

                    if Nw > 0:

                        if nw1 > 0:
                            benefit_w += (nw1/Nw)*( # leader is of strategy 1
                                    (N1wcl/nw1)*( # leader is a cooperator
                                        eps1+
                                        (1-pF[0,0])*((Nwc-s1[2])*eps1 + (Nwd-(1-s1[2]))*eps)+
                                        (1-pF[1,0])*(Nsc*eps1 + Nsd*eps)+
                                        pF[0,0]*(Nw-1)*(eps1**2+eps**2)+pF[1,0]*Ns*(eps1**2+eps**2)
                                    )+
                                    (N1wdl/nw1)*( # leader is a defector
                                        eps+
                                        (1-pF[0,0])*((Nwc-s1[2])*eps1 + (Nwd-(1-s1[2]))*eps)+
                                        (1-pF[1,0])*(Nsc*eps1 + Nsd*eps)+
                                        pF[0,0]*(Nw-1)*(2*eps1*eps)+pF[1,0]*Ns*(2*eps1*eps)
                                    )
                                )
                            
                    if Ns > 0:

                        if ns1 > 0:
                            benefit_s += (ns1/Ns)*( # leader is of strategy 1
                                (N1scl/ns1)*( # leader is a cooperator
                                    eps1+
                                    (1-pF[0,1])*(Nwc*eps1 + Nwd*eps)+
                                    (1-pF[1,1])*((Nsc-s1[3])*eps1 + (Nsd-(1-s1[3]))*eps)+
                                    pF[0,1]*Nw*(eps1**2+eps**2)+pF[1,1]*(Ns-1)*(eps1**2+eps**2)
                                )+
                                (N1sdl/ns1)*( # leader is a defector
                                    eps+
                                    (1-pF[0,1])*(Nwc*eps1 + Nwd*eps)+
                                    (1-pF[1,1])*((Nsc-s1[3])*eps1 + (Nsd-(1-s1[3]))*eps)+
                                    pF[0,1]*Nw*(2*eps1*eps)+pF[1,1]*(Ns-1)*(2*eps1*eps)
                                )
                            )

                    benefit += pNs*(((Nw*fw)/(Nw*fw+Ns*fs))*benefit_w + ((Ns*fs)/(Nw*fw+Ns*fs))*benefit_s)
                    cost += pNs*(((Nw*fw)/(Nw*fw+Ns*fs))*cost_w + ((Ns*fs)/(Nw*fw+Ns*fs))*cost_s)


                res[idps] += (benefit/N) * data[idr, iddl, idps, strat]
            

        ax.set_xticks(np.linspace(0, pSv.shape[0]-1, nticksX))
        ax.set_xticklabels(np.linspace(pSv[0],pSv[-1],nticksX), fontsize=12)
        ax.set_yticks(np.linspace(0, 1, 3))
        ax.set_yticklabels(np.linspace(0,1,3), fontsize=12)
        ax.set_ylim(0.0, 1.0)
        ax.plot(res, label='$\Delta_f=\Delta_f=%d$'%deltaF, color=cmap((iddl)/(len(deltaLv))))

        if i==nr-1: ax.set_xlabel(r'$p_s$', fontsize=fntsize)
        if j==0: ax.set_ylabel(r'cooperation level', fontsize=fntsize)
        ax.text(20,1.06,"$r$=%d" % rv[idr], size=13)
        #ax.set_ylim(0, 1)

legend_elements = [Line2D([], [], marker='None', label='Leader: $\Delta_l=\Delta_f$', linestyle='None')]
legend_elements += [Line2D([], [], marker='s', color=cmap((idx)/(len(deltaLv))), label='%d'%deltaLv[idx],
                          markerfacecolor=cmap((idx)/(len(deltaLv))), markersize=10, linestyle='None') for idx in range(len(deltaLv))]
plt.legend( loc='upper center', bbox_to_anchor=(-2.1, -0.6),
          fancybox=True, shadow=False, ncol=7, columnspacing=0.0, handles=legend_elements,handletextpad=-0.3,fontsize=13)
plt.savefig('newtests/4bits/singleleader/clplots.png', bbox_inches='tight', dpi=300)

plt.show()