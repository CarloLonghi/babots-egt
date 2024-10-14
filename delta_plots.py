import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

file = 'results/h4/cl/res_4strats_M0_f0_deltaeffect'
data = np.load(file + '.npy')

nr = 2
nc = 2
fntsize=15

pSv=np.linspace(0.,1.,num=50)
deltav=[0, 1, 2, 4, 8]
f=0
betaF=1.
N = 9
eps = 0.01
eps1 = 1 - eps
rv=[5,7]


fig,axs=plt.subplots(nrows=nr, ncols=nc, sharex='all', sharey='all', figsize=(5,5))
fig.subplots_adjust(hspace=0.9, wspace=0.2)
nticksY=6
nticksX=3

cmap = plt.get_cmap('viridis')


for idr, r in enumerate(rv):
    
    i = 0
    j = idr % nc
    ax=axs[i,j]

    deltaF = 1
    for iddl, deltaL in enumerate(deltav):
        ss = 1/(1+np.exp(-deltaL))
        sw=1-ss
        res = np.zeros((pSv.shape[0]))
        for strat in range(4):
            for idps, pS in enumerate(pSv):
                pF = np.zeros((2,2))
                pF[0,0] = 1/(1+np.exp(-betaF*(f)))
                pF[1,1] = 1/(1+np.exp(-betaF*(f)))
                pF[0,1] = 1/(1+np.exp(-betaF*(f+deltaF)))
                pF[1,0] = 1/(1+np.exp(-betaF*(f-deltaF)))
                
                stratW = strat%2
                stratS = strat//2

                pW = 1 - pS
                Nw = N * pW
                Ns = N * pS

                Nwc = pW * (N * stratW)
                Nwd = (N * pW) - Nwc
                Nsc = pS * (N * stratS)
                Nsd = (N * pS) - Nsc

                coops_w = 0
                coops_s = 0

                if Nw > 0:
                    coops_w = (
                        (Nwc/Nw)*( # leader is a cooperator
                            eps1 + 
                            (1-pF[0,0])*((Nwc-1)*eps1 + Nwd*eps)+
                            (1-pF[1,0])*(Nsc*eps1 + Nsd*eps)+
                            pF[0,0]*(Nw-1)*(eps1**2+eps**2)+pF[1,0]*Ns*(eps1**2+eps**2)
                        ) + (Nwd/Nw)*( # leader is a defector
                            eps + 
                            (1-pF[0,0])*(Nwc*eps1 + (Nwd-1)*eps)+
                            (1-pF[1,0])*(Nsc*eps1 + Nsd*eps)+
                            pF[0,0]*(Nw-1)*(2*eps*eps1) + pF[1,0]*Ns*(2*eps*eps1)
                        )
                    )

                if Ns > 0:
                    coops_s = (
                        (Nsc/Ns)*( # leader is a cooperator
                            eps1 + 
                            (1-pF[0,1])*(Nwc*eps1 + Nwd*eps)+
                            (1-pF[1,1])*((Nsc-1)*eps1 + Nsd*eps)+
                            pF[0,1]*Nw*(eps1**2+eps**2)+pF[1,1]*(Ns-1)*(eps1**2+eps**2)
                        ) + (Nsd/Ns)*( # leader is a defector
                            eps + 
                            (1-pF[0,1])*(Nwc*eps1 + Nwd*eps)+
                            (1-pF[1,1])*(Nsc*eps1 + (Nsd-1)*eps)+
                            pF[0,1]*Nw*(2*eps*eps1) + pF[1,1]*(Ns-1)*(2*eps*eps1)
                        )
                    )

                total_s = ss * Ns
                total_w = sw * Nw
                pl = 1 / (1+np.exp(-deltaL*pS))

                cl_nol = (Nwc + Nsc) / N

                cl = (((Nw*sw)/(Nw*sw+Ns*ss))*coops_w + ((Ns*ss)/(Nw*sw+Ns*ss))*coops_s) / N
                res[idps] += cl * data[idr, 0, iddl, idps, strat]
            

        ax.set_xticks(np.linspace(0, pSv.shape[0]-1, nticksX))
        ax.set_xticklabels(np.linspace(pSv[0],pSv[-1],nticksX), fontsize=12)
        ax.set_yticks(np.linspace(0, 1, 3))
        ax.set_yticklabels(np.linspace(0,1,3), fontsize=12)
        # ax.set_ylim(0.0, 1.0)
        ax.plot(res, label='$\Delta_l=%d$'%deltaF, color=cmap((iddl)/(len(deltav))))

        ax.set_xlabel(r'$p_s$', fontsize=fntsize)
        if j==0: ax.set_ylabel(r'cooperation level', fontsize=fntsize)
        ax.text(20,1.06,"$r$=%d" % rv[idr], size=13)

    i = 1
    j = idr % nc
    ax=axs[i,j]

    deltaL = 1
    for iddf, deltaF in enumerate(deltav):
        ss = 1/(1+np.exp(-deltaL))
        sw=1-ss
        res = np.zeros((pSv.shape[0]))
        for strat in range(4):
            for idps, pS in enumerate(pSv):
                pF = np.zeros((2,2))
                pF[0,0] = 1/(1+np.exp(-betaF*(f)))
                pF[1,1] = 1/(1+np.exp(-betaF*(f)))
                pF[0,1] = 1/(1+np.exp(-betaF*(f+deltaF)))
                pF[1,0] = 1/(1+np.exp(-betaF*(f-deltaF)))
                
                stratW = strat%2
                stratS = strat//2

                pW = 1 - pS
                Nw = N * pW
                Ns = N * pS

                Nwc = pW * (N * stratW)
                Nwd = (N * pW) - Nwc
                Nsc = pS * (N * stratS)
                Nsd = (N * pS) - Nsc

                coops_w = 0
                coops_s = 0

                if Nw > 0:
                    coops_w = (
                        (Nwc/Nw)*( # leader is a cooperator
                            eps1 + 
                            (1-pF[0,0])*((Nwc-1)*eps1 + Nwd*eps)+
                            (1-pF[1,0])*(Nsc*eps1 + Nsd*eps)+
                            pF[0,0]*(Nw-1)*(eps1**2+eps**2)+pF[1,0]*Ns*(eps1**2+eps**2)
                        ) + (Nwd/Nw)*( # leader is a defector
                            eps + 
                            (1-pF[0,0])*(Nwc*eps1 + (Nwd-1)*eps)+
                            (1-pF[1,0])*(Nsc*eps1 + Nsd*eps)+
                            pF[0,0]*(Nw-1)*(2*eps*eps1) + pF[1,0]*Ns*(2*eps*eps1)
                        )
                    )

                if Ns > 0:
                    coops_s = (
                        (Nsc/Ns)*( # leader is a cooperator
                            eps1 + 
                            (1-pF[0,1])*(Nwc*eps1 + Nwd*eps)+
                            (1-pF[1,1])*((Nsc-1)*eps1 + Nsd*eps)+
                            pF[0,1]*Nw*(eps1**2+eps**2)+pF[1,1]*(Ns-1)*(eps1**2+eps**2)
                        ) + (Nsd/Ns)*( # leader is a defector
                            eps + 
                            (1-pF[0,1])*(Nwc*eps1 + Nwd*eps)+
                            (1-pF[1,1])*(Nsc*eps1 + (Nsd-1)*eps)+
                            pF[0,1]*Nw*(2*eps*eps1) + pF[1,1]*(Ns-1)*(2*eps*eps1)
                        )
                    )

                total_s = ss * Ns
                total_w = sw * Nw
                pl = 1 / (1+np.exp(-deltaL*pS))

                cl_nol = (Nwc + Nsc) / N

                cl = (((Nw*sw)/(Nw*sw+Ns*ss))*coops_w + ((Ns*ss)/(Nw*sw+Ns*ss))*coops_s) / N
                res[idps] += cl * data[idr, 1, iddf, idps, strat]
            

        ax.set_xticks(np.linspace(0, pSv.shape[0]-1, nticksX))
        ax.set_xticklabels(np.linspace(pSv[0],pSv[-1],nticksX), fontsize=12)
        ax.set_yticks(np.linspace(0, 1, 3))
        ax.set_yticklabels(np.linspace(0,1,3), fontsize=12)
        # ax.set_ylim(0.0, 1.0)
        ax.plot(res, label='$\Delta_f=%d$'%deltaF, color=cmap((iddf)/(len(deltav))))

        ax.set_xlabel(r'$p_s$', fontsize=fntsize)
        if j==0 and i==nr//2: ax.set_ylabel(r'cooperation level', fontsize=fntsize)
        ax.text(20,1.06,"$r$=%d" % rv[idr], size=13)


legend_elements1 = [Line2D([], [], marker='None', label='$\Delta_l=$', linestyle='None')]
legend_elements1 += [Line2D([], [], marker='s', color=cmap((idx)/(len(deltav))), label='%d'%deltav[idx],
                          markerfacecolor=cmap((idx)/(len(deltav))), markersize=10, linestyle='None') for idx in range(len(deltav))]
legend_elements2 = [Line2D([], [], marker='None', label='$\Delta_f=$', linestyle='None')]
legend_elements2 += [Line2D([], [], marker='s', color=cmap((idx)/(len(deltav))), label='%d'%deltav[idx],
                          markerfacecolor=cmap((idx)/(len(deltav))), markersize=10, linestyle='None') for idx in range(len(deltav))]
legend1 = plt.legend( loc='upper left', bbox_to_anchor=(-1.02, 1.7),
          fancybox=True, shadow=False, ncol=7, columnspacing=0.0, handles=legend_elements1,handletextpad=-0.3,fontsize=13)
plt.legend( loc='upper right', bbox_to_anchor=(0.73, -0.38),
          fancybox=True, shadow=False, ncol=7, columnspacing=0.0, handles=legend_elements2,handletextpad=-0.3,fontsize=13)
plt.gca().add_artist(legend1)
ax.text(-18, 3.2,'$\Delta_f=1$',size=fntsize)
ax.text(-18,1.2,'$\Delta_l=1$',size=fntsize)
plt.savefig(file+'.png', bbox_inches='tight', dpi=300)

plt.show()