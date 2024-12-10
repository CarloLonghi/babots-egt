import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

def gaussian(x, mu, sig):
    return np.exp(-((x - mu) ** 2) / (2 * (sig ** 2))) / (np.sqrt(2 * np.pi) * sig)

file = 'results/multileader/cl/res_4strats_M0_f0'
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
        pleadS = 1/(1+np.exp(-deltaL))
        pleadW=1-pleadS
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

                Nwcl = Nwc * pleadW
                Nscl = Nsc * pleadS
                Nwdl = Nwd * pleadW
                Nsdl = Nsd * pleadS
                Nwl = Nwcl + Nwdl
                Nsl = Nscl + Nsdl

                pwc = 0; pwd = 0; psc = 0; psd = 0; pwl = 0; psl = 0
                if Nw > 0:
                    pwc = Nwc / Nw
                    pwd = 1 - pwc
                    pwl = Nwl / Nw
                if Ns > 0:
                    psc = Nsc / Ns
                    psd = 1 - psc
                    psl = Nsl / Ns

                strengths = np.linspace(0.1, 0.9, N)
                sPs = gaussian(strengths, 0.5, 0.5)
                sPs = sPs / sPs.sum()
                sPw = gaussian(strengths, 0, 0.5)
                sPw = sPw / sPw.sum()
                strongS = (sPs * strengths).sum()
                weakS = (sPw * strengths).sum()

                follow_s = (pleadS * Nsl) / (pleadS * Nsl + pleadW * Nwl)
                follow_w = (pleadW * Nwl) / (pleadS * Nsl + pleadW * Nwl)

                totls = pleadS * Nsl + pleadW * Nwl
                looklead = 1 / (1 + np.exp(-totls))

                coop_w = 0
                coop_s = 0

                cl = (
                    (Nwcl + Nscl) * eps1 + (Nwdl + Nsdl) * eps + # leaders
                    (N - Nsl - Nwl) * ( # non leaders
                        pW * ( # weak
                            looklead * (
                                follow_w * ( # choose a weak leader
                                    (1 - pF[0, 0]) * (pwc * eps1 + pwd * eps) + # not follow
                                    (pF[0, 0] * (pwc * (eps1**2 + eps**2) + pwd * (2*eps1*eps))) # follow
                                ) + 
                                follow_s * ( # choose a strong leader
                                    (1 - pF[0, 1]) * (pwc * eps1 + pwd * eps) +
                                    (pF[0, 1] * (psc * (eps1**2 + eps**2) + psd * (2*eps1*eps)))
                                )
                            ) + (1 - looklead) * (
                                pwc * eps1 + pwd * eps
                            )
                        ) +
                        pS * ( #strong
                            looklead * (
                                follow_w * ( # choose a weak leader
                                    (1 - pF[1, 0]) * (psc * eps1 + psd * eps) + # not follow
                                    (pF[1, 0] * (pwc * (eps1**2 + eps**2) + pwd * (2*eps1*eps))) # follow
                                ) + 
                                follow_s * ( # choose a strong leader
                                    (1 - pF[1, 1]) * (psc * eps1 + psd * eps) +
                                    (pF[1, 1] * (psc * (eps1**2 + eps**2) + psd * (2*eps1*eps)))
                                )   
                            ) + (1 - looklead) * (
                                psc * eps1 + psd * eps
                            )                         
                        )
                    )
                )

                cl = cl / N

                res[idps] += cl * data[idr, iddl, idps, strat]
            

        ax.set_xticks(np.linspace(0, pSv.shape[0]-1, nticksX))
        ax.set_xticklabels(np.linspace(pSv[0],pSv[-1],nticksX), fontsize=12)
        ax.set_yticks(np.linspace(0, 1, 3))
        ax.set_yticklabels(np.linspace(0,1,3), fontsize=12)
        ax.set_ylim(0.0, 1.0)
        ax.plot(res, label='$\Delta_f=\Delta_f=%d$'%deltaF, color=cmap((iddl+1)/(len(deltaLv)+1)))

        if i==nr-1: ax.set_xlabel(r'$p_s$', fontsize=fntsize)
        if j==0 and i==nr//2: ax.set_ylabel(r'cooperation level', fontsize=fntsize)
        ax.text(20,1.06,"$r$=%d" % rv[idr], size=13)

legend_elements = [Line2D([], [], marker='None', label='Leader: $\Delta_l=\Delta_f$', linestyle='None')]
legend_elements += [Line2D([], [], marker='s', color=cmap((idx+1)/(len(deltaLv)+1)), label='%d'%deltaLv[idx],
                          markerfacecolor=cmap((idx+1)/(len(deltaLv)+1)), markersize=10, linestyle='None') for idx in range(len(deltaLv))]
plt.legend( loc='upper center', bbox_to_anchor=(-2., -0.6),
          fancybox=True, shadow=False, ncol=7, columnspacing=0.0, handles=legend_elements,handletextpad=-0.3,fontsize=13)
plt.savefig('multileader_fig_new.png', bbox_inches='tight', dpi=300)

plt.show()