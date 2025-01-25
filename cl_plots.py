import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import math

file = './2leaders/res_4strats_M0_f0'
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

                benefit_ss = 0
                benefit_ww = 0
                benefit_sw = 0

                if Nw > 0:
                    benefit_ww = (
                        ((Nwc/Nw)*((Nwc-1)/(Nw-1)))*( # both leaders are cooperators
                            eps1*2 + 
                            (1-pF[0,0])*((Nwc-2)*eps1 + Nwd*eps)+
                            (1-pF[1,0])*(Nsc*eps1 + Nsd*eps)+
                            pF[0,0]*(Nw-2)*(eps1**2+eps**2)+pF[1,0]*Ns*(eps1**2+eps**2)
                        ) + ((Nwd/Nw)*((Nwd-1)/(Nw-1)))*( # both leaders are defectors
                            # TODO multiply by probability that there are two weak defectors???
                            eps*2 + 
                            (1-pF[0,0])*(Nwc*eps1 + (Nwd-2)*eps)+
                            (1-pF[1,0])*(Nsc*eps1 + Nsd*eps)+
                            pF[0,0]*(Nw-2)*(2*eps*eps1) + pF[1,0]*Ns*(2*eps*eps1)
                        ) + ((Nwc/Nw)*(Nwd/(Nw-1)))*2*( # one cooperator one defector
                            eps1+eps +
                            (1-pF[0,0])*((Nwc-1)*eps1 + (Nwd-1)*eps)+
                            (1-pF[1,0])*(Nsc*eps1 + Nsd*eps)+
                            (1/2)*(pF[0,0]*(Nw-2)*(eps1**2+eps**2)+pF[1,0]*Ns*(eps1**2+eps**2)) + # choose cooperating leader
                            (1/2)*(pF[0,0]*(Nw-2)*(2*eps*eps1)+pF[1,0]*Ns*(2*eps*eps1)) # choose defecting leader
                        )
                    )

                if Ns > 0:
                    benefit_ss = (
                        ((Nsc/Ns)*((Nsc-1)/(Ns-1)))*( # both leaders are cooperators
                            eps1*2 + 
                            (1-pF[0,1])*(Nwc*eps1 + Nwd*eps)+
                            (1-pF[1,1])*((Nsc-2)*eps1 + Nsd*eps)+
                            pF[0,1]*Nw*(eps1**2+eps**2)+pF[1,1]*(Ns-2)*(eps1**2+eps**2)
                        ) + ((Nsd/Ns)*((Nsd-1)/(Ns-1)))*( # both leaders are defectors
                            eps*2 + 
                            (1-pF[0,1])*(Nwc*eps1 + Nwd*eps)+
                            (1-pF[1,1])*(Nsc*eps1 + (Nsd-2)*eps)+
                            pF[0,1]*Nw*(2*eps*eps1) + pF[1,1]*(Ns-2)*(2*eps*eps1)
                        ) + ((Nsc/Ns)*(Nsd/(Ns-1)))*2*( # one cooperator one defector
                            eps1+eps +
                            (1-pF[0,1])*(Nwc*eps1 + Nwd*eps)+
                            (1-pF[1,1])*((Nsc-1)*eps1 + (Nsd-1)*eps)+
                            (1/2)*(pF[0,1]*Nw*(eps1**2+eps**2)+pF[1,1]*(Ns-2)*(eps1**2+eps**2)) + # choose cooperating leader
                            (1/2)*(pF[0,1]*Nw*(2*eps*eps1)+pF[1,1]*(Ns-2)*(2*eps*eps1)) # choose defecting leader
                        )
                    )

                if Ns > 0 and Nw > 0:
                    benefit_sw = (
                        ((Nsc/Ns)*(Nwc/Nw))*( # both leaders are cooperators
                            eps1*2 +
                            (
                                (fw/(fw+fs))*(
                                    (1-pF[0,0])*((Nwc-1)*eps1+Nwd*eps)+
                                    (1-pF[1,0])*((Nsc-1)*eps1+Nsd*eps)+
                                    pF[0,0]*(Nw-1)*(eps1**2+eps**2)+pF[1,0]*(Ns-1)*(eps1**2+eps**2)
                                ) + 
                                (fs/(fw+fs))*(
                                    (1-pF[0,1])*((Nwc-1)*eps1+Nwd*eps)+
                                    (1-pF[1,1])*((Nsc-1)*eps1+Nsd*eps)+
                                    pF[0,1]*(Nw-1)*(eps1**2+eps**2)+pF[1,1]*(Ns-1)*(eps1**2+eps**2)                                    
                                )
                            )
                        )+
                        ((Nsd/Ns)*(Nwd/Nw))*( # both leaders are defectors
                            eps*2 +
                            (
                                (fw/(fw+fs))*(
                                    (1-pF[0,0])*(Nwc*eps1+(Nwd-1)*eps)+
                                    (1-pF[1,0])*(Nsc*eps1+(Nsd-1)*eps)+
                                    pF[0,0]*(Nw-1)*(2*eps1*eps)+pF[1,0]*(Ns-1)*(2*eps1*eps)
                                ) + 
                                (fs/(fw+fs))*(
                                    (1-pF[0,1])*(Nwc*eps1+(Nwd-1)*eps)+
                                    (1-pF[1,1])*(Nsc*eps1+(Nsd-1)*eps)+
                                    pF[0,1]*(Nw-1)*(2*eps1*eps)+pF[1,1]*(Ns-1)*(2*eps1*eps)                                    
                                )
                            )
                        )+
                        ((Nsd/Ns)*(Nwc/Nw))*( # one weak cooperator one strong defector
                            eps1+eps +
                            (
                                (fw/(fw+fs))*(
                                    (1-pF[0,0])*((Nwc-1)*eps1+Nwd*eps)+
                                    (1-pF[1,0])*(Nsc*eps1+(Nsd-1)*eps)+
                                    pF[0,0]*(Nw-1)*(eps1**2+eps**2)+pF[1,0]*(Ns-1)*(eps1**2+eps**2)
                                ) + 
                                (fs/(fw+fs))*(
                                    (1-pF[0,1])*((Nwc-1)*eps1+Nwd*eps)+
                                    (1-pF[1,1])*(Nsc*eps1+(Nsd-1)*eps)+
                                    pF[0,1]*(Nw-1)*(2*eps1*eps)+pF[1,1]*(Ns-1)*(2*eps1*eps)                                    
                                )
                            )
                        )+
                        ((Nsc/Ns)*(Nwd/Nw))*( # one strong cooperator one weak defector
                            eps1+eps +
                            (
                                (fw/(fw+fs))*(
                                    (1-pF[0,0])*(Nwc*eps1+(Nwd-1)*eps)+
                                    (1-pF[1,0])*((Nsc-1)*eps1+Nsd*eps)+
                                    pF[0,0]*(Nw)*(2*eps1*eps)+pF[1,0]*(Ns)*(2*eps1*eps)
                                ) + 
                                (fs/(fw+fs))*(
                                    (1-pF[0,1])*(Nwc*eps1+(Nwd-1)*eps)+
                                    (1-pF[1,1])*((Nsc-1)*eps1+Nsd*eps)+
                                    pF[0,1]*(Nw-1)*(eps1**2+eps**2)+pF[1,1]*(Ns-1)*(eps1**2+eps**2)                                    
                                )
                            )
                        )                                                                                    
                    )                

                benefit = (
                    (pW**N)*benefit_ww + # everyone is weak
                    (pS**N)*benefit_ss + # everyone is strong
                    ((pW**(N-1)*pS*N))*( # one strong everyone else weak
                        (((N-1)*fw)/((N-1)*fw+fs)*((N-2)*fw)/((N-2)*fw+fs))*benefit_ww +
                        (((N-1)*fw)/((N-1)*fw+fs)*(fs)/((N-2)*fw+fs))*benefit_sw+
                        ((fs)/((N-1)*fw+fs)*((N-1)*fw)/((N-1)*fw))*benefit_sw
                    )+
                    ((pS**(N-1)*pW*N))*( # one weak everyone else strong
                        (((N-1)*fs)/((N-1)*fs+fw)*((N-2)*fs)/((N-2)*fs+fw))*benefit_ss +
                        (((N-1)*fs)/((N-1)*fs+fw)*(fw)/((N-2)*fs+fw))*benefit_sw+
                        ((fw)/((N-1)*fs+fw)*((N-1)*fs)/((N-1)*fs))*benefit_sw
                    )
                )

                for nums in range(2, N-1):
                    numw = N - nums
                    benefit += (pW**numw * pS**nums)*(math.factorial(N)/(math.factorial(nums)*math.factorial(N-nums)))*(
                        ((numw*fw)/(numw*fw+nums*fs)*((numw-1)*fw)/((numw-1)*fw+nums*fs))*benefit_ww+
                        ((nums*fs)/(numw*fw+nums*fs)*((nums-1)*fs)/(numw*fw+(nums-1)*fs))*benefit_ss+
                        ((numw*fw)/(numw*fw+nums*fs)*(nums*fs)/((numw-1)*fw+nums*fs))*benefit_sw+
                        ((nums*fs)/(numw*fs+nums*fs)*(numw*fw)/(numw*fw+(nums-1)*fs))*benefit_sw
                    )

                benefit /= N
                
                res[idps] += benefit * data[idr, iddl, idps, strat]
            

        ax.set_xticks(np.linspace(0, pSv.shape[0]-1, nticksX))
        ax.set_xticklabels(np.linspace(pSv[0],pSv[-1],nticksX), fontsize=12)
        ax.set_yticks(np.linspace(0, 1, 3))
        ax.set_yticklabels(np.linspace(0,1,3), fontsize=12)
        # ax.set_ylim(0.0, 1.0)
        ax.plot(res, label='$\Delta_f=\Delta_f=%d$'%deltaF, color=cmap((iddl+1)/(len(deltaLv)+1)))

        if i==nr-1: ax.set_xlabel(r'$p_s$', fontsize=fntsize)
        if j==0: ax.set_ylabel(r'cooperation level', fontsize=fntsize)
        ax.text(20,1.06,"$r$=%d" % rv[idr], size=13)

legend_elements = [Line2D([], [], marker='None', label='Leader: $\Delta_l=\Delta_f$', linestyle='None')]
legend_elements += [Line2D([], [], marker='s', color=cmap((idx+1)/(len(deltaLv)+1)), label='%d'%deltaLv[idx],
                          markerfacecolor=cmap((idx+1)/(len(deltaLv)+1)), markersize=10, linestyle='None') for idx in range(len(deltaLv))]
plt.legend( loc='upper center', bbox_to_anchor=(-2.1, -0.6),
          fancybox=True, shadow=False, ncol=7, columnspacing=0.0, handles=legend_elements,handletextpad=-0.3,fontsize=13)
plt.savefig('./2leaders/cl_fig.png', bbox_inches='tight', dpi=300)

plt.show()