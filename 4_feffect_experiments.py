import numpy as np
import evoEGT as evo
from heterogeneous4 import calcH, calcWCD

def coop_pF_r(r,M,N,HZ,beta,eps,pSv,deltaFv,fv,betaF):
# Input: pFv, rv, Mv (vectors with values of pF, r, and M), N, HZ (H or Z), beta, eps
# Output: matrix with the fraction of cooperators as a function of pF and r
    if np.isscalar(HZ):
        H=calcH(N-1,HZ-1)

    MAT = np.zeros((len(fv), len(deltaFv), len(pSv), 4))

    for idf, f, in enumerate(fv):
        for iddf, deltaF in enumerate(deltaFv):
            deltaL=deltaF
            for idps, pS in enumerate(pSv):
                pF=np.zeros((2,2))
                # pF[0,0] = 1/(1+np.exp(-betaF*(f+deltaF)))
                # pF[1,1] = 1/(1+np.exp(-betaF*(f-deltaF)))
                # pF[0,1] = 1/(1+np.exp(-betaF*(f+2*deltaF)))
                # pF[1,0] = 1/(1+np.exp(-betaF*(f-2*deltaF)))

                pF[0,0] = 1/(1+np.exp(-betaF*(f)))
                pF[1,1] = 1/(1+np.exp(-betaF*(f)))
                pF[0,1] = 1/(1+np.exp(-betaF*(f+deltaF)))
                pF[1,0] = 1/(1+np.exp(-betaF*(f-deltaF)))

                WCD=calcWCD(N,eps,pF,deltaL,pS,M)
                #Wgen=transfW2Wgen(WCD) # transforming to evoEGT format
                print(r,pS)
                SD,fixM = evo.Wgroup2SD(WCD,H,[r,-1.],beta,infocheck=False)
                MAT[idf, iddf, idps, :] = SD[:,0]
    return MAT

def plotCOOPheat(data,fv,deltaFv,pSv,r,label):
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    nr = 3
    nc = 2
    fntsize=15

    betaF=1.
    N = 9
    eps = 0.01
    eps1 = 1 - eps

    fig,axs=plt.subplots(nrows=nr, ncols=nc, sharex='all', sharey='all', figsize=(5,6))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    cmap = plt.get_cmap('viridis')

    for idf, f in enumerate(fv):
        i = idf // nc
        j = idf % nc
        ax=axs[i,j]
        for iddf, deltaF in enumerate(deltaFv):
            deltaL = deltaF
            ss = 1/(1+np.exp(-deltaL))
            sw=1-ss
            res = np.zeros((pSv.shape[0]))
            for strat in range(4):
                for idps, pS in enumerate(pSv):
                    pF = np.zeros((2,2))
                    # pF[0,0] = 1/(1+np.exp(-betaF*(f+deltaF)))
                    # pF[1,1] = 1/(1+np.exp(-betaF*(f-deltaF)))
                    # pF[0,1] = 1/(1+np.exp(-betaF*(f+2*deltaF)))
                    # pF[1,0] = 1/(1+np.exp(-betaF*(f-2*deltaF)))

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
                    # cl_l = (((Nw*sw)/(Nw*sw+Ns*ss))*coops_w+((Ns*ss)/(Nw*sw+Ns*ss))*coops_s) / N
                    # cl = pl*cl_l + (1-pl)*cl_nol

                    res[idps] += cl * data[idf, iddf, idps, strat]
                
            nticksY=6
            nticksX=3
            ax.set_xticks(np.linspace(0, pSv.shape[0]-1, nticksX))
            ax.set_xticklabels(np.linspace(pSv[0],pSv[-1],nticksX), fontsize=12)
            ax.set_yticks(np.linspace(0, 1, 3))
            ax.set_yticklabels(np.linspace(0,1,3), fontsize=12)
            # ax.set_ylim(0.0, 1.0)
            ax.plot(res, label='$\Delta_f=%d$'%deltaF, color=cmap((iddf)/len(deltaFv)))
            if i==nr-1: ax.set_xlabel(r'$p_s$', fontsize=fntsize)
            if j==0 and i==nr//2: ax.set_ylabel(r'cooperation level', fontsize=fntsize)
            ax.text(20,1.06,"$f$=%d" % fv[idf], size=13)   

    legend_elements = [Line2D([], [], marker='None', label='$\Delta_l=\Delta_f=$', linestyle='None')]
    legend_elements += [Line2D([], [], marker='s', color=cmap((idx)/len(deltaFv)), label='%d'%deltaFv[idx],
                            markerfacecolor=cmap((idx)/len(deltaFv)), markersize=10, linestyle='None') for idx in range(len(deltaFv))]
    ax.legend( loc='upper center', bbox_to_anchor=(-0.11, -0.4),
          fancybox=True, shadow=False, ncol=6, columnspacing=0, handles=legend_elements,handletextpad=-0.1, fontsize=13)
    fig.savefig(label+'.png',bbox_inches='tight',dpi=300)
    plt.show()
    fig.clf() 

    return

def plotsingleheat(MAT,fv,rv,label):
# Input: MAT (matrix from "coop_pF_r" function), pFv, rv ,Mv (vectors with values of pF, r, and M), label (name for the output file)
# Output: heatmap plot of the fraction of cooperators as a function of pF and r, for different M
    import matplotlib.pyplot as plt
    fntsize=12
    f,ax=plt.subplots()
    h=ax.imshow(MAT,origin='lower', interpolation='none',aspect='auto')
    nticksY=5
    nticksX=3
    ax.set_xticks(np.linspace(0, MAT.shape[1]-1, nticksX))
    ax.set_yticks(np.linspace(0, MAT.shape[0]-1, nticksY))
    ax.set_xticklabels(np.linspace(fv[0],fv[-1],nticksX))
    ax.set_yticklabels(np.linspace(rv[0],rv[-1],nticksY))
    ax.set_xlabel(r'$f$', fontsize=fntsize)
    ax.set_ylabel(r'$r$', fontsize=fntsize)
#cb=f.colorbar(h, fraction=0.1,format='%.2f')
    #cb.set_label(label=r'$f_C$')
    f.savefig(label+'.png',bbox_inches='tight',dpi=300)
    f.clf()     
    return


if __name__ == "__main__":

    import time

    t0=time.time()

####### Plot heatmap #########################################
    eps=0.01 #0.01
    Z=100
    N=9
    beta=1.
    M=0
    fv=[-8,-4,-2,-1,0,1]
    betaF=1
    r=5

    deltaFv=[0, 1, 2, 4, 8]
    pSv=np.linspace(0,1.,num=50)
    
    labfilenpy='results/h4/cl/res_4strats_M0_f0_f'
    # MAT=coop_pF_r(r,M,N,Z,beta,eps,pSv,deltaFv,fv,betaF)
    # np.save(labfilenpy,MAT)             # save matrix for heatmap
    # print('data saved to file!')
    
    MAT=np.load(labfilenpy+'.npy')      # load matrix for heatmap 
    plotCOOPheat(MAT,fv,deltaFv,pSv,r,labfilenpy)      # plot heatmap
    #plotsingleheat(MAT,fv,rv,labfilenpy)
#####################################################