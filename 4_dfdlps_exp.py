import numpy as np
import evoEGT as evo
from heterogeneous4 import calcH, calcWCD

def coop_pF_r(rv,M,N,HZ,beta,eps,pSv,f,betaF,deltav):
# Input: pFv, rv, Mv (vectors with values of pF, r, and M), N, HZ (H or Z), beta, eps
# Output: matrix with the fraction of cooperators as a function of pF and r
    if np.isscalar(HZ):
        H=calcH(N-1,HZ-1)

    MAT = np.zeros((2, len(deltav), len(pSv), len(rv), 4))

    for idr, r in enumerate(rv):
        for iddl, deltaL in enumerate(deltav):
            deltaF = 1
            pF=np.zeros((2,2))
            pF[0,0] = 1/(1+np.exp(-betaF*(f)))
            pF[1,1] = 1/(1+np.exp(-betaF*(f)))
            pF[0,1] = 1/(1+np.exp(-betaF*(f+deltaF)))
            pF[1,0] = 1/(1+np.exp(-betaF*(f-deltaF)))
            for idps, pS in enumerate(pSv):
                WCD=calcWCD(N,eps,pF,deltaL,pS,M)
                print(deltaF,deltaL,pS)
                SD,fixM = evo.Wgroup2SD(WCD,H,[r,-1.],beta,infocheck=False)
                best_s = np.argmax(SD)
                if SD[best_s] >= 0.5:
                    MAT[0, iddl, idps, idr, best_s] = SD[best_s]

        for iddf, deltaF in enumerate(deltav):
            deltaL = 1
            pF=np.zeros((2,2))
            pF[0,0] = 1/(1+np.exp(-betaF*(f)))
            pF[1,1] = 1/(1+np.exp(-betaF*(f)))
            pF[0,1] = 1/(1+np.exp(-betaF*(f+deltaF)))
            pF[1,0] = 1/(1+np.exp(-betaF*(f-deltaF)))
            for idps, pS in enumerate(pSv):
                WCD=calcWCD(N,eps,pF,deltaL,pS,M)
                print(deltaF,deltaL,pS)
                SD,fixM = evo.Wgroup2SD(WCD,H,[r,-1.],beta,infocheck=False)
                best_s = np.argmax(SD)
                if SD[best_s] >= 0.5:
                    MAT[1, iddf, idps, idr, best_s] = SD[best_s]
    return MAT

def plotCOOPheat(MAT,deltaFv,pSv,rv,label):
# Input: MAT (matrix from "coop_pF_r" function), pFv, rv ,Mv (vectors with values of pF, r, and M), label (name for the output file)
# Output: heatmap plot of the fraction of cooperators as a function of pF and r, for different M
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib
    fntsize=20
    nr=1
    nc=3
    f,axs=plt.subplots(nrows=nr, ncols=nc, sharex='all', sharey='all', figsize=(17,5))
    f.subplots_adjust(hspace=0.4, wspace=0.2)
    k=-1
    for idx in range(len(rv)):
        i = 0
        j = idx % nc

        ax=axs[j]
        k=k+1
        cmaps=['Greens','Reds','Blues','Purples']
        for strat in range(4):
            step=0.025
            levels = np.arange(0.5-step, 1., step) + step
            h=ax.contourf(MAT[:,:,j,strat],levels,cmap=cmaps[strat], origin='lower',)
        #h=ax.imshow(MAT[:,:,k],origin='lower', interpolation='none',aspect='auto',vmin=0,vmax=4)
        nticksY=5
        nticksX=3
        ax.set_xticks(np.linspace(0, MAT.shape[1]-1, nticksX))
        ax.set_yticks(np.linspace(0, MAT.shape[0]-1, nticksY))
        ax.set_xticklabels(np.linspace(pSv[0],pSv[-1],nticksX), fontsize=18)
        ax.set_yticklabels(np.linspace(deltaFv[0],deltaFv[-1],nticksY), fontsize=18)
        ax.set_ylim(0,MAT.shape[1]-1)
        ax.text(22.5,50,"$r=%d$" % rv[j], size=20)
        if i==nr-1: ax.set_xlabel(r'$p_s$', fontsize=fntsize)
        if j==0: 
            ax.set_ylabel(r'$\Delta_f, \Delta_l$', fontsize=fntsize)

        # insert markers for invasion graphs
        # if idx == 0:
        #     points_x = [MAT.shape[1] // 2 - 0.5, MAT.shape[1] // 2 - 0.5, MAT.shape[1] // 2 - 0.5]
        #     points_y = [0.5, MAT.shape[0] // 8 - 0.5, MAT.shape[0] // 4 - 0.5]
        #     ax.scatter(points_x, points_y, color='goldenrod', marker='o')
    
    labels = ['ALLD', 'WCSD', 'WDSC', 'ALLC']
    patches = [mpatches.Patch(color=plt.get_cmap(cmaps[i])(0.9), label=labels[i]) for i in range(4)]
    plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(-0.7 , -0.2),
          fancybox=True, shadow=False, ncol=4, fontsize=20, columnspacing=1)
    # ax.text(-12,125,"$\Delta_f=1$", size=13)
    # ax.text(-12,57,"$\Delta_l=1$", size=13)

    f.subplots_adjust(right=0.9)
    cbar_ax = f.add_axes([0.93, 0.15, 0.02, 0.7])
    f.colorbar(matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(vmin=0.5, vmax=1), cmap='Greys'), cax=cbar_ax)
    cbar_ax.tick_params(labelsize=18)

    # box = ax.get_position()
    # fi.set_position([box.x0, box.y0 + box.height * 0.1,
    #              box.width, box.height * 0.9])
    #f.subplots_adjust(bottom=0.1)
    # cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    # cb = f.colorbar(h, cax=cbar_ax)
    # cb.set_ticks([1,2,3,4])
    # cb.set_ticklabels(['ALLD','WCSD','WDSC','ALLC'])

#cb=f.colorbar(h, fraction=0.1,format='%.2f')
    #cb.set_label(label=r'$f_C$')
    f.savefig('figures/presentation.png',bbox_inches='tight',dpi=300)
    #plt.show()
    f.clf()     
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
    f.savefig('presentationimg.png',bbox_inches='tight',dpi=300)
    f.clf()     
    return


if __name__ == "__main__":

    import time

    t0=time.time()

#### One try ########################################    
    # eps=0. #0.01
    # Z=100
    # N=9
    # r=9.8
    # beta=1.
    # H=calcH(N-1,Z-1)
    # payparam=np.array([r,-1.]) # assuming c=-1
    # WCD=calcWCD(N,eps,pF=0.,M=0)
    # print('WCD')
    # print('benef(1)')
    # print(WCD[...,0])
    # print('cost(0)')
    # print(WCD[...,1])
    # print('WCD*payparam')
    # print(np.dot(WCD,payparam))
    # #Wtotavg=calcWtotavg(WCD,N,Z)
    # Wg=transfW2Wgen(WCD)
    # SD,fixM=evo.Wgroup2SD(Wg,H,payparam,beta,infocheck=True)
    # print('fixM')
    # print(fixM)
    # print('SD')
    # print(SD)
    # print('time spent: ',time.time()-t0)
#####################################################

####### Plot heatmap #########################################
    eps=0.01 #0.01
    Z=100
    N=9
    beta=1.
    M=0
    betaF=1

    f=0

    deltaLv=np.linspace(0,8,num=50)
    pSv=np.linspace(0.,1.,num=50)
    rv=[5,6,7]
    
    # labfilenpy='results/h4/ps/sfmodel_4strats_M0_dl8_f0_dfpsr'
    labfilenpy='results/h4/ps/heterogeneous_leader_M0_f0_dfdlrps_alt'
    # MAT=coop_pF_r(rv,M,N,Z,beta,eps,pSv,f,betaF,deltaLv)
    # np.save(labfilenpy,MAT)             # save matrix for heatmap
    # print('data saved to file!')
    
    MAT=np.load(labfilenpy+'.npy')      # load matrix for heatmap 
    plotCOOPheat(MAT,deltaLv,pSv,rv,labfilenpy)      # plot heatmap
    #plotsingleheat(MAT,fv,rv,labfilenpy)
#####################################################