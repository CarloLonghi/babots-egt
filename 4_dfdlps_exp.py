import numpy as np
import evoEGT as evo
from heterogeneous4_leaderstrat import calcH, calcWCD

def coop_pF_r(rv,M,N,HZ,beta,eps,pSv,f,betaF,deltav):
# Input: pFv, rv, Mv (vectors with values of pF, r, and M), N, HZ (H or Z), beta, eps
# Output: matrix with the fraction of cooperators as a function of pF and r
    if np.isscalar(HZ):
        H=calcH(N-1,HZ-1)

    MAT = np.zeros((len(deltav), len(pSv), len(rv), 4))

    for iddf, deltaF in enumerate(deltav):
        deltaL = deltaF
        pF=np.zeros((2,2))
        pF[0,0] = 1/(1+np.exp(-betaF*(f)))
        pF[1,1] = 1/(1+np.exp(-betaF*(f)))
        pF[0,1] = 1/(1+np.exp(-betaF*(f+deltaF)))
        pF[1,0] = 1/(1+np.exp(-betaF*(f-deltaF)))
        for idps, pS in enumerate(pSv):
            WCD=calcWCD(N,eps,pF,deltaL,pS,M)
            print(deltaF,pS)
            for idr, r in enumerate(rv):
                SD,fixM = evo.Wgroup2SD(WCD,H,[r,-1.],beta,infocheck=False)
                best_s = np.argmax(SD)
                if SD[best_s] >= 0.5:
                    MAT[iddf, idps, idr, best_s] = SD[best_s]
    return MAT

def plotCOOPheat(MAT,deltaFv,pSv,rv,label):
# Input: MAT (matrix from "coop_pF_r" function), pFv, rv ,Mv (vectors with values of pF, r, and M), label (name for the output file)
# Output: heatmap plot of the fraction of cooperators as a function of pF and r, for different M
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib
    fntsize=20
    nr=2
    nc=5
    f,axs=plt.subplots(nrows=nr, ncols=nc, sharex='all', sharey='all', figsize=(15,5))
    f.subplots_adjust(hspace=0.4, wspace=0.2)
    k=-1
    for idx in range(len(rv)):
        i = idx // nc
        j = idx % nc

        ax=axs[i,j]
        k=k+1
        cmaps=['Greens','Reds','Blues','Purples']
        for strat in range(4):
            step=0.025
            levels = np.arange(0.5-step, 1., step) + step
            h=ax.contourf(MAT[:,:,idx,strat],levels,cmap=cmaps[strat], origin='lower',)
        #h=ax.imshow(MAT[:,:,k],origin='lower', interpolation='none',aspect='auto',vmin=0,vmax=4)
        nticksY=5
        nticksX=3
        ax.set_xticks(np.linspace(0, MAT.shape[1]-1, nticksX))
        ax.set_yticks(np.linspace(0, MAT.shape[0]-1, nticksY))
        ax.set_xticklabels(np.linspace(pSv[0],pSv[-1],nticksX), fontsize=18)
        ax.set_yticklabels(np.linspace(deltaFv[0],deltaFv[-1],nticksY), fontsize=18)
        ax.set_ylim(0,MAT.shape[1]-1)
        ax.text(22.5,50,"$r=%d$" % rv[idx], size=20)
        if i==nr-1: ax.set_xlabel(r'$p_s$', fontsize=fntsize)
        if j==0: 
            ax.set_ylabel(r'$\Delta_f, \Delta_l$', fontsize=fntsize)

        # insert markers for invasion graphs
        # if idx == 0:
        #     points_x = [MAT.shape[1] // 2 - 0.5, MAT.shape[1] // 2 - 0.5, MAT.shape[1] // 2 - 0.5]
        #     points_y = [0.5, MAT.shape[0] // 8 - 0.5, MAT.shape[0] // 4 - 0.5]
        #     ax.scatter(points_x, points_y, color='goldenrod', marker='o')
    
    labels = ['ALLD', 'NCLD', 'NDLC', 'ALLC']
    patches = [mpatches.Patch(color=plt.get_cmap(cmaps[i])(0.9), label=labels[i]) for i in range(4)]
    plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(-1.9 , -0.5),
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
    f.savefig('newtests/2bits/leadstrat/singleleader/multisd.png',bbox_inches='tight',dpi=300)
    #plt.show()
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
    betaF=1

    f=0

    deltaLv=np.linspace(0,8,num=50)
    pSv=np.linspace(0.,1.,num=50)
    rv=np.linspace(1, 10, 10)
    
    # labfilenpy='results/h4/ps/sfmodel_4strats_M0_dl8_f0_dfpsr'
    labfilenpy='newtests/2bits/leadstrat/singleleader/res_2bits_singleleader_deltaps'
    MAT=coop_pF_r(rv,M,N,Z,beta,eps,pSv,f,betaF,deltaLv)
    np.save(labfilenpy,MAT)             # save matrix for heatmap
    print('data saved to file!')
    
    MAT=np.load(labfilenpy+'.npy')      # load matrix for heatmap 
    plotCOOPheat(MAT,deltaLv,pSv,rv,labfilenpy)      # plot heatmap
    #plotsingleheat(MAT,fv,rv,labfilenpy)
#####################################################