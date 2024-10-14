import numpy as np
import evoEGT as evo
from heterogeneous4 import calcH, calcWCD

def coop_pF_r(rv,M,N,HZ,beta,eps,pSv,fv,betaF,deltaF,deltaL):
# Input: pFv, rv, Mv (vectors with values of pF, r, and M), N, HZ (H or Z), beta, eps
# Output: matrix with the fraction of cooperators as a function of pF and r
    if np.isscalar(HZ):
        H=calcH(N-1,HZ-1)

    MAT = np.zeros((len(rv), len(fv), len(pSv), 4))

    for idf, f in enumerate(fv):
        pF=np.zeros((2,2))
        # pF[0,0] = 1/(1+np.exp(-betaF*(f+deltaL)))
        # pF[1,1] = 1/(1+np.exp(-betaF*(f-deltaL)))
        # pF[0,1] = 1/(1+np.exp(-betaF*(f+deltaL+deltaF)))
        # pF[1,0] = 1/(1+np.exp(-betaF*(f-deltaL-deltaF)))
        pF[0,0] = 1/(1+np.exp(-betaF*(f)))
        pF[1,1] = 1/(1+np.exp(-betaF*(f)))
        pF[0,1] = 1/(1+np.exp(-betaF*(f+deltaF)))
        pF[1,0] = 1/(1+np.exp(-betaF*(f-deltaF)))
        for idr, r in enumerate(rv):
            for idps, pS in enumerate(pSv):
                WCD=calcWCD(N,eps,pF,deltaL,pS,M)
                #Wgen=transfW2Wgen(WCD) # transforming to evoEGT format
                print(f,r,pS)
                SD,fixM = evo.Wgroup2SD(WCD,H,[r,-1.],beta,infocheck=False)
                best_s = np.argmax(SD)
                if SD[best_s] >= 0.5:
                    #MAT[idr, idf, idps] = best_s+1
                    MAT[idr, idf, idps, best_s] = SD[best_s]
    return MAT

def plotCOOPheat(MAT,fv,pSv,rv,label):
# Input: MAT (matrix from "coop_pF_r" function), pFv, rv ,Mv (vectors with values of pF, r, and M), label (name for the output file)
# Output: heatmap plot of the fraction of cooperators as a function of pF and r, for different M
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    fntsize=12
    nr=5
    nc=2
    f,axs=plt.subplots(nrows=nr, ncols=nc, sharex='all', sharey='all')
    f.subplots_adjust(hspace=0.2, wspace=0.2)
    k=-1
    for idx in range(len(pSv)):
        i = idx // nc
        j = idx % nc

        ax=axs[i,j]
        k=k+1
        cmaps=['Greens','Reds','Blues','Purples']
        for strat in range(4):
            step=0.025
            levels = np.arange(0.4, 1., step) + step
            h=ax.contourf(MAT[:,:,k,strat],levels,cmap=cmaps[strat], origin='lower',)
        #h=ax.imshow(MAT[:,:,k],origin='lower', interpolation='none',aspect='auto',vmin=0,vmax=4)
        nticksY=5
        nticksX=3
        ax.set_xticks(np.linspace(0, MAT.shape[1]-1, nticksX))
        ax.set_yticks(np.linspace(0, MAT.shape[0]-1, nticksY))
        ax.set_xticklabels(np.linspace(fv[0],fv[-1],nticksX))
        ax.set_yticklabels(np.linspace(rv[0],rv[-1],nticksY))
        ax.text(25,50,"$p_S=%.2f$" % pSv[k], size=10 )
        if i==nr-1: ax.set_xlabel(r'$f$', fontsize=fntsize)
        if j==0: ax.set_ylabel(r'$r$', fontsize=fntsize)
    
    labels = ['ALLD', 'WCSD', 'WDSC', 'ALLC']
    patches = [mpatches.Patch(color=plt.get_cmap(cmaps[i])(0.9), label=labels[i]) for i in range(4)]
    plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(-0.1, -.8),
          fancybox=True, shadow=False, ncol=4)

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
    f.savefig(label+'.png',bbox_inches='tight',dpi=300)
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
    f.savefig(label+'.png',bbox_inches='tight',dpi=300)
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

    deltaL=0
    deltaF=0

    fv=np.linspace(0,4.,num=50)
    #deltaFv=[0,1,2,4,6,8]
    pSv=np.linspace(0.01,.99,num=10)
    rv=np.linspace(6.,10.,num=50)
    
    # labfilenpy='results/h4/ps/sfmodel_4strats_M0_dl8_f0_dfpsr'
    labfilenpy='results/h4/ps/heterogeneous_leader_M0_df4_dl4_f_ps_r'
    # MAT=coop_pF_r(rv,M,N,Z,beta,eps,pSv,fv,betaF,deltaF,deltaL)
    # np.save(labfilenpy,MAT)             # save matrix for heatmap
    # print('data saved to file!')
    
    MAT=np.load(labfilenpy+'.npy')      # load matrix for heatmap 
    plotCOOPheat(MAT,fv,pSv,rv,labfilenpy)      # plot heatmap
    #plotsingleheat(MAT,fv,rv,labfilenpy)
#####################################################