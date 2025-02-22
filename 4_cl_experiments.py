import numpy as np
import evoEGT as evo
from heterogeneous4 import calcH, calcWCD

def coop_pF_r(rv,M,N,HZ,beta,eps,pSv,deltaLv,f,betaF):
# Input: pFv, rv, Mv (vectors with values of pF, r, and M), N, HZ (H or Z), beta, eps
# Output: matrix with the fraction of cooperators as a function of pF and r
    if np.isscalar(HZ):
        H=calcH(N-1,HZ-1)

    MAT = np.zeros((len(rv), len(deltaLv), len(pSv), 4))

    for iddl, deltaL in enumerate(deltaLv):
        deltaF=deltaL
        for idps, pS in enumerate(pSv):
            pF=np.zeros((2,2))
            pF[0,0] = 1/(1+np.exp(-betaF*(f)))
            pF[1,1] = 1/(1+np.exp(-betaF*(f)))
            pF[0,1] = 1/(1+np.exp(-betaF*(f+deltaF)))
            pF[1,0] = 1/(1+np.exp(-betaF*(f-deltaF)))

            WCD=calcWCD(N,eps,pF,deltaL,pS,M)
            print(deltaL, pS)
            for idr, r, in enumerate(rv):
                SD,fixM = evo.Wgroup2SD(WCD,H,[r,-1.],beta,infocheck=False)
                MAT[idr, iddl, idps] = SD[:,0]
    return MAT

def plotCOOPheat(MAT,f1v,f2v,sv,label):
# Input: MAT (matrix from "coop_pF_r" function), pFv, rv ,Mv (vectors with values of pF, r, and M), label (name for the output file)
# Output: heatmap plot of the fraction of cooperators as a function of pF and r, for different M
    import matplotlib.pyplot as plt
    fntsize=12
    nr=5
    nc=2
    f,axs=plt.subplots(nrows=nr, ncols=nc, sharex='all', sharey='all')
    f.subplots_adjust(hspace=0.2, wspace=0.2)
    k=-1
    for idx in range(len(sv)):
        i = idx // nc
        j = idx % nc

        ax=axs[i,j]
        k=k+1
        h=ax.imshow(MAT[:,:,k],origin='lower', interpolation='none',aspect='auto',vmin=0,vmax=4)
        nticksY=5
        nticksX=3
        ax.set_xticks(np.linspace(0, MAT.shape[1]-1, nticksX))
        ax.set_yticks(np.linspace(0, MAT.shape[0]-1, nticksY))
        ax.set_xticklabels(np.linspace(f2v[0],f2v[-1],nticksX))
        ax.set_yticklabels(np.linspace(f2v[0],f2v[-1],nticksY))
        ax.text(25,50,"$S=%.2f$" % sv[k], size=10 )
        if i==nr-1: ax.set_xlabel(r'$p_{F_{ww,ss}}$', fontsize=fntsize)
        if j==0: ax.set_ylabel(r'$p_{F_{ws}}$', fontsize=fntsize)
    
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = f.colorbar(h, cax=cbar_ax)
    cb.set_ticks([1,2,3,4])
    cb.set_ticklabels(['ALLD','WCSD','WDSC','ALLC'])

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

####### Plot heatmap #########################################
    eps=0.01 #0.01
    Z=100
    N=9
    beta=1.
    M=0
    f=0
    betaF=1

    deltaLv=[0, 1, 2, 4, 8]
    pSv=np.linspace(0,1.,num=50)
    rv=np.linspace(1,10,num=10)
    
    labfilenpy='newtests/res_2bits_singleleader'
    MAT=coop_pF_r(rv,M,N,Z,beta,eps,pSv,deltaLv,f,betaF)
    np.save(labfilenpy,MAT)             # save matrix for heatmap
    print('data saved to file!')
    
    # MAT=np.load(labfilenpy+'.npy')      # load matrix for heatmap 
    # plotCOOPheat(MAT,f1v,f2v,sv,labfilenpy)      # plot heatmap
    #plotsingleheat(MAT,fv,rv,labfilenpy)
#####################################################