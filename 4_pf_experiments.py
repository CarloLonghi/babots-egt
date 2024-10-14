import numpy as np
import evoEGT as evo
from heterogeneous2 import calcH, calcWCD

def coop_pF_r(r,M,N,HZ,beta,eps,pS,f1v,f2v,sv):
# Input: pFv, rv, Mv (vectors with values of pF, r, and M), N, HZ (H or Z), beta, eps
# Output: matrix with the fraction of cooperators as a function of pF and r
    if np.isscalar(HZ):
        H=calcH(N-1,HZ-1)

    MAT = np.zeros((len(f1v), len(f2v), len(sv)))

    for idf1, f1 in enumerate(f1v):
        for idf2, f2 in enumerate(f2v):
            for ids, s in enumerate(sv):
                pF=np.zeros((2,2))
                # pF[0,0] = 1/(1+np.exp(-betaF*f))
                # pF[1,1] = 1/(1+np.exp(-betaF*f))
                # pF[0,1] = 1/(1+np.exp(-betaF*(f+deltaF)))
                # pF[1,0] = 1/(1+np.exp(-betaF*(f-deltaF)))
                pF[0,0] = f2
                pF[0,1] = f1
                pF[1,0] = 0
                pF[1,1] = f2
                WCD=calcWCD(N,eps,pF,s,pS,M)
                #Wgen=transfW2Wgen(WCD) # transforming to evoEGT format
                print(f1,r,pS)
                SD,fixM = evo.Wgroup2SD(WCD,H,[r,-1.],beta,infocheck=False)
                best_s = np.argmax(SD)
                if SD[best_s] >= 0.5:
                    MAT[idf1, idf2, ids] = best_s+1
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

    f1v=np.linspace(0,1.,num=50)
    f2v=np.linspace(0,1.,num=50)
    sv=np.linspace(.5,1.,num=10)
    pS=0.6
    r=9
    
    labfilenpy='results/h4/pf/heterogeneous_leader_pf0100_1001_ps06_r9_M0_N9_f1f1s_2strats'
    MAT=coop_pF_r(r,M,N,Z,beta,eps,pS,f1v,f2v,sv)
    np.save(labfilenpy,MAT)             # save matrix for heatmap
    print('data saved to file!')
    
    MAT=np.load(labfilenpy+'.npy')      # load matrix for heatmap
    plotCOOPheat(MAT,f1v,f2v,sv,labfilenpy)      # plot heatmap
    #plotsingleheat(MAT,fv,rv,labfilenpy)
#####################################################