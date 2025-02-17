import numpy as np
import evoEGT as evo
from heterogeneous4 import calcH, calcWCD

def coop_pF_r(rv,M,N,HZ,beta,eps,pSv,f,betaF,deltaFv):
# Input: pFv, rv, Mv (vectors with values of pF, r, and M), N, HZ (H or Z), beta, eps
# Output: matrix with the fraction of cooperators as a function of pF and r
    if np.isscalar(HZ):
        H=calcH(N-1,HZ-1)

    MAT = np.zeros((len(deltaFv), len(pSv), len(rv), 4))

    for iddf, deltaF in enumerate(deltaFv):
        pF=np.zeros((2,2))

        pF[0,0] = 1/(1+np.exp(-betaF*(f)))
        pF[1,1] = 1/(1+np.exp(-betaF*(f)))
        pF[0,1] = 1/(1+np.exp(-betaF*(f+deltaF)))
        pF[1,0] = 1/(1+np.exp(-betaF*(f-deltaF)))

        deltaL = deltaF
        for idps, pS in enumerate(pSv):
            WCD=calcWCD(N,eps,pF,deltaL,pS,M)
            print(deltaF, pS)
            for idr, r in enumerate(rv):
                SD,fixM = evo.Wgroup2SD(WCD,H,[r,-1.],beta,infocheck=False)
                MAT[iddf, idps, idr, :] = SD[:,0]
    return MAT

def plotCOOPheat(MAT,deltaFv,pSv,r,label):
# Input: MAT (matrix from "coop_pF_r" function), pFv, rv ,Mv (vectors with values of pF, r, and M), label (name for the output file)
# Output: heatmap plot of the fraction of cooperators as a function of pF and r, for different M
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    fntsize=13
    nr=2
    nc=2
    f,axs=plt.subplots(nrows=nr, ncols=nc, sharex='all', sharey='all', figsize=(5,5))
    f.subplots_adjust(hspace=0.2, wspace=0.2)
    labels = ['ALLD', 'WCSD', 'WDSC', 'ALLC']
    for strat in range(4):
        i = strat // nc
        j = strat % nc

        ax=axs[i,j]
        cmaps=['Greens','Reds','Blues','Purples']
        step=0.025
        levels = np.arange(0, 1., step) + step
        h=ax.contourf(MAT[:,:,r-1,strat],levels,cmap=cmaps[strat], origin='lower',)
        #h=ax.imshow(MAT[:,:,k],origin='lower', interpolation='none',aspect='auto',vmin=0,vmax=4)
        nticksY=5
        nticksX=3
        ax.set_xticks(np.linspace(0, MAT.shape[1]-1, nticksX))
        ax.set_yticks(np.linspace(0, MAT.shape[0]-1, nticksY))
        ax.set_xticklabels(np.linspace(pSv[0],pSv[-1],nticksX), fontsize=10)
        ax.set_yticklabels(np.linspace(deltaFv[0],deltaFv[-1],nticksY), fontsize=10)
        ax.text(17.5,50,labels[strat], size=fntsize)
        if i==nr-1: ax.set_xlabel(r'$p_s$', fontsize=fntsize)
        if j==0: ax.set_ylabel(r'$\Delta_f, \Delta_l$', fontsize=fntsize)
    
    f.savefig(f'./newtests/2bits/multileader/singlestrat_2bits_multileader_r{r}.png',bbox_inches='tight',dpi=300)
    plt.show()
    return

# def plotsingleheat(MAT,fv,rv,label):
# # Input: MAT (matrix from "coop_pF_r" function), pFv, rv ,Mv (vectors with values of pF, r, and M), label (name for the output file)
# # Output: heatmap plot of the fraction of cooperators as a function of pF and r, for different M
#     import matplotlib.pyplot as plt
#     fntsize=12
#     f,ax=plt.subplots()
#     h=ax.imshow(MAT,origin='lower', interpolation='none',aspect='auto')
#     nticksY=5
#     nticksX=3
#     ax.set_xticks(np.linspace(0, MAT.shape[1]-1, nticksX))
#     ax.set_yticks(np.linspace(0, MAT.shape[0]-1, nticksY))
#     ax.set_xticklabels(np.linspace(fv[0],fv[-1],nticksX))
#     ax.set_yticklabels(np.linspace(rv[0],rv[-1],nticksY))
#     ax.set_xlabel(r'$f$', fontsize=fntsize)
#     ax.set_ylabel(r'$r$', fontsize=fntsize)
# #cb=f.colorbar(h, fraction=0.1,format='%.2f')
#     #cb.set_label(label=r'$f_C$')
#     f.savefig('multileader_single_strategies_r5.png',bbox_inches='tight',dpi=300)
#     f.clf()     
#     return


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

    deltaFv=np.linspace(0,8,num=50)
    pSv=np.linspace(0.,1.,num=50)
    rv=np.linspace(1, 10, 10)
    
    # labfilenpy='results/h4/ps/sfmodel_4strats_M0_dl8_f0_dfpsr'
    labfilenpy='./newtests/2bits/multileader/singlestrat_2bits_multileader'
    MAT=coop_pF_r(rv,M,N,Z,beta,eps,pSv,f,betaF,deltaFv)
    np.save(labfilenpy,MAT)             # save matrix for heatmap
    print('data saved to file!')
    
    MAT=np.load(labfilenpy+'.npy')      # load matrix for heatmap 
    plotCOOPheat(MAT,deltaFv,pSv,10,labfilenpy)      # plot heatmap
    #plotsingleheat(MAT,fv,rv,labfilenpy)
#####################################################