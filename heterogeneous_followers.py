#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:40:35 2019

@author: abraxas
"""

import evoEGT as evo

import numpy as np


def calcH(N,Z):   
# Input: N group size, Z population size
# Output: H[k,K] hypergeometric function (k individuals in group, K individuals in population)
    import numpy as np
    from scipy.stats import hypergeom  
    H=np.zeros((N+1,Z+1))
    for K in range(0,Z+1):         
        for k in range(0,N+1):
            H[k,K]=hypergeom.pmf(k,Z,K,N)
    return H

def aeps(pact,eps):
# Input: pact probability to perform the desired action (without error), eps probability of comitting an error 
# Output: action actually performed 
    return pact*(1.-2.*eps)+eps


def transfW2Wgen(Wori):
# transform WCD (Wori) into W for calcWpop (Wgen) 
    N1=Wori.shape[1]
    Wgen=np.zeros((2,2,N1,2))-777
    Wgen[1,0,:,:]=Wori[1,:,:]
    Wgen[0,1,:,:]=np.flip(Wori[0,:,:],axis=0)
    return Wgen



def calcWCD(N,eps,pS,M,pFW,pFS,bL):
# Input: N group size, eps error when trying to perform an action, r multiplicative constant for the PGG (assuming c=1), pF probability of following leader, M number of individuals that need to cooperate in order to get any benefit
# Output: WCD[i,k,ip] payoffs (i=0 defector, i=1 cooperator; k number of cooperators in the group; ip coef associated to the parameter payoffs r (ip=0) and c (ip=1))
    WCD=np.zeros((2,N+1,2))
    eps1=1.-eps
    pW = 1. - pS
    for k in range(0,N+1): # k number of cooperators
        kw = pW*k
        Nw = pW*N
        ks = pS*k
        Ns = pS*N

        benefit = ((Nw/N)*((1-pFW)*(k*eps1+(N-k)*eps)
                          +pFW*N*((k/N)*(eps1**2+eps**2)+((N-k)/N)*(2*eps*eps1)))
                    +(Ns/N)*((1-pFS)*(k*eps1+(N-k)*eps)
                            +pFS*N*((k/N)*(eps1**2+eps**2)+((N-k)/N)*(2*eps*eps1))))

        for i in [0,1]:    # i=1 cooperator, i=0 defector

            cost = ((Nw/N)*((1-pFW)*aeps(i,eps)+pFW*((k/N)*(eps1**2+eps**2)+((N-k)/N)*(2*eps*eps1)))
                    +(Ns/N)*((1-pFS)*aeps(i,eps)+pFS*((k/N)*(eps1**2+eps**2)+((N-k)/N)*(2*eps*eps1))))

            if (benefit>=M): 
                WCD[i,k,0]=benefit/N  # only if enough individuals cooperates
                cost -= (Ns/N)*(1-pFS)*bL
            WCD[i,k,1]=cost
        WCD[1,0,:]=-999
        WCD[0,N,:]=-999
    return WCD 


def coop_pF_r(r,M,N,HZ,beta,eps,pS,pFWv,pFSv,bLv):
# Input: pFv, rv, Mv (vectors with values of pF, r, and M), N, HZ (H or Z), beta, eps
# Output: matrix with the fraction of cooperators as a function of pF and r
    if np.isscalar(HZ):
        H=calcH(N-1,HZ-1)

    MAT = np.zeros((len(pFWv), len(pFSv), len(bLv)))

    for idbl, bL in enumerate(bLv):
        for idpfw, pFW in enumerate(pFWv):
            for idpfs, pFS in enumerate(pFSv):
                if pFS <= pFW:
                    WCD=calcWCD(N,eps,pS,M,pFW,pFS,bL)
                    Wgen=transfW2Wgen(WCD) # transforming to evoEGT format
                    print(bL,pFW,pFS)
                    SD,fixM = evo.Wgroup2SD(Wgen,H,[r,-1.],beta,infocheck=False)
                    MAT[idpfw, idpfs, idbl] = SD[1]
    return MAT

def plotCOOPheat(MAT,bLv,pFSv,pFWv,label):
# Input: MAT (matrix from "coop_pF_r" function), pFv, rv ,Mv (vectors with values of pF, r, and M), label (name for the output file)
# Output: heatmap plot of the fraction of cooperators as a function of pF and r, for different M
    import matplotlib.pyplot as plt
    fntsize=12
    nr=3
    nc=4
    f,axs=plt.subplots(nrows=nr, ncols=nc, sharex='all', sharey='all', figsize=(15,15))
    f.subplots_adjust(hspace=0.2, wspace=0.2)
    k=-1
    for idx in range(len(bLv)):
        i = idx // nc
        j = idx % nc

        ax=axs[i,j]
        k=k+1
        h=ax.imshow(MAT[:,:,k],origin='lower', interpolation='none',aspect='auto')
        nticksY=5
        nticksX=3
        ax.set_xticks(np.linspace(0, MAT.shape[1]-1, nticksX))
        ax.set_yticks(np.linspace(0, MAT.shape[0]-1, nticksY))
        ax.set_xticklabels(np.linspace(pFSv[0],pFSv[-1],nticksX))
        ax.set_yticklabels(np.linspace(pFWv[0],pFWv[-1],nticksY))
        ax.text(20,40,"$B_L=%.1f$" % bLv[k], size=20 )
        if i==nr-1: ax.set_xlabel(r'$p_FS$', fontsize=fntsize)
        if j==0: ax.set_ylabel(r'$p_FW$', fontsize=fntsize)
#cb=f.colorbar(h, fraction=0.1,format='%.2f')
    #cb.set_label(label=r'$f_C$')
    f.savefig('data_followers/'+label+'.eps',bbox_inches='tight',dpi=300)
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
    # eps=0.01 #0.01
    # Z=100
    # N=9
    # beta=1.
    # M=N/2
    # betaF=1
    # r = 5

    # pSv=np.linspace(0,1.,num=11)
    # pFSv=np.linspace(0,.5,num=50)
    # pFWv=np.linspace(0.5,1.,num=50)
    
    # labfilenpy='data_followers/heterogeneous_r1'
    # MAT=coop_pF_r(r,M,N,Z,beta,eps,pSv,pFWv,pFSv)
    # np.save(labfilenpy,MAT)             # save matrix for heatmap
    # print('data saved to file!')
    
    # MAT=np.load(labfilenpy+'.npy')      # load matrix for heatmap 
    # label='heterogeneous_r1'
    # plotCOOPheat(MAT,pSv,pFSv,pFWv,label)      # plot heatmap
#####################################################
    
    eps=0.01 #0.01
    Z=100
    N=9
    beta=1.
    M=N/2
    betaF=1
    r = 5
    pS = 0.5

    bLv=np.linspace(0,1.,num=10)
    pFSv=np.linspace(0,.5,num=50)
    pFWv=np.linspace(0.5,1.,num=50)
    
    labfilenpy='data_followers/heterogeneous_r5_ps05_bl'
    MAT=coop_pF_r(r,M,N,Z,beta,eps,pS,pFWv,pFSv,bLv)
    np.save(labfilenpy,MAT)             # save matrix for heatmap
    print('data saved to file!')
    
    MAT=np.load(labfilenpy+'.npy')      # load matrix for heatmap 
    label='heterogeneous_r5_ps05_bl'
    plotCOOPheat(MAT,bLv,pFSv,pFWv,label)      # plot heatmap
    
    