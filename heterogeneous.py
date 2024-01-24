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



def calcWCD(N,eps,pF,delta_l,pS=0.5,M=0.):
# Input: N group size, eps error when trying to perform an action, r multiplicative constant for the PGG (assuming c=1), pF probability of following leader, M number of individuals that need to cooperate in order to get any benefit
# Output: WCD[i,k,ip] payoffs (i=0 defector, i=1 cooperator; k number of cooperators in the group; ip coef associated to the parameter payoffs r (ip=0) and c (ip=1))
    WCD=np.zeros((2,N+1,2))
    eps1=1.-eps
    pW = 1. - pS
    fw = 1 / (1+np.exp(delta_l))
    fs = 1 / (1+np.exp(-delta_l))
    for k in range(0,N+1): # k number of cooperators
        kw = pW*k
        Nw = pW*N
        ks = pS*k
        Ns = pS*N
        benefit_w = ((kw/Nw)*(
                eps1
                +(1-pF[0,0])*((kw-1)*eps1 + (Nw-kw)*eps)
                +(1-pF[1,0])*(ks*eps1 + (Ns-ks)*eps)
                +pF[0,0]*(Nw-1)*(eps1**2+eps**2)
                +pF[1,0]*Ns*(eps1**2+eps**2)
            )
            +(1-(kw/Nw))*(
                eps
                +(1-pF[0,0])*(kw*eps1 + (Nw-kw-1)*eps)
                +(1-pF[1,0])*(ks*eps1 + (Ns-ks)*eps)
                +pF[0,0]*(Nw-1)*(2*eps*eps1)
                +pF[1,0]*Ns*(2*eps*eps1)
            )
        )
        benefit_s = ((ks/Ns)*(
                eps1
                +(1-pF[0,1])*(kw*eps1+(Nw-kw)*eps)
                +(1-pF[1,1])*((ks-1)*eps1+(Ns-ks)*eps)
                +pF[0,1]*Nw*(eps1**2+eps**2)
                +pF[1,1]*(Ns-1)*(eps1**2+eps**2)
            )
            +(1-(ks/Ns))*(
                eps
                +(1-pF[0,1])*(kw*(1-eps)+(Nw-kw)*eps)
                +(1-pF[1,1])*(ks*eps1+(Ns-ks-1)*eps)
                +pF[0,1]*Nw*(2*eps*eps1)
                +pF[1,1]*(Ns-1)*(2*eps*eps1)
            )
        )

        benefit = ((Nw*fw)/(Nw*fw+Ns*fs))*benefit_w + ((Ns*fs)/(Nw*fw+Ns*fs))*benefit_s

        pfw_tilde = (Nw/N)*pF[0,0] + (Ns/N)*pF[0,1]
        pfs_tilde = (Nw/N)*pF[1,0] + (Ns/N)*pF[1,1]
        for i in [0,1]:    # i=1 cooperator, i=0 defector
            cost = ((1/N)*aeps(i,eps)
                    + (1-(1/N))*(
                        (Nw/N)*(
                            (1-pfw_tilde)*aeps(i,eps)
                            +pfw_tilde*(eps1*aeps((k-i)/(N-1),eps) + eps*aeps((N-k-1-i)/(N-1),eps))
                        )
                        +(Ns/N)*(
                            (1-pfs_tilde)*aeps(i,eps)
                            +pfs_tilde*(eps1*aeps((k-i)/(N-1),eps) + eps*aeps((N-k-1-i)/(N-1),eps))
                        )
                    )
            )

            if (benefit>=M): WCD[i,k,0]=benefit/N  # only if enough individuals cooperates
            WCD[i,k,1]=cost
        WCD[1,0,:]=-999
        WCD[0,N,:]=-999
    return WCD 


def coop_pF_r(rv,M,N,HZ,beta,eps,pS,deltaFv,betaF,fv):
# Input: pFv, rv, Mv (vectors with values of pF, r, and M), N, HZ (H or Z), beta, eps
# Output: matrix with the fraction of cooperators as a function of pF and r
    if np.isscalar(HZ):
        H=calcH(N-1,HZ-1)

    MAT = np.zeros((len(deltaFv), len(fv), len(rv)))

    for idr in range(len(rv)):
        r = rv[idr]
        for ideltaF in range(len(deltaFv)):
            deltaF=deltaFv[ideltaF]
            deltaL=deltaF
            for idf in range(len(fv)):
                f = fv[idf]
                pF=np.zeros((2,2))
                pF[0,0] = 1/(1+np.exp(-betaF*f))
                pF[1,1] = 1/(1+np.exp(-betaF*f))
                pF[0,1] = 1/(1+np.exp(-betaF*(f+deltaF)))
                pF[1,0] = 1/(1+np.exp(-betaF*(f-deltaF)))
                WCD=calcWCD(N,eps,pF,deltaL,pS,M)
                Wgen=transfW2Wgen(WCD) # transforming to evoEGT format
                print(r,deltaF,f)
                SD,fixM = evo.Wgroup2SD(Wgen,H,[r,-1.],beta,infocheck=False)
                MAT[ideltaF, idf, idr] = SD[1]
    return MAT

def plotCOOPheat(MAT,fv,rv,deltaFv,label):
# Input: MAT (matrix from "coop_pF_r" function), pFv, rv ,Mv (vectors with values of pF, r, and M), label (name for the output file)
# Output: heatmap plot of the fraction of cooperators as a function of pF and r, for different M
    import matplotlib.pyplot as plt
    fntsize=12
    nr=2
    nc=5
    f,axs=plt.subplots(nrows=nr, ncols=nc, sharex='all', sharey='all')
    f.subplots_adjust(hspace=0.2, wspace=0.2)
    k=-1
    for idx in range(len(rv)):
        i = idx // nc
        j = idx % nc

        ax=axs[i,j]
        k=k+1
        h=ax.imshow(MAT[:,:,k],origin='lower', interpolation='none',aspect='auto')
        nticksY=5
        nticksX=3
        ax.set_xticks(np.linspace(0, MAT.shape[1]-1, nticksX))
        ax.set_yticks(np.linspace(0, MAT.shape[0]-1, nticksY))
        ax.set_xticklabels(np.linspace(fv[0],fv[-1],nticksX))
        ax.set_yticklabels(np.linspace(deltaFv[0],deltaFv[-1],nticksY))
        ax.text(25,50,"$r=%s$" % str(rv[k]), size=10 )
        if i==nr-1: ax.set_xlabel(r'$f$', fontsize=fntsize)
        if j==0: ax.set_ylabel(r'$\Delta_f$', fontsize=fntsize)
#cb=f.colorbar(h, fraction=0.1,format='%.2f')
    #cb.set_label(label=r'$f_C$')
    f.savefig(label+'.eps',bbox_inches='tight',dpi=300)
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
    pS=0.2
    betaF=1

    rv=range(1,11)
    fv=np.linspace(-8,8,num=50)
    deltaFv=np.linspace(0,10,num=50)
    
    labfilenpy='coop_pF_r_M_N9'
    MAT=coop_pF_r(rv,M,N,Z,beta,eps,pS,deltaFv,betaF,fv)
    np.save(labfilenpy,MAT)             # save matrix for heatmap
    print('data saved to file!')
    
    MAT=np.load(labfilenpy+'.npy')      # load matrix for heatmap 
    label='heatCD_N9_s02'
    plotCOOPheat(MAT,fv,rv,deltaFv,label)      # plot heatmap
#####################################################
    
    