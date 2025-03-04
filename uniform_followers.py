#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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



def calcWCD(N,eps,pF=0.5,M=0.,cL=0):
# Input: N group size, eps error when trying to perform an action, r multiplicative constant for the PGG (assuming c=1), pF probability of following leader, M number of individuals that need to cooperate in order to get any benefit
# Output: WCD[i,k,ip] payoffs (i=0 defector, i=1 cooperator; k number of cooperators in the group; ip coef associated to the parameter payoffs r (ip=0) and c (ip=1))
    WCD=np.zeros((2,N+1,2))
    eps1=1.-eps
    for k in range(0,N+1): # k number of cooperators

        benefit = (
            (1 - pF) * (k*eps1 + (N-k)*eps)
            + pF * N * ((k/N)*(eps1**2+eps**2) + ((N-k)/N)*(2*eps*eps1))
        )

        for i in [0,1]:    # i=1 cooperator, i=0 defector
            cost = ((1-pF) * aeps(i, eps)
                    + pF*((k/N)*(eps1**2+eps**2) + ((N-k)/N)*(2*eps*eps1)))

            if (benefit>=M): WCD[i,k,0]=benefit/N  # only if enough individuals cooperate
            else:
                cost += (1-pF)*aeps(i,eps)*cL
            WCD[i,k,1]=cost
        WCD[1,0,:]=-999
        WCD[0,N,:]=-999
    return WCD 


def coop_pF_r(pFv,rv,M,N,HZ,beta,eps,cLv):
# Input: pFv, rv, Mv (vectors with values of pF, r, and M), N, HZ (H or Z), beta, eps
# Output: matrix with the fraction of cooperators as a function of pF and r
    if np.isscalar(HZ):
        H=calcH(N-1,HZ-1)
    npF=len(pFv)
    nr=len(rv)
    ncl=len(cLv)
    MAT=np.zeros((npF,nr,ncl))
    for icl in range(ncl):
        cL=cLv[icl]
        for ipF in range(npF):
            pF=pFv[ipF]
            WCD=calcWCD(N,eps,pF=pF,M=M,cL=cL)
            Wgen=transfW2Wgen(WCD) # transforming to evoEGT format
            for ir in range(nr):
                r=rv[ir]
                print(ipF,ir,pF,r)
                SD,fixM = evo.Wgroup2SD(Wgen,H,[r,-1.],beta,infocheck=False)
                MAT[ir,ipF,icl] = SD[1]
    return MAT

def plotCOOPheat(MAT,pFv,rv,cLv,label):
# Input: MAT (matrix from "coop_pF_r" function), pFv, rv ,Mv (vectors with values of pF, r, and M), label (name for the output file)
# Output: heatmap plot of the fraction of cooperators as a function of pF and r, for different M
    import matplotlib.pyplot as plt
    fntsize=12
    nr=5
    nc=2
    f,axs=plt.subplots(nrows=nr, ncols=nc, sharex='all', sharey='all')
    f.subplots_adjust(hspace=0.2, wspace=0.2)
    k=-1
    for i in range(nr):
        for j in range(nc):
            ax=axs[i,j]
            k=k+1
            h=ax.imshow(MAT[:,:,k],origin='lower', interpolation='none',aspect='auto')
            nticksY=5
            nticksX=5
            ax.set_xticks(np.linspace(0, MAT.shape[1]-1, nticksX))
            ax.set_yticks(np.linspace(0, MAT.shape[0]-1, nticksY))
            ax.set_xticklabels(np.linspace(pFv[0],pFv[-1],nticksX))
            ax.set_yticklabels(np.linspace(rv[0],rv[-1],nticksY))
            ax.text(33,40,"$c_L=%s$" % str(cLv[k]), size=10 )
            if i==nr-1: ax.set_xlabel(r'$p_F$', fontsize=fntsize)
            if j==0: ax.set_ylabel(r'$r$', fontsize=fntsize)
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
    # pFv=np.linspace(0.,1.,num=50)
    # rv=np.linspace(1.,2.*N,num=50)
    # Mv=[0,1,2,3,4,5,6,7,8,9]
    
    # labfilenpy='data_followers/uniform'
    # MAT=coop_pF_r(pFv,rv,Mv,N,Z,beta,eps)  # calculate matrix for heatmap
    # np.save(labfilenpy,MAT)             # save matrix for heatmap
    # print('data saved to file!')
    
    # MAT=np.load(labfilenpy+'.npy')      # load matrix for heatmap 
    # label='uniformfollowers'
    # plotCOOPheat(MAT,pFv,rv,Mv,label)      # plot heatmap
#####################################################
    
    eps=0.01 #0.01
    Z=100
    N=9
    beta=1.
    pFv=np.linspace(0.,1.,num=50)
    rv=np.linspace(1.,2.*N,num=50)
    M=N/2
    cLv = [1,2,3,4,5,6,7,8,9,10]
    
    labfilenpy='data_followers/uniform_cl'
    MAT=coop_pF_r(pFv,rv,M,N,Z,beta,eps,cLv)  # calculate matrix for heatmap
    np.save(labfilenpy,MAT)             # save matrix for heatmap
    print('data saved to file!')
    
    MAT=np.load(labfilenpy+'.npy')      # load matrix for heatmap 
    label='uniformfollowers_cl'
    plotCOOPheat(MAT,pFv,rv,cLv,label)      # plot heatmap
    
    