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

def fl(sl,eps):
    eps1 = 1-eps
    if sl == 0:
        return 2*eps*eps1
    else:
        return eps1**2+eps**2

def calcWCD(N,eps,pF,deltaL,pS,M):
# Input: N group size, eps error when trying to perform an action, r multiplicative constant for the PGG (assuming c=1), pF probability of following leader, M number of individuals that need to cooperate in order to get any benefit
# Output: WCD[i,k,ip] payoffs (i=0 defector, i=1 cooperator; k number of cooperators in the group; ip coef associated to the parameter payoffs r (ip=0) and c (ip=1))
    WCD=np.zeros((4,4,N+1,2))
    eps1=1.-eps

    pW = 1. - pS
    pleadS = 1 / (1 + np.exp(-deltaL))
    pleadW = 1 - pleadS
    Nw = N*pW
    Ns = N*pS

    for i in range(4):
        s1=[i%2,i//2] # s:[w,s], 0:[0,0], 1:[1,0], 2:[0,1], 3:[1,1]
        for j in range(4):
            s2=[j%2,j//2]
            for k in range(0,N+1): # k: number of players following strategy s1

                Nwc = pW*(k*s1[0]+(N-k)*s2[0])
                Nsc = pS*(k*s1[1]+(N-k)*s2[1])
                Nwd = N*pW-Nwc
                Nsd = N*pS-Nsc

                Nwcl = Nwc * pleadW
                Nscl = Nsc * pleadS
                Nwdl = Nwd * pleadW
                Nsdl = Nsd * pleadS
                Nwl = Nwcl + Nwdl
                Nsl = Nscl + Nsdl

                pwc = 0; pwd = 0; psc = 0; psd = 0; pwl = 0; psl = 0
                if Nw > 0:
                    pwc = Nwc / Nw
                    pwd = 1 - pwc
                    pwl = Nwl / Nw
                if Ns > 0:
                    psc = Nsc / Ns
                    psd = 1 - psc
                    psl = Nsl / Ns

                follow_s = (pleadS * Nsl) / (pleadS * Nsl + pleadW * Nwl)
                follow_w = (pleadW * Nwl) / (pleadS * Nsl + pleadW * Nwl)

                benefit = (
                    (Nwcl + Nscl) * eps1 + (Nwdl + Nsdl) * eps + # leaders
                    (N - Nsl - Nwl) * ( # non leaders
                        pW * ( # weak
                            follow_w * ( # choose a weak leader
                                (1 - pF[0, 0]) * (pwc * eps1 + pwd * eps) + # not follow
                                (pF[0, 0] * (pwc * (eps1**2 + eps**2) + pwd * (2*eps1*eps))) # follow
                            ) + 
                            follow_s * ( # choose a strong leader
                                (1 - pF[0, 1]) * (pwc * eps1 + pwd * eps) +
                                (pF[0, 1] * (psc * (eps1**2 + eps**2) + psd * (2*eps1*eps)))
                            )
                        ) +
                        pS * ( #strong
                            follow_w * ( # choose a weak leader
                                (1 - pF[1, 0]) * (psc * eps1 + psd * eps) + # not follow
                                (pF[1, 0] * (pwc * (eps1**2 + eps**2) + pwd * (2*eps1*eps))) # follow
                            ) + 
                            follow_s * ( # choose a strong leader
                                (1 - pF[1, 1]) * (psc * eps1 + psd * eps) +
                                (pF[1, 1] * (psc * (eps1**2 + eps**2) + psd * (2*eps1*eps)))
                            )                            
                        )
                    )
                )

                cost = (
                    pW * ( # focus player is weak
                        pwl * aeps(s1[0], eps) + # focus playes is a leader
                        (1 - pwl) * ( # is not a leader
                            follow_w * ( # choose weak leader
                                (1 - pF[0, 0]) * aeps(s1[0], eps) +
                                pF[0, 0] * (pwc * (eps1**2 + eps**2) + pwd * (2*eps1*eps))
                            ) + 
                            follow_s * (
                                (1 - pF[0, 1]) * aeps(s1[0], eps) +
                                pF[0, 1] * (psc * (eps1**2 + eps**2) + psd * (2*eps1*eps))                                
                            )
                        )
                    ) +
                    pS * ( # focus player is strong
                        psl * aeps(s1[1], eps) + # focus playes is a leader
                        (1 - psl) * ( # is not a leader
                            follow_w * ( # choose weak leader
                                (1 - pF[1, 0]) * aeps(s1[1], eps) +
                                pF[1, 0] * (pwc * (eps1**2 + eps**2) + pwd * (2*eps1*eps))
                            ) + 
                            follow_s * (
                                (1 - pF[1, 1]) * aeps(s1[1], eps) +
                                pF[1, 1] * (psc * (eps1**2 + eps**2) + psd * (2*eps1*eps))                                
                            )
                        )
                    )
                )                

                if benefit > M:
                    WCD[i,j,k,0] = benefit/N

                WCD[i,j,k,1] = cost
                
    # WCD[1,0,:]=-999
    # WCD[0,N,:]=-999

    return WCD 
    
def gaussian(x, mu, sig):
    return np.exp(-((x - mu) ** 2) / (2 * (sig ** 2))) / (np.sqrt(2 * np.pi) * sig)
