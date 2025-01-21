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
    WCD=np.zeros((16,16,N+1,2))
    eps1=1.-eps

    pW = 1. - pS
    pleadS = 1 / (1 + np.exp(-deltaL))
    pleadW = 1 - pleadS
    Nw = N*pW
    Ns = N*pS

    for i in range(16):
        s1=[i%8%4%2//1, i%8%4//2, i%8//4, i//8] # s:[LW, LS, nLW, nLS]
        for j in range(16):
            s2=[j%8%4%2//1, j%8%4//2, j%8//4, j//8]
            for k in range(0,N+1): # k: number of players following strategy s1

                # Nwc = pW * (k * (pleadW * s1[0] + (1 - pleadW) * s1[2]) + 
                #             (N - k) * (pleadW * s2[0] + (1 - pleadW) * s2[2]))
                # Nsc = pS * (k * (pleadS * s1[1] + (1 - pleadS) * s1[3]) +
                #             (N - k) * (pleadS * s2[1] + (1 - pleadS) * s2[3]))

                Nwcl = pW * (pleadW * (k * s1[0] + (N - k) * s2[0]))
                Nscl = pS * (pleadS * (k * s1[1] + (N - k) * s2[1]))
                Nwdl = pW * (pleadW * (k * (1 - s1[0]) + (N - k)* (1 - s2[0])))
                Nsdl = pS * (pleadS * (k * (1 - s1[1]) + (N - k) * (1 - s2[1])))

                Nwcnl = pW * ((1 - pleadW) * (k * s1[2] + (N - k) * s2[2]))
                Nscnl = pS * ((1 - pleadS) * (k * s1[3] + (N - k) * s2[3]))
                Nwdnl = pW * ((1 - pleadW) * (k * (1 - s1[2]) + (N - k)* (1 - s2[2])))
                Nsdnl = pS * ((1 - pleadS) * (k * (1 - s1[3]) + (N - k) * (1 - s2[3])))   

                Nwl = Nwcl + Nwdl
                Nsl = Nscl + Nsdl
                Nwc = Nwcl + Nwcnl
                Nsc = Nscl + Nscnl

                pwc = 0; pwd = 0; psc = 0; psd = 0; pwcl = 0; pwdl = 0; pscl = 0; psdl = 0; pwl = 0; psl = 0
                if Nw > 0:
                    pwc = Nwcnl / (Nwcnl+Nwdnl)
                    pwd = 1 - pwc
                    pwcl = Nwcl / Nwl
                    pwdl = Nwdl / Nwl
                    pwl = Nwl / Nw
                if Ns > 0:
                    psc = Nscnl / (Nscnl+Nsdnl)
                    psd = 1 - psc
                    pscl = Nscl / Nsl
                    psdl = Nsdl / Nsl
                    psl = Nsl / Ns

                follow_s = (pleadS * Nsl) / (pleadS * Nsl + pleadW * Nwl)
                follow_w = (pleadW * Nwl) / (pleadS * Nsl + pleadW * Nwl)
                p1leader = 1 - (((1 - pleadW)**Nw) * ((1 - pleadS)**Ns)) # prob that there is at least 1 leader

                # benefit = (
                #     pW*( # weak players
                #         pleadW*((k * s1[0] + (N - k) * s2[0])*eps1 + (k * (1 - s1[0]) + (N - k)* (1 - s2[0]))*eps)+ #leaders
                #         (1-pleadW)*( # non-leaders
                #             p1leader*( # find at least one leader
                #                 follow_w*( # choose a weak leader
                #                     (1-pF[0,0])*((k * s1[2] + (N - k) * s2[2])*eps1 + (k * (1 - s1[2]) + (N - k)* (1 - s2[2]))*eps)+
                #                     pF[0,0]
                #                 )
                #             )+
                #             (1-p1leader)*(

                #             )
                #         )
                #     )
                # )

                benefit = (
                    (Nwcl + Nscl) * eps1 + (Nwdl + Nsdl) * eps + # leaders
                    (N - Nsl - Nwl) * ( # non leaders
                        pW * ( # weak
                            p1leader * ( # find a leader
                                follow_w * ( # choose a weak leader
                                    (1 - pF[0, 0]) * (pwc * eps1 + pwd * eps) + # not follow
                                    (pF[0, 0] * (pwcl * (eps1**2 + eps**2) + pwdl * (2*eps1*eps))) # follow
                                ) + 
                                follow_s * ( # choose a strong leader
                                    (1 - pF[0, 1]) * (pwc * eps1 + pwd * eps) +
                                    (pF[0, 1] * (pscl * (eps1**2 + eps**2) + psdl * (2*eps1*eps)))
                                )
                            ) + (1 - p1leader) * ( # do not find a leader
                                pwc * eps1 + pwd * eps
                            )
                        ) +
                        pS * ( #strong
                            p1leader * (
                                follow_w * ( # choose a weak leader
                                    (1 - pF[1, 0]) * (psc * eps1 + psd * eps) + # not follow
                                    (pF[1, 0] * (pwcl * (eps1**2 + eps**2) + pwdl * (2*eps1*eps))) # follow
                                ) + 
                                follow_s * ( # choose a strong leader
                                    (1 - pF[1, 1]) * (psc * eps1 + psd * eps) +
                                    (pF[1, 1] * (pscl * (eps1**2 + eps**2) + psdl * (2*eps1*eps)))
                                )
                            ) + (1 - p1leader) * (
                                psc * eps1 + psd * eps
                            )                       
                        )
                    )
                )

                cost = (
                    pW * ( # focus player is weak
                        p1leader*(
                            pwl * aeps(s1[0], eps) + # focus playes is a leader
                            (1 - pwl) * ( # is not a leader
                                    follow_w * ( # choose weak leader
                                        (1 - pF[0, 0]) * aeps(s1[2], eps) +
                                        pF[0, 0] * (pwcl * (eps1**2 + eps**2) + pwdl * (2*eps1*eps))
                                    ) + 
                                    follow_s * (
                                        (1 - pF[0, 1]) * aeps(s1[2], eps) +
                                        pF[0, 1] * (pscl * (eps1**2 + eps**2) + psdl * (2*eps1*eps))                                
                                    )
                            )
                        ) + (1 - p1leader) * aeps(s1[2], eps)
                    ) +
                    pS * ( # focus player is strong
                        p1leader *(
                            psl * aeps(s1[1], eps) + # focus playes is a leader
                            (1 - psl) * ( # is not a leader
                                    follow_w * ( # choose weak leader
                                        (1 - pF[1, 0]) * aeps(s1[3], eps) +
                                        pF[1, 0] * (pwcl * (eps1**2 + eps**2) + pwdl * (2*eps1*eps))
                                    ) + 
                                    follow_s * (
                                        (1 - pF[1, 1]) * aeps(s1[3], eps) +
                                        pF[1, 1] * (pscl * (eps1**2 + eps**2) + psdl * (2*eps1*eps))                                
                                    )
                            )
                        ) + (1 - p1leader) * (aeps(s1[3], eps))
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
