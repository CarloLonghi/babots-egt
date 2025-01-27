#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:40:35 2019

@author: abraxas
"""

import evoEGT as evo

import numpy as np
import math


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
    S = 1/(1+np.exp(-deltaL))
    fw = 1 - S
    fs = S
    Nw = N*pW
    Ns = N*pS
    for i in range(4):
        s1=[i%2,i//2] # s:[w,s], 0:[0,0], 1:[1,0], 2:[0,1], 3:[1,1]
        for j in range(4):
            s2=[j%2,j//2]
            for k in range(0,N+1): # k number of cooperators

                benefit_ss = 0
                benefit_ww = 0
                benefit_sw = 0
                cost_ss = 0
                cost_ww = 0
                cost_sw = 0

                for n1s in range(k+1):
                    n1w = k-n1s
                    for n2s in range(N-k+1):
                        n2w = (N-k)-n2s
                        Nw = n1w+n2w
                        Ns = n1s+n2s
                        Nwc = n1w*s1[0]+n2w*s2[0]
                        Nwd = Nw-Nwc
                        Nsc = n1s*s1[1]+n2s*s2[1]
                        Nsd = Ns-Nsc

                        prob = (pS**n1s*pW**n1w)*(pS**n2s*pW**n2w)*(
                            math.factorial(k)/(math.factorial(n1s)*math.factorial(n1w))*
                            math.factorial(N-k)/(math.factorial(n2s)*math.factorial(n2w))
                        )

                        prob_ww = 0
                        prob_ss = 0
                        prob_sw = 0
                        if Nw >= 2:
                            prob_ww = (Nw*fw)/(Nw*fw+Ns*fs)*((Nw-1)*fw)/((Nw-1)*fw+Ns*fs)
                        if Ns >= 2:
                            prob_ss = (Ns*fs)/(Nw*fw+Ns*fs)*((Ns-1)*fs)/((Ns-1)*fs+Nw*fw)
                        if Nw >= 1 and Ns >= 1:
                            prob_sw = 1 - (prob_ww + prob_ss)


                        if Nwc >= 2:
                            benefit_ww += prob*prob_ww*(
                                ((Nwc/Nw)*((Nwc-1)/(Nw-1)))*( # both leaders are cooperators
                                    eps1*2 + 
                                    (1-pF[0,0])*((Nwc-2)*eps1 + Nwd*eps)+
                                    (1-pF[1,0])*(Nsc*eps1 + Nsd*eps)+
                                    pF[0,0]*(Nw-2)*(eps1**2+eps**2)+pF[1,0]*Ns*(eps1**2+eps**2)
                                )
                            )
                            cost_ww += prob*prob_ww*(
                                ((Nwc/Nw)*((Nwc-1)/(Nw-1)))*(
                                    pW*(
                                        s1[0]*( # focus is cooperator
                                            (2/Nwc)*eps1+
                                            (1-(2/Nwc))*(
                                                (1-pF[0,0])*eps1+pF[0,0]*(eps1**2+eps**2)
                                            )
                                        )+
                                        (1-s1[0])*( # focus is defector
                                            (1-pF[0,0])*eps+pF[0,0]*(eps1**2+eps**2)
                                        )
                                    )+
                                    pS*(
                                        (1-pF[1,0])*aeps(s1[1],eps)+pF[1,0]*(eps1**2+eps**2)
                                    )
                                )
                            )
                        
                        if Nwd >= 2:
                            benefit_ww += prob*prob_ww*(
                                ((Nwd/Nw)*((Nwd-1)/(Nw-1)))*( # both leaders are defectors
                                    eps*2 + 
                                    (1-pF[0,0])*(Nwc*eps1 + (Nwd-2)*eps)+
                                    (1-pF[1,0])*(Nsc*eps1 + Nsd*eps)+
                                    pF[0,0]*(Nw-2)*(2*eps*eps1) + pF[1,0]*Ns*(2*eps*eps1)
                                )
                            )
                            cost_ww += prob*prob_ww*(
                                ((Nwd/Nw)*((Nwd-1)/(Nw-1)))*(
                                    pW*(
                                        (1-s1[0])*(
                                            (2/Nwd)*eps+
                                            (1-(2/Nwd))*(
                                                (1-pF[0,0])*eps+pF[0,0]*(2*eps1*eps)
                                            )
                                        )+
                                        s1[0]*(
                                            (1-pF[0,0])*eps1+pF[0,0]*(2*eps1*eps)
                                        )
                                    )+
                                    pS*(
                                        (1-pF[1,0])*aeps(s1[1],eps)+pF[1,0]*(2*eps1*eps)
                                    )
                                )
                            )

                        if Nwc >= 1 and Nwd >= 1:
                            benefit_ww += prob*prob_ww*(
                                ((Nwc/Nw)*(Nwd/(Nw-1)))*2*( # one cooperator one defector
                                    eps1+eps +
                                    (1-pF[0,0])*((Nwc-1)*eps1 + (Nwd-1)*eps)+
                                    (1-pF[1,0])*(Nsc*eps1 + Nsd*eps)+
                                    (1/2)*(pF[0,0]*(Nw-2)*(eps1**2+eps**2)+pF[1,0]*Ns*(eps1**2+eps**2)) + # choose cooperating leader
                                    (1/2)*(pF[0,0]*(Nw-2)*(2*eps*eps1)+pF[1,0]*Ns*(2*eps*eps1)) # choose defecting leader
                                )
                            )
                            cost_ww += prob*prob_ww*(
                                ((Nwc/Nw)*(Nwd/(Nw-1)))*2*(
                                    pW*(
                                        (1-s1[0])*(
                                            (1/Nwd)*eps+
                                            (1-(1/Nwd))*(
                                                (1-pF[0,0])*eps+pF[0,0]*(0.5*(2*eps1*eps)+0.5*(eps1**2+eps**2))
                                            )
                                        )+
                                        s1[0]*(
                                            (1/Nwc)*eps1+
                                            (1-(1/Nwc))*(
                                                (1-pF[0,0])*eps1+pF[0,0]*(0.5*(2*eps1*eps)+0.5*(eps1**2+eps**2))
                                            )
                                        )
                                    )+
                                    pS*(
                                        (1-pF[1,0])*aeps(s1[1],eps)+pF[1,0]*(0.5*(2*eps1*eps)+0.5*(eps1**2+eps**2))
                                    )
                                )
                            )

                        if Nsc >= 2:
                            benefit_ss += prob*prob_ss*(
                                ((Nsc/Ns)*((Nsc-1)/(Ns-1)))*( # both leaders are cooperators
                                    eps1*2 + 
                                    (1-pF[0,1])*(Nwc*eps1 + Nwd*eps)+
                                    (1-pF[1,1])*((Nsc-2)*eps1 + Nsd*eps)+
                                    pF[0,1]*Nw*(eps1**2+eps**2)+pF[1,1]*(Ns-2)*(eps1**2+eps**2)
                                )
                            )
                            cost_ss += prob*prob_ss*(
                                ((Nsc/Ns)*((Nsc-1)/(Ns-1)))*(
                                    pS*(
                                        s1[1]*( # focus is cooperator
                                            (2/Nsc)*eps1+
                                            (1-(2/Nsc))*(
                                                (1-pF[1,1])*eps1+pF[1,1]*(eps1**2+eps**2)
                                            )
                                        )+
                                        (1-s1[1])*( # focus is defector
                                            (1-pF[1,1])*eps+pF[1,1]*(eps1**2+eps**2)
                                        )
                                    )+
                                    pW*(
                                        (1-pF[0,1])*aeps(s1[0],eps)+pF[0,1]*(eps1**2+eps**2)
                                    )
                                )
                            )

                        if Nsd >= 2:
                            benefit_ss += prob*prob_ss*(
                                ((Nsd/Ns)*((Nsd-1)/(Ns-1)))*( # both leaders are defectors
                                    eps*2 + 
                                    (1-pF[0,1])*(Nwc*eps1 + Nwd*eps)+
                                    (1-pF[1,1])*(Nsc*eps1 + (Nsd-2)*eps)+
                                    pF[0,1]*Nw*(2*eps*eps1) + pF[1,1]*(Ns-2)*(2*eps*eps1)
                                )
                            )
                            cost_ss += prob*prob_ss*(
                                ((Nsd/Ns)*((Nsd-1)/(Ns-1)))*(
                                    pS*(
                                        (1-s1[1])*(
                                            (2/Nsd)*eps+
                                            (1-(2/Nsd))*(
                                                (1-pF[1,1])*eps+pF[1,1]*(2*eps1*eps)
                                            )
                                        )+
                                        s1[1]*(
                                            (1-pF[1,1])*eps1+pF[1,1]*(2*eps1*eps)
                                        )
                                    )+
                                    pW*(
                                        (1-pF[0,1])*aeps(s1[0],eps)+pF[0,1]*(2*eps1*eps)
                                    )
                                )
                            )
                        
                        if Nsc >= 1 and Nsd >= 1:
                            benefit_ss += prob*prob_ss*(
                                ((Nsc/Ns)*(Nsd/(Ns-1)))*2*( # one cooperator one defector
                                    eps1+eps +
                                    (1-pF[0,1])*(Nwc*eps1 + Nwd*eps)+
                                    (1-pF[1,1])*((Nsc-1)*eps1 + (Nsd-1)*eps)+
                                    (1/2)*(pF[0,1]*Nw*(eps1**2+eps**2)+pF[1,1]*(Ns-2)*(eps1**2+eps**2)) + # choose cooperating leader
                                    (1/2)*(pF[0,1]*Nw*(2*eps*eps1)+pF[1,1]*(Ns-2)*(2*eps*eps1)) # choose defecting leader
                                )
                            )
                            cost_ss += prob*prob_ss*(
                                ((Nsc/Ns)*(Nsd/(Ns-1)))*2*(
                                    pS*(
                                        (1-s1[1])*(
                                            (1/Nsd)*eps+
                                            (1-(1/Nsd))*(
                                                (1-pF[1,1])*eps+pF[1,1]*(0.5*(2*eps1*eps)+0.5*(eps1**2+eps**2))
                                            )
                                        )+
                                        s1[1]*(
                                            (1/Nsc)*eps1+
                                            (1-(1/Nsc))*(
                                                (1-pF[1,1])*eps1+pF[1,1]*(0.5*(2*eps1*eps)+0.5*(eps1**2+eps**2))
                                            )
                                        )
                                    )+
                                    pW*(
                                        (1-pF[0,1])*aeps(s1[0],eps)+pF[0,1]*(0.5*(2*eps1*eps)+0.5*(eps1**2+eps**2))
                                    )
                                ) 
                            )

                        if Nsc >= 1 and Nwc >= 1:
                            benefit_sw += prob*prob_sw*(
                                ((Nsc/Ns)*(Nwc/Nw))*( # both leaders are cooperators
                                    eps1*2 +
                                    (
                                        (fw/(fw+fs))*(
                                            (1-pF[0,0])*((Nwc-1)*eps1+Nwd*eps)+
                                            (1-pF[1,0])*((Nsc-1)*eps1+Nsd*eps)+
                                            pF[0,0]*(Nw-1)*(eps1**2+eps**2)+pF[1,0]*(Ns-1)*(eps1**2+eps**2)
                                        ) + 
                                        (fs/(fw+fs))*(
                                            (1-pF[0,1])*((Nwc-1)*eps1+Nwd*eps)+
                                            (1-pF[1,1])*((Nsc-1)*eps1+Nsd*eps)+
                                            pF[0,1]*(Nw-1)*(eps1**2+eps**2)+pF[1,1]*(Ns-1)*(eps1**2+eps**2)                                    
                                        )
                                    )
                                )
                            )
                            cost_sw += prob*prob_sw*(
                                ((Nsc/Ns)*(Nwc/Nw))*(
                                    pW*(
                                        s1[0]*( # focus is cooperator
                                            (1/Nwc)*eps1+
                                            (1-(1/Nwc))*(
                                                (fw/(fw+fs))*(1-pF[0,0])*eps1+pF[0,0]*(eps1**2+eps**2)+
                                                (fs/(fw+fs))*(1-pF[0,1])*eps1+pF[0,1]*(eps1**2+eps**2)
                                            )
                                        )+
                                        (1-s1[0])*( # focus is defector
                                            (fw/(fw+fs))*(1-pF[0,0])*eps+pF[0,0]*(eps1**2+eps**2)+
                                            (fs/(fw+fs))*(1-pF[0,1])*eps+pF[0,1]*(eps1**2+eps**2)
                                        )
                                    )+
                                    pS*(
                                        s1[1]*( # focus is cooperator
                                            (1/Nsc)*eps1+
                                            (1-(1/Nsc))*(
                                                (fw/(fw+fs))*(1-pF[1,0])*eps1+pF[1,0]*(eps1**2+eps**2)+
                                                (fs/(fw+fs))*(1-pF[1,1])*eps1+pF[1,1]*(eps1**2+eps**2)
                                            )
                                        )+
                                        (1-s1[1])*( # focus is defector
                                            (fw/(fw+fs))*(1-pF[1,0])*eps+pF[1,0]*(eps1**2+eps**2)+
                                            (fs/(fw+fs))*(1-pF[1,1])*eps+pF[1,1]*(eps1**2+eps**2)
                                        )                        )
                                )
                            )

                        if Nsd >= 1 and Nwd >= 1:
                            benefit_sw += prob*prob_sw*(
                                ((Nsd/Ns)*(Nwd/Nw))*( # both leaders are defectors
                                    eps*2 +
                                    (
                                        (fw/(fw+fs))*(
                                            (1-pF[0,0])*(Nwc*eps1+(Nwd-1)*eps)+
                                            (1-pF[1,0])*(Nsc*eps1+(Nsd-1)*eps)+
                                            pF[0,0]*(Nw-1)*(2*eps1*eps)+pF[1,0]*(Ns-1)*(2*eps1*eps)
                                        ) + 
                                        (fs/(fw+fs))*(
                                            (1-pF[0,1])*(Nwc*eps1+(Nwd-1)*eps)+
                                            (1-pF[1,1])*(Nsc*eps1+(Nsd-1)*eps)+
                                            pF[0,1]*(Nw-1)*(2*eps1*eps)+pF[1,1]*(Ns-1)*(2*eps1*eps)                                    
                                        )
                                    )
                                )
                            )
                            cost_sw += prob*prob_sw*(
                                ((Nsd/Ns)*(Nwd/Nw))*(
                                    pW*(
                                        (1-s1[0])*(
                                            (1/Nwd)*eps+
                                            (1-(1/Nwd))*(
                                                (fw/(fw+fs))*(1-pF[0,0])*eps+pF[0,0]*(2*eps1*eps)+
                                                (fs/(fw+fs))*(1-pF[0,1])*eps+pF[0,1]*(2*eps1*eps)
                                            )
                                        )+
                                        s1[0]*(
                                            (fw/(fw+fs))*(1-pF[0,0])*eps1+pF[0,0]*(2*eps1*eps)+
                                            (fs/(fw+fs))*(1-pF[0,1])*eps1+pF[0,1]*(2*eps1*eps)
                                        )
                                    )+
                                    pS*(
                                        (1-s1[1])*(
                                            (1/Nsd)*eps+
                                            (1-(1/Nsd))*(
                                                (fw/(fw+fs))*(1-pF[1,0])*eps+pF[1,0]*(2*eps1*eps)+
                                                (fs/(fw+fs))*(1-pF[1,1])*eps+pF[1,1]*(2*eps1*eps)
                                            )
                                        )+
                                        s1[1]*(
                                            (fw/(fw+fs))*(1-pF[1,0])*eps1+pF[1,0]*(2*eps1*eps)+
                                            (fs/(fw+fs))*(1-pF[1,1])*eps1+pF[1,1]*(2*eps1*eps)
                                        ) 
                                    )
                                )
                            )
                        
                        if Nwc >= 1 and Nsd >= 1:
                            benefit_sw += prob*prob_sw*(
                                ((Nsd/Ns)*(Nwc/Nw))*( # one weak cooperator one strong defector
                                    eps1+eps +
                                    (
                                        (fw/(fw+fs))*(
                                            (1-pF[0,0])*((Nwc-1)*eps1+Nwd*eps)+
                                            (1-pF[1,0])*(Nsc*eps1+(Nsd-1)*eps)+
                                            pF[0,0]*(Nw-1)*(eps1**2+eps**2)+pF[1,0]*(Ns-1)*(eps1**2+eps**2)
                                        ) + 
                                        (fs/(fw+fs))*(
                                            (1-pF[0,1])*((Nwc-1)*eps1+Nwd*eps)+
                                            (1-pF[1,1])*(Nsc*eps1+(Nsd-1)*eps)+
                                            pF[0,1]*(Nw-1)*(2*eps1*eps)+pF[1,1]*(Ns-1)*(2*eps1*eps)                                    
                                        )
                                    )
                                )
                            )
                            cost_sw += prob*prob_sw*(
                                ((Nsd/Ns)*(Nwc/Nw))*(
                                    pW*(
                                        s1[0]*(
                                            (1/Nwc)*eps1+
                                            (1-(1/Nwc))*(
                                                (fw/(fw+fs))*(1-pF[0,0])*eps1+pF[0,0]*(eps1**2+eps**2)+
                                                (fs/(fw+fs))*(1-pF[0,1])*eps1+pF[0,1]*(2*eps1*eps)
                                            )
                                        )+
                                        (1-s1[0])*(
                                            (fw/(fw+fs))*(1-pF[0,0])*eps+pF[0,0]*(eps1**2+eps**2)+
                                            (fs/(fw+fs))*(1-pF[0,1])*eps+pF[0,1]*(2*eps1*eps)
                                        )
                                    )+
                                    pS*(
                                        (1-s1[1])*(
                                            (1/Nsd)*eps+
                                            (1-(1/Nsd))*(
                                                (fw/(fw+fs))*(1-pF[1,0])*eps+pF[1,0]*(eps1**2+eps**2)+
                                                (fs/(fw+fs))*(1-pF[1,1])*eps+pF[1,1]*(2*eps1*eps)
                                            )
                                        )+
                                        s1[1]*(
                                            (fw/(fw+fs))*(1-pF[1,0])*eps1+pF[1,0]*(eps1**2+eps**2)+
                                            (fs/(fw+fs))*(1-pF[1,1])*eps1+pF[1,1]*(2*eps1*eps)
                                        ) 
                                    )
                                )
                            )
                        
                        if Nwd >= 1 and Nsc >= 1:
                            benefit_sw += prob*prob_sw*(
                                ((Nsc/Ns)*(Nwd/Nw))*( # one strong cooperator one weak defector
                                    eps1+eps +
                                    (
                                        (fw/(fw+fs))*(
                                            (1-pF[0,0])*(Nwc*eps1+(Nwd-1)*eps)+
                                            (1-pF[1,0])*((Nsc-1)*eps1+Nsd*eps)+
                                            pF[0,0]*(Nw)*(2*eps1*eps)+pF[1,0]*(Ns)*(2*eps1*eps)
                                        ) + 
                                        (fs/(fw+fs))*(
                                            (1-pF[0,1])*(Nwc*eps1+(Nwd-1)*eps)+
                                            (1-pF[1,1])*((Nsc-1)*eps1+Nsd*eps)+
                                            pF[0,1]*(Nw-1)*(eps1**2+eps**2)+pF[1,1]*(Ns-1)*(eps1**2+eps**2)                                    
                                        )
                                    )
                                )
                            )
                            cost_sw += prob*prob_sw*(
                                ((Nsc/Ns)*(Nwd/Nw))*(
                                    pW*(
                                        (1-s1[0])*(
                                            (1/Nwd)*eps+
                                            (1-(1/Nwd))*(
                                                (fw/(fw+fs))*(1-pF[0,0])*eps+pF[0,0]*(2*eps1*eps)+
                                                (fs/(fw+fs))*(1-pF[0,1])*eps+pF[0,1]*(eps1**2+eps**2)
                                            )
                                        )+
                                        s1[0]*(
                                            (fw/(fw+fs))*(1-pF[0,0])*eps1+pF[0,0]*(2*eps1*eps)+
                                            (fs/(fw+fs))*(1-pF[0,1])*eps1+pF[0,1]*(eps1**2+eps**2)
                                        )
                                    )+
                                    pS*(
                                        s1[1]*(
                                            (1/Nsc)*eps1+
                                            (1-(1/Nsc))*(
                                                (fw/(fw+fs))*(1-pF[1,0])*eps1+pF[1,0]*(2*eps1*eps)+
                                                (fs/(fw+fs))*(1-pF[1,1])*eps1+pF[1,1]*(eps1**2+eps**2)
                                            )
                                        )+
                                        (1-s1[1])*(
                                            (fw/(fw+fs))*(1-pF[1,0])*eps+pF[1,0]*(2*eps1*eps)+
                                            (fs/(fw+fs))*(1-pF[1,1])*eps+pF[1,1]*(eps1**2+eps**2)
                                        ) 
                                    )
                                )
                            )

                benefit = benefit_ww + benefit_ss + benefit_sw
                cost = cost_ww + cost_ss + cost_sw

                if benefit > M:
                    WCD[i,j,k,0] = benefit/N

                WCD[i,j,k,1] = cost
                
    # WCD[1,0,:]=-999
    # WCD[0,N,:]=-999

    return WCD 
    
    