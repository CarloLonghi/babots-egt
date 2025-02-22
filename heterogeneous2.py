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
    WCD=np.zeros((2,2,N+1,2))
    eps1=1.-eps

    pW = 1. - pS
    pleadS = 1 / (1 + np.exp(-deltaL))
    pleadW = 1 - pleadS

    for i in range(2):
        s1=i # s:[w,s], 0:[0,0], 1:[1,0], 2:[0,1], 3:[1,1]
        for j in range(2):
            s2=j
            for k in range(1,N): # k: number of players following strategy s1
                benefit = 0
                cost = 0
                for Ns in range(N+1):
                    Nw = N - Ns
                    pNs = math.factorial(N)/(math.factorial(Ns)*math.factorial(Nw)) * pS**Ns * pW**Nw
                    for ns1 in range(max(k-Nw, 0), min(k, Ns)+1):
                        nw1 = k - ns1
                        ns2 = Ns - ns1
                        nw2 = N - k - ns2
                        prob = math.factorial(Ns)/(math.factorial(ns1)*math.factorial(Ns-ns1))
                        prob *= math.factorial(Nw)/(math.factorial(nw1)*math.factorial(Nw-nw1))
                        prob /= math.factorial(N)/(math.factorial(k)*math.factorial(N-k))                         
                        for nsl in range(Ns+1):
                            probsl = math.factorial(Ns)/(math.factorial(nsl)*math.factorial(Ns-nsl)) * pleadS**nsl * (1-pleadS)**(Ns-nsl)
                            for nwl in range(Nw+1):
                                probwl = math.factorial(Nw)/(math.factorial(nwl)*math.factorial(Nw-nwl)) * pleadW**nwl * (1-pleadW)**(Nw-nwl)
                                probl = probsl * probwl
                                for ns1l in range(max(nsl-ns2, 0), min(ns1, nsl)+1):
                                    ns2l = nsl - ns1l
                                    probsldist = math.factorial(ns1)/(math.factorial(ns1l)*math.factorial(ns1-ns1l))
                                    probsldist *= math.factorial(ns2)/(math.factorial(ns2l)*math.factorial(ns2-ns2l))
                                    probsldist /= math.factorial(Ns)/(math.factorial(nsl)*math.factorial(Ns-nsl))
                                    for nw1l in range(max(nwl-nw2, 0), min(nw1, nwl)+1):
                                        nw2l = nwl - nw1l

                                        probwldist = math.factorial(nw1)/(math.factorial(nw1l)*math.factorial(nw1-nw1l))
                                        probwldist *= math.factorial(nw2)/(math.factorial(nw2l)*math.factorial(nw2-nw2l))
                                        probwldist /= math.factorial(Nw)/(math.factorial(nwl)*math.factorial(Nw-nwl))

                                        Nwc = (nw1-nw1l)*s1 + (nw2-nw2l)*s2
                                        Nsc = (ns1-ns1l)*s1 + (ns2-ns2l)*s2
                                        Nwd = (nw1-nw1l)*(1-s1) + (nw2-nw2l)*(1-s2)
                                        Nsd = (ns1-ns1l)*(1-s1) + (ns2-ns2l)*(1-s2)

                                        Nwcl = nw1l*s1 + nw2l*s2
                                        Nscl = ns1l*s1 + ns2l*s2
                                        Nwdl = nw1l*(1-s1) + nw2l*(1-s2)
                                        Nsdl = ns1l*(1-s1) + ns2l*(1-s2)
                                        Nwl = Nwcl + Nwdl
                                        Nsl = Nscl + Nsdl

                                        Nwnl = Nw - Nwl
                                        Nsnl = Ns - Nsl

                                        final_prob = pNs*prob*probl*probsldist*probwldist

                                        if nwl + nsl == 0:
                                            benefit += final_prob*(
                                                (Nwc+Nsc)*eps1 + (Nwd+Nsd)*eps
                                            )
                                            cost += final_prob*(
                                                (nw1/k)*aeps(s1, eps)+
                                                (ns1/k)*aeps(s1, eps)
                                            )
                                        else:
                                            follow_s = (pleadS * Nsl) / (pleadS * Nsl + pleadW * Nwl)
                                            follow_w = (pleadW * Nwl) / (pleadS * Nsl + pleadW * Nwl)  
                                            benefit += final_prob*(
                                                (Nwcl + Nscl)*eps1 + (Nwdl + Nsdl)*eps # leaders
                                            )
                                            if Nwl > 0:
                                                benefit += final_prob*(
                                                    follow_w*(
                                                        (1-pF[0,0])*(Nwc*eps1 + Nwd*eps)+
                                                        (1-pF[1,0])*(Nsc*eps1 + Nsd*eps)+
                                                        pF[0,0]*Nwnl*((Nwcl/Nwl)*(eps1**2+eps**2) + (Nwdl/Nwl)*(2*eps1*eps))+
                                                        pF[1,0]*Nsnl*((Nwcl/Nwl)*(eps1**2+eps**2) + (Nwdl/Nwl)*(2*eps1*eps))
                                                    )
                                                )

                                                if nw1 > 0:
                                                    cost += final_prob*(
                                                        (nw1/k)*(
                                                            (nw1l/nw1)*(aeps(s1, eps))+
                                                            (1-(nw1l/nw1))*(
                                                                follow_w*(
                                                                    (1-pF[0,0])*aeps(s1, eps)+
                                                                    pF[0,0]*((Nwcl/Nwl)*(eps1**2+eps**2) + (Nwdl/Nwl)*(2*eps1*eps))
                                                                )
                                                            )
                                                        )
                                                    )       
                                                if ns1 > 0:
                                                    cost += final_prob*(
                                                        (ns1/k)*(
                                                            (1-(ns1l/ns1))*(
                                                                follow_w*(
                                                                    (1-pF[1,0])*aeps(s1, eps)+
                                                                    pF[1,0]*((Nwcl/Nwl)*(eps1**2+eps**2) + (Nwdl/Nwl)*(2*eps1*eps))
                                                                )
                                                            )
                                                        )
                                                    )
                                            
                                            if Nsl > 0:
                                                benefit += final_prob*(
                                                    follow_s*(
                                                        (1-pF[0,1])*(Nwc*eps1 + Nwd*eps)+
                                                        (1-pF[1,1])*(Nsc*eps1 + Nsd*eps)+
                                                        pF[0,1]*Nwnl*((Nscl/Nsl)*(eps1**2+eps**2) + (Nsdl/Nsl)*(2*eps1*eps))+
                                                        pF[1,1]*Nsnl*((Nscl/Nsl)*(eps1**2+eps**2) + (Nsdl/Nsl)*(2*eps1*eps))                                                
                                                    )
                                                )
                                                if nw1 > 0:
                                                    cost += final_prob*(
                                                        (nw1/k)*(
                                                            (1-(nw1l/nw1))*(
                                                                follow_s*(
                                                                    (1-pF[0,1])*aeps(s1, eps)+
                                                                    pF[0,1]*((Nscl/Nsl)*(eps1**2+eps**2) + (Nsdl/Nsl)*(2*eps1*eps))
                                                                )
                                                            )
                                                        )
                                                    )
                                                if ns1 > 0:
                                                    cost += final_prob*(
                                                        (ns1/k)*(
                                                            (ns1l/ns1)*(aeps(s1, eps))+
                                                            (1-(ns1l/ns1))*(
                                                                follow_s*(
                                                                    (1-pF[1,1])*aeps(s1, eps)+
                                                                    pF[1,1]*((Nscl/Nsl)*(eps1**2+eps**2) + (Nsdl/Nsl)*(2*eps1*eps))
                                                                )
                                                            )
                                                        )
                                                    )                                                

                if benefit > M:
                    WCD[i,j,k,0] = benefit/N
                WCD[i,j,k,1] = cost

            benefit = 0
            cost = 0
            for Ns in range(N+1):
                ns1 = Ns
                Nw = N - Ns
                nw1 = Nw
                pNs = math.factorial(N)/(math.factorial(Ns)*math.factorial(Nw)) * pS**Ns * pW**Nw
                for ns1l in range(ns1+1):
                    probs1l = math.factorial(ns1)/(math.factorial(ns1l)*math.factorial(ns1-ns1l)) * pleadS**ns1l * (1-pleadS)**(ns1-ns1l)
                    for nw1l in range(nw1+1):
                        probw1l = math.factorial(nw1)/(math.factorial(nw1l)*math.factorial(nw1-nw1l)) * pleadW**nw1l * (1-pleadW)**(nw1-nw1l)
                        
                        probl = probs1l*probw1l

                        Nwc = (nw1-nw1l)*s1
                        Nsc = (ns1-ns1l)*s1
                        Nwd = (nw1-nw1l)*(1-s1)
                        Nsd = (ns1-ns1l)*(1-s1)

                        Nwcl = nw1l*s1
                        Nscl = ns1l*s1
                        Nwdl = nw1l*(1-s1)
                        Nsdl = ns1l*(1-s1)
                        Nwl = Nwcl + Nwdl
                        Nsl = Nscl + Nsdl

                        Nwnl = Nw - Nwl
                        Nsnl = Ns - Nsl

                        if nw1l + ns1l == 0:
                            benefit += pNs*probl*(
                                (Nwc+Nsc)*eps1 + (Nwd+Nsd)*eps
                            )
                            cost += pNs*probl*(
                                (nw1/N)*aeps(s1, eps)+
                                (ns1/N)*aeps(s1, eps)
                            )
                        else:
                            follow_s = (pleadS * Nsl) / (pleadS * Nsl + pleadW * Nwl)
                            follow_w = (pleadW * Nwl) / (pleadS * Nsl + pleadW * Nwl)      
                            benefit += pNs*probl*(
                                (Nwcl + Nscl) * eps1 + (Nwdl + Nsdl) * eps # leaders
                            )                      
                            if Nwl > 0:
                                benefit += pNs*probl*(
                                    follow_w*(
                                        (1-pF[0,0])*(Nwc*eps1 + Nwd*eps)+
                                        (1-pF[1,0])*(Nsc*eps1 + Nsd*eps)+
                                        pF[0,0]*Nwnl*((Nwcl/Nwl)*(eps1**2+eps**2) + (Nwdl/Nwl)*(2*eps1*eps))+
                                        pF[1,0]*Nsnl*((Nwcl/Nwl)*(eps1**2+eps**2) + (Nwdl/Nwl)*(2*eps1*eps))
                                    )
                                )

                                if nw1 > 0:
                                    cost += pNs*probl*(
                                        (nw1/N)*(
                                            (nw1l/nw1)*(aeps(s1, eps))+
                                            (1-(nw1l/nw1))*(
                                                follow_w*(
                                                    (1-pF[0,0])*aeps(s1, eps)+
                                                    pF[0,0]*((Nwcl/Nwl)*(eps1**2+eps**2) + (Nwdl/Nwl)*(2*eps1*eps))
                                                )
                                            )
                                        )
                                    )
                                if ns1 > 0:
                                    cost += pNs*probl*(
                                        (ns1/N)*(
                                            (1-(ns1l/ns1))*(
                                                follow_w*(
                                                    (1-pF[1,0])*aeps(s1, eps)+
                                                    pF[1,0]*((Nwcl/Nwl)*(eps1**2+eps**2) + (Nwdl/Nwl)*(2*eps1*eps))
                                                )
                                            )
                                        )
                                    )                                

                            if Nsl > 0:
                                benefit += pNs*probl*(
                                    follow_s*(
                                        (1-pF[0,1])*(Nwc*eps1 + Nwd*eps)+
                                        (1-pF[1,1])*(Nsc*eps1 + Nsd*eps)+
                                        pF[0,1]*Nwnl*((Nscl/Nsl)*(eps1**2+eps**2) + (Nsdl/Nsl)*(2*eps1*eps))+
                                        pF[1,1]*Nsnl*((Nscl/Nsl)*(eps1**2+eps**2) + (Nsdl/Nsl)*(2*eps1*eps))                                                
                                    )
                                )
                                if nw1 > 0:
                                    cost += pNs*probl*(
                                        (nw1/N)*(
                                            (1-(nw1l/nw1))*(
                                                follow_s*(
                                                    (1-pF[0,1])*aeps(s1, eps)+
                                                    pF[0,1]*((Nscl/Nsl)*(eps1**2+eps**2) + (Nsdl/Nsl)*(2*eps1*eps))
                                                )
                                            )
                                        )
                                    )
                                if ns1 > 0:
                                    cost +=pNs*probl*(
                                        (ns1/N)*(
                                            (ns1l/ns1)*(aeps(s1, eps))+
                                            (1-(ns1l/ns1))*(
                                                follow_s*(
                                                    (1-pF[1,1])*aeps(s1, eps)+
                                                    pF[1,1]*((Nscl/Nsl)*(eps1**2+eps**2) + (Nsdl/Nsl)*(2*eps1*eps))
                                                )
                                            )
                                        )
                                    )                                
            if benefit > M:
                WCD[i,j,N,0] = benefit/N
            WCD[i,j,N,1] = cost

            WCD[i,j,0,0] = 0
            WCD[i,j,0,1] = 0                 
                
    return WCD 
    
def gaussian(x, mu, sig):
    return np.exp(-((x - mu) ** 2) / (2 * (sig ** 2))) / (np.sqrt(2 * np.pi) * sig)
