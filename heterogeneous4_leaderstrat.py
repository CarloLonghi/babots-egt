#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    for i in range(4):
        s1=[i%2,i//2] # s:[nL,L], 0:[0,0], 1:[1,0], 2:[0,1], 3:[1,1]
        for j in range(4):
            s2=[j%2,j//2]
            for k in range(1,N): # k number of cooperators
                benefit = 0
                cost = 0
                for Ns in range(N+1):
                    Nw = N - Ns
                    pNs = math.factorial(N)/(math.factorial(Ns)*math.factorial(Nw)) * pS**Ns * pW**Nw                    
                    for ns1 in range(max(k-Nw, 0), min(k, Ns)+1):
                        nw1 = k - ns1
                        ns2 = Ns - ns1
                        nw2 = Nw - nw1

                        Nwcnl = (k-ns1)*s1[0] + (N-k-ns2)*s2[0]
                        Nscnl = ns1*s1[0] + ns2*s2[0]
                        Nwdnl = (k-ns1)*(1-s1[0]) + (N-k-ns2)*(1-s2[0])
                        Nsdnl = ns1*(1-s1[0]) + ns2*(1-s2[0])

                        benefit_s = 0
                        benefit_w = 0
                        cost_s = 0
                        cost_w = 0

                        if Nw > 0:
                            benefit_w += (
                                (nw1/Nw)*( # leader is of strategy 1
                                    aeps(s1[1], eps) + 
                                    (1-pF[0,0])*((Nwcnl-s1[0])*eps1 + (Nwdnl-(1-s1[0]))*eps)+
                                    (1-pF[1,0])*(Nscnl*eps1 + Nsdnl*eps)+
                                    pF[0,0]*(Nw-1)*(s1[1]*(eps1**2+eps**2) + (1-s1[1])*(2*eps1*eps))
                                    +pF[1,0]*Ns*(s1[1]*(eps1**2+eps**2) + (1-s1[1])*(2*eps1*eps))
                                ) + (nw2/Nw)*( # leader is of strategy 2
                                    aeps(s2[1], eps) + 
                                    (1-pF[0,0])*((Nwcnl-s2[0])*eps1 + (Nwdnl-(1-s2[0]))*eps)+
                                    (1-pF[1,0])*(Nscnl*eps1 + Nsdnl*eps)+
                                    pF[0,0]*(Nw-1)*(s2[1]*(eps1**2+eps**2) + (1-s2[1])*(2*eps1*eps))
                                    +pF[1,0]*Ns*(s2[1]*(eps1**2+eps**2) + (1-s2[1])*(2*eps1*eps))
                                )
                            )

                            cost_w += (nw2/Nw)*( # leader is of strategy 2
                                (nw1/k)*( # focus player is weak
                                    (1-pF[0,0])*aeps(s1[0], eps)+
                                    pF[0,0]*(s2[1]*(eps1**2+eps**2) + (1-s2[1])*(2*eps1*eps))
                                )+(ns1/k)*( # focus player is strong
                                    (1-pF[1,0])*aeps(s1[0],eps)+
                                    pF[1,0]*(s2[1]*(eps1**2+eps**2) + (1-s2[1])*(2*eps1*eps))
                                )
                            )

                            if nw1 > 0:
                                cost_w += (nw1/Nw)*( # leader is of strategy 1
                                    (nw1/k)*( # focus player is weak
                                        (1/nw1)*aeps(s1[1], eps)+
                                        (1-(1/nw1))*(
                                            (1-pF[0,0])*aeps(s1[0], eps)+
                                            pF[0,0]*(s1[1]*(eps1**2+eps**2) + (1-s1[1])*(2*eps1*eps))
                                        )
                                    )+(ns1/k)*( # focus player is strong
                                        (1-pF[1,0])*aeps(s1[0],eps)+
                                        pF[1,0]*(s1[1]*(eps1**2+eps**2) + (1-s1[1])*(2*eps1*eps))
                                    )
                                )

                        if Ns > 0:

                            benefit_s += (
                                (ns1/Ns)*( # leader is of strategy 1
                                    aeps(s1[1], eps) + 
                                    (1-pF[0,1])*(Nwcnl*eps1 + Nwdnl*eps)+
                                    (1-pF[1,1])*((Nscnl-s1[0])*eps1 + (Nsdnl-(1-s1[0]))*eps)+
                                    pF[0,1]*Nw*(s1[1]*(eps1**2+eps**2) + (1-s1[1])*(2*eps1*eps))
                                    +pF[1,1]*(Ns-1)*(s1[1]*(eps1**2+eps**2) + (1-s1[1])*(2*eps1*eps))
                                ) + (ns2/Ns)*( # leader is of strategy 2
                                    aeps(s2[1], eps) + 
                                    (1-pF[0,1])*(Nwcnl*eps1 + Nwdnl*eps)+
                                    (1-pF[1,1])*((Nscnl-s2[0])*eps1 + (Nsdnl-(1-s2[0]))*eps)+
                                    pF[0,1]*Nw*(s2[1]*(eps1**2+eps**2) + (1-s2[1])*(2*eps1*eps))
                                    +pF[1,1]*(Ns-1)*(s2[1]*(eps1**2+eps**2) + (1-s2[1])*(2*eps1*eps))
                                )
                            )

                            cost_s += (ns2/Ns)*( # leader is of strategy 2
                                (nw1/k)*( # focus player is weak
                                    (1-pF[0,1])*aeps(s1[0], eps)+
                                    pF[0,1]*(s2[1]*(eps1**2+eps**2) + (1-s2[1])*(2*eps1*eps))
                                )+(ns1/k)*( # focus player is strong
                                    (1-pF[1,1])*aeps(s1[0],eps)+
                                    pF[1,1]*(s2[1]*(eps1**2+eps**2) + (1-s2[1])*(2*eps1*eps))
                                )
                            )

                            if ns1 > 0:
                                cost_s += (ns1/Ns)*( # leader is of strategy 1
                                    (ns1/k)*( # focus player is strong
                                        (1/ns1)*aeps(s1[1], eps)+
                                        (1-(1/ns1))*(
                                            (1-pF[1,1])*aeps(s1[0], eps)+
                                            pF[1,1]*(s1[1]*(eps1**2+eps**2) + (1-s1[1])*(2*eps1*eps))
                                        )
                                    )+(nw1/k)*( # focus player is weak
                                        (1-pF[0,1])*aeps(s1[0],eps)+
                                        pF[0,1]*(s1[1]*(eps1**2+eps**2) + (1-s1[1])*(2*eps1*eps))
                                    )
                                )

                        prob = math.factorial(Ns)/(math.factorial(ns1)*math.factorial(Ns-ns1))
                        prob *= math.factorial(Nw)/(math.factorial(nw1)*math.factorial(Nw-nw1))
                        prob /= math.factorial(N)/(math.factorial(k)*math.factorial(N-k))                            
                        benefit += pNs*prob*(((Nw*fw)/(Nw*fw+Ns*fs))*benefit_w + ((Ns*fs)/(Nw*fw+Ns*fs))*benefit_s)
                        cost += pNs*prob*(((Nw*fw)/(Nw*fw+Ns*fs))*cost_w + ((Ns*fs)/(Nw*fw+Ns*fs))*cost_s)

                if benefit > M:
                    WCD[i,j,k,0] = benefit/N
                WCD[i,j,k,1] = cost                            

            benefit = 0
            cost = 0
            for Ns in range(N+1):
                ns1 = Ns
                nw1 = N - ns1
                Nw = N - Ns
                pNs = math.factorial(N)/(math.factorial(Ns)*math.factorial(Nw)) * pS**Ns * pW**Nw     

                Nwcnl = (N-ns1)*s1[0]
                Nscnl = ns1*s1[0]
                Nwdnl = (N-ns1)*(1-s1[0])
                Nsdnl = ns1*(1-s1[0])

                benefit_s = 0
                benefit_w = 0
                cost_s = 0
                cost_w = 0

                if Nw > 0:
                    benefit_w += (
                        (nw1/Nw)*( # leader is of strategy 1
                            aeps(s1[1], eps) + 
                            (1-pF[0,0])*((Nwcnl-s1[0])*eps1 + (Nwdnl-(1-s1[0]))*eps)+
                            (1-pF[1,0])*(Nscnl*eps1 + Nsdnl*eps)+
                            pF[0,0]*(Nw-1)*(s1[1]*(eps1**2+eps**2) + (1-s1[1])*(2*eps1*eps))
                            +pF[1,0]*Ns*(s1[1]*(eps1**2+eps**2) + (1-s1[1])*(2*eps1*eps))
                        )
                    )

                    if nw1 > 0:
                        cost_w += (nw1/Nw)*( # leader is of strategy 1
                            (nw1/N)*( # focus player is weak
                                (1/nw1)*aeps(s1[1], eps)+
                                (1-(1/nw1))*(
                                    (1-pF[0,0])*aeps(s1[0], eps)+
                                    pF[0,0]*(s1[1]*(eps1**2+eps**2) + (1-s1[1])*(2*eps1*eps))
                                )
                            )+(ns1/N)*( # focus player is strong
                                (1-pF[1,0])*aeps(s1[0],eps)+
                                pF[1,0]*(s1[1]*(eps1**2+eps**2) + (1-s1[1])*(2*eps1*eps))
                            )
                        )

                if Ns > 0:
                    benefit_s += (
                        (ns1/Ns)*( # leader is of strategy 1
                            aeps(s1[1], eps) + 
                            (1-pF[0,1])*(Nwcnl*eps1 + Nwdnl*eps)+
                            (1-pF[1,1])*((Nscnl-s1[0])*eps1 + (Nsdnl-(1-s1[0]))*eps)+
                            pF[0,1]*Nw*(s1[1]*(eps1**2+eps**2) + (1-s1[1])*(2*eps1*eps))
                            +pF[1,1]*(Ns-1)*(s1[1]*(eps1**2+eps**2) + (1-s1[1])*(2*eps1*eps))
                        )
                    )

                    if ns1 > 0:
                        cost_s += (ns1/Ns)*( # leader is of strategy 1
                            (ns1/N)*( # focus player is strong
                                (1/ns1)*aeps(s1[1], eps)+
                                (1-(1/ns1))*(
                                    (1-pF[1,1])*aeps(s1[0], eps)+
                                    pF[1,1]*(s1[1]*(eps1**2+eps**2) + (1-s1[1])*(2*eps1*eps))
                                )
                            )+(nw1/N)*( # focus player is weak
                                (1-pF[0,1])*aeps(s1[0],eps)+
                                pF[0,1]*(s1[1]*(eps1**2+eps**2) + (1-s1[1])*(2*eps1*eps))
                            )
                        )

                benefit += pNs*(((Nw*fw)/(Nw*fw+Ns*fs))*benefit_w + ((Ns*fs)/(Nw*fw+Ns*fs))*benefit_s)
                cost += pNs*(((Nw*fw)/(Nw*fw+Ns*fs))*cost_w + ((Ns*fs)/(Nw*fw+Ns*fs))*cost_s)

            if benefit > M:
                WCD[i,j,N,0] = benefit/N
            WCD[i,j,N,1] = cost 

            WCD[i,j,0,0] = 0
            WCD[i,j,0,1] = 0              
                
    return WCD 
    
    