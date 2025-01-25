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

                Nwc = pW*(k*s1[0]+(N-k)*s2[0])
                Nsc = pS*(k*s1[1]+(N-k)*s2[1])
                Nc = Nwc + Nsc
                Nd = N - Nc
                Nwd = N*pW-Nwc
                Nsd = N*pS-Nsc

                benefit_ss = 0
                benefit_ww = 0
                benefit_sw = 0
                cost_ss = 0
                cost_ww = 0
                cost_sw = 0

                if Nw > 0:
                    benefit_ww = (
                        ((Nwc/Nw)*((Nwc-1)/(Nw-1)))*( # both leaders are cooperators
                            eps1*2 + 
                            (1-pF[0,0])*((Nwc-2)*eps1 + Nwd*eps)+
                            (1-pF[1,0])*(Nsc*eps1 + Nsd*eps)+
                            pF[0,0]*(Nw-2)*(eps1**2+eps**2)+pF[1,0]*Ns*(eps1**2+eps**2)
                        ) + ((Nwd/Nw)*((Nwd-1)/(Nw-1)))*( # both leaders are defectors
                            eps*2 + 
                            (1-pF[0,0])*(Nwc*eps1 + (Nwd-2)*eps)+
                            (1-pF[1,0])*(Nsc*eps1 + Nsd*eps)+
                            pF[0,0]*(Nw-2)*(2*eps*eps1) + pF[1,0]*Ns*(2*eps*eps1)
                        ) + ((Nwc/Nw)*(Nwd/(Nw-1)))*2*( # one cooperator one defector
                            eps1+eps +
                            (1-pF[0,0])*((Nwc-1)*eps1 + (Nwd-1)*eps)+
                            (1-pF[1,0])*(Nsc*eps1 + Nsd*eps)+
                            (1/2)*(pF[0,0]*(Nw-2)*(eps1**2+eps**2)+pF[1,0]*Ns*(eps1**2+eps**2)) + # choose cooperating leader
                            (1/2)*(pF[0,0]*(Nw-2)*(2*eps*eps1)+pF[1,0]*Ns*(2*eps*eps1)) # choose defecting leader
                        )
                    )

                    cost_ww = pW*( # focus player is weak
                        (2/Nw)*aeps(s1[0],eps)+
                        (1-(2/Nw))*(
                            (1-pF[0,0])*aeps(s1[0],eps)+
                            pF[0,0]*(
                                ((Nwc/Nw)*((Nwc-1)/(Nw-1)))*(eps1**2+eps**2) +
                                ((Nwd/Nw)*((Nwd-1)/(Nw-1)))*(2*eps1*eps) +
                                ((Nwc/Nw)*(Nwd/(Nw-1)))*2*((1/2)*(eps1**2+eps**2)+(1/2)*(2*eps1*eps)))
                        )
                        ) + pS*( # focus player is strong
                            (1-pF[1,0])*aeps(s1[1],eps)+
                            pF[1,0]*(
                                ((Nwc/Nw)*((Nwc-1)/(Nw-1)))*(eps1**2+eps**2) +
                                ((Nwd/Nw)*((Nwd-1)/(Nw-1)))*(2*eps1*eps) +
                                ((Nwc/Nw)*(Nwd/(Nw-1)))*2*((1/2)*(eps1**2+eps**2)+(1/2)*(2*eps1*eps)))                            
                        )  

                if Ns > 0:
                    benefit_ss = (
                        ((Nsc/Ns)*((Nsc-1)/(Ns-1)))*( # both leaders are cooperators
                            eps1*2 + 
                            (1-pF[0,1])*(Nwc*eps1 + Nwd*eps)+
                            (1-pF[1,1])*((Nsc-2)*eps1 + Nsd*eps)+
                            pF[0,1]*Nw*(eps1**2+eps**2)+pF[1,1]*(Ns-2)*(eps1**2+eps**2)
                        ) + ((Nsd/Ns)*((Nsd-1)/(Ns-1)))*( # both leaders are defectors
                            eps*2 + 
                            (1-pF[0,1])*(Nwc*eps1 + Nwd*eps)+
                            (1-pF[1,1])*(Nsc*eps1 + (Nsd-2)*eps)+
                            pF[0,1]*Nw*(2*eps*eps1) + pF[1,1]*(Ns-2)*(2*eps*eps1)
                        ) + ((Nsc/Ns)*(Nsd/(Ns-1)))*2*( # one cooperator one defector
                            eps1+eps +
                            (1-pF[0,1])*(Nwc*eps1 + Nwd*eps)+
                            (1-pF[1,1])*((Nsc-1)*eps1 + (Nsd-1)*eps)+
                            (1/2)*(pF[0,1]*Nw*(eps1**2+eps**2)+pF[1,1]*(Ns-2)*(eps1**2+eps**2)) + # choose cooperating leader
                            (1/2)*(pF[0,1]*Nw*(2*eps*eps1)+pF[1,1]*(Ns-2)*(2*eps*eps1)) # choose defecting leader
                        )
                    )

                    cost_ss = pW*( # focus player is weak
                        (1-pF[0,1])*aeps(s1[0],eps)+
                        pF[0,1]*(
                            ((Nsc/Ns)*((Nsc-1)/(Ns-1)))*(eps1**2+eps**2) +
                            ((Nsd/Ns)*((Nsd-1)/(Ns-1)))*(2*eps1*eps) +
                            ((Nsc/Ns)*(Nsd/(Ns-1)))*2*((1/2)*(eps1**2+eps**2)+(1/2)*(2*eps1*eps))                           
                        )
                    ) + pS*( # focus player is strong
                        (2/Ns)*aeps(s1[1],eps)+
                        (1-(2/Ns))*(
                            (1-pF[1,1])*aeps(s1[1],eps)+
                            pF[1,1]*(
                            ((Nsc/Ns)*((Nsc-1)/(Ns-1)))*(eps1**2+eps**2) +
                            ((Nsd/Ns)*((Nsd-1)/(Ns-1)))*(2*eps1*eps) +
                            ((Nsc/Ns)*(Nsd/(Ns-1)))*2*((1/2)*(eps1**2+eps**2)+(1/2)*(2*eps1*eps))                                
                            )                        
                        )
                    )      

                if Ns > 0 and Nw > 0:
                    benefit_sw = (
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
                        )+
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
                        )+
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
                        )+
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

                    cost_sw = pW*( # focus player is weak
                            (1/Nw)*aeps(s1[0],eps)+
                            (1-(1/Nw))*(
                                ((Nsc/Ns)*(Nwc/Nw))*((fw/(fw+fs))*(pF[0,0]*(eps1**2+eps**2) + (1-pF[0,0])*aeps(s1[0],eps))+
                                                     (fs/(fw+fs))*(pF[0,1]*(eps1**2+eps**2) + (1 - pF[0,1])*aeps(s1[0],eps)))+
                                ((Nsd/Ns)*(Nwd/Nw))*((fw/(fw+fs))*(pF[0,0]*(2*eps1*eps) + (1-pF[0,0]*aeps(s1[0],eps)))+
                                                     (fs/(fw+fs))*(pF[0,1]*(2*eps1*eps) + (1-pF[0,1]*aeps(s1[0],eps))))+
                                ((Nsd/Ns)*(Nwc/Nw))*((fw/(fw+fs))*(pF[0,0]*(eps1**2+eps**2) + (1-pF[0,0])*aeps(s1[0],eps))+
                                                     (fs/(fw+fs))*(pF[0,1]*(2*eps1*eps) + (1-pF[0,1])*aeps(s1[0],eps))) + 
                                ((Nsc/Ns)*(Nwd/Nw))*((fs/(fw+fs))*(pF[0,1]*(eps1**2+eps**2) + (1-pF[0,1])*aeps(s1[0],eps))+
                                                     (fw/(fw+fs))*(pF[0,0]*(2*eps1*eps) + (1-pF[0,0])*aeps(s1[0],eps)))
                            )
                        ) + pS*( # focus player is strong
                            (1/Ns)*aeps(s1[1],eps)+
                            (1-(1/Ns))*(
                                ((Nsc/Ns)*(Nwc/Nw))*((fw/(fw+fs))*(pF[1,0]*(eps1**2+eps**2) + (1-pF[1,0])*aeps(s1[1],eps))+
                                                     (fs/(fw+fs))*(pF[1,1]*(eps1**2+eps**2) + (1 - pF[0,1])*aeps(s1[1],eps)))+
                                ((Nsd/Ns)*(Nwd/Nw))*((fw/(fw+fs))*(pF[1,0]*(2*eps1*eps) + (1-pF[1,0]*aeps(s1[1],eps)))+
                                                     (fs/(fw+fs))*(pF[1,1]*(2*eps1*eps) + (1-pF[1,1]*aeps(s1[1],eps))))+
                                ((Nsd/Ns)*(Nwc/Nw))*((fw/(fw+fs))*(pF[1,0]*(eps1**2+eps**2) + (1-pF[1,0])*aeps(s1[1],eps))+
                                                     (fs/(fw+fs))*(pF[1,1]*(2*eps1*eps) + (1-pF[1,1])*aeps(s1[1],eps))) + 
                                ((Nsc/Ns)*(Nwd/Nw))*((fs/(fw+fs))*(pF[1,1]*(eps1**2+eps**2) + (1-pF[1,1])*aeps(s1[1],eps))+
                                                     (fw/(fw+fs))*(pF[1,0]*(2*eps1*eps) + (1-pF[1,0])*aeps(s1[1],eps)))
                            )
                        )   

                benefit = (
                    (pW**N)*benefit_ww + # everyone is weak
                    (pS**N)*benefit_ss + # everyone is strong
                    ((pW**(N-1)*pS*N))*( # one strong everyone else weak
                        (((N-1)*fw)/((N-1)*fw+fs)*((N-2)*fw)/((N-2)*fw+fs))*benefit_ww +
                        (((N-1)*fw)/((N-1)*fw+fs)*(fs)/((N-2)*fw+fs))*benefit_sw+
                        ((fs)/((N-1)*fw+fs)*((N-1)*fw)/((N-1)*fw))*benefit_sw
                    )+
                    ((pS**(N-1)*pW*N))*( # one weak everyone else strong
                        (((N-1)*fs)/((N-1)*fs+fw)*((N-2)*fs)/((N-2)*fs+fw))*benefit_ss +
                        (((N-1)*fs)/((N-1)*fs+fw)*(fw)/((N-2)*fs+fw))*benefit_sw+
                        ((fw)/((N-1)*fs+fw)*((N-1)*fs)/((N-1)*fs))*benefit_sw
                    )
                )

                for nums in range(2, N-1):
                    numw = N - nums
                    benefit += (pW**numw * pS**nums)*(math.factorial(N)/(math.factorial(nums)*math.factorial(N-nums)))*(
                        ((numw*fw)/(numw*fw+nums*fs)*((numw-1)*fw)/((numw-1)*fw+nums*fs))*benefit_ww+
                        ((nums*fs)/(numw*fw+nums*fs)*((nums-1)*fs)/(numw*fw+(nums-1)*fs))*benefit_ss+
                        ((numw*fw)/(numw*fw+nums*fs)*(nums*fs)/((numw-1)*fw+nums*fs))*benefit_sw+
                        ((nums*fs)/(numw*fs+nums*fs)*(numw*fw)/(numw*fw+(nums-1)*fs))*benefit_sw
                    )

                cost = (
                    (pW**N)*cost_ww + # everyone is weak
                    (pS**N)*cost_ss + # everyone is strong
                    ((pW**(N-1)*pS*N))*( # one strong everyone else weak
                        (((N-1)*fw)/((N-1)*fw+fs)*((N-2)*fw)/((N-2)*fw+fs))*cost_ww +
                        (((N-1)*fw)/((N-1)*fw+fs)*(fs)/((N-2)*fw+fs))*cost_sw+
                        ((fs)/((N-1)*fw+fs)*((N-1)*fw)/((N-1)*fw))*cost_sw
                    )+
                    ((pS**(N-1)*pW*N))*( # one weak everyone else strong
                        (((N-1)*fs)/((N-1)*fs+fw)*((N-2)*fs)/((N-2)*fs+fw))*cost_ss +
                        (((N-1)*fs)/((N-1)*fs+fw)*(fw)/((N-2)*fs+fw))*cost_sw+
                        ((fw)/((N-1)*fs+fw)*((N-1)*fs)/((N-1)*fs))*cost_sw
                    )
                )

                for nums in range(2, N-1):
                    numw = N - nums
                    cost += (pW**numw * pS**nums)*(math.factorial(N)/(math.factorial(nums)*math.factorial(N-nums)))*(
                        ((numw*fw)/(numw*fw+nums*fs)*((numw-1)*fw)/((numw-1)*fw+nums*fs))*cost_ww+
                        ((nums*fs)/(numw*fw+nums*fs)*((nums-1)*fs)/(numw*fw+(nums-1)*fs))*cost_ss+
                        ((numw*fw)/(numw*fw+nums*fs)*(nums*fs)/((numw-1)*fw+nums*fs))*cost_sw+
                        ((nums*fs)/(numw*fs+nums*fs)*(numw*fw)/(numw*fw+(nums-1)*fs))*cost_sw
                    )                

                if benefit > M:
                    WCD[i,j,k,0] = benefit/N

                WCD[i,j,k,1] = cost
                
    # WCD[1,0,:]=-999
    # WCD[0,N,:]=-999

    return WCD 
    
    