#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:40:35 2019

@author: abraxas
"""

import evoEGT as evo

import numpy as np
from itertools import combinations
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

def calcWCD(N,eps,beta,pS,deltaL,M):
# Input: N group size, eps error when trying to perform an action, r multiplicative constant for the PGG (assuming c=1), pF probability of following leader, M number of individuals that need to cooperate in order to get any benefit
# Output: WCD[i,k,ip] payoffs (i=0 defector, i=1 cooperator; k number of cooperators in the group; ip coef associated to the parameter payoffs r (ip=0) and c (ip=1))
    WCD=np.zeros((4,4,N+1,2))
    eps1=1.-eps

    x = np.linspace(-deltaL, deltaL, N)
    strengths = 1 / (1 + np.exp(-(x + pS)))

    s_diff = np.array([[strengths[i] - strengths[j] for j in range(N)] for i in range(N)])
    p1leader = 1 - np.prod(1 - strengths)

    for i in range(4):
        s1=[i%2,i//2] # s:[w,s], 0:[0,0], 1:[1,0], 2:[0,1], 3:[1,1]
        for j in range(4):
            s2=[j%2,j//2]
            for k in range(0,N+1): # k: number of players following strategy s1

                combs = np.array(list(combinations(range(N), k)))
                benefit = 0
                cost = 0
                for s1_idx in combs:

                    strategies = [1 if s in s1_idx else 0 for s in range(N)]
                    actions = np.array([strengths[s] * s1[1] + (1 - strengths[s]) * s1[0] if strategies[s] == 1
                               else strengths[s] * s2[1] + (1 - strengths[s]) * s2[0]
                               for s in range(N)])
                    
                    others = np.array([[strengths[j] for j in range(N) if j != i] for i in range(N)])
                    leader_choice = np.array([(others[i] * others[i]) / sum(others[i] * others[i]) for i in range(N)])
                    leader_actions = np.array([[actions[j] for j in range(N) if j != i] for i in range(N)])
                    diff = np.array([[s_diff[i][j] for i in range(N) if i != j] for j in range(N)])
                    fp = 1 / (1 + np.exp(-(diff)))
                    following = np.expand_dims((1 - strengths), 1) * p1leader * leader_choice * fp * (
                        leader_actions * (eps1**2 + eps**2) + (1 - leader_actions) * (2 * eps1 * eps))
                    following = np.sum(following, axis=1)
                    not_following = np.expand_dims((1 - strengths), 1) * p1leader * leader_choice * (1 - fp) * (
                        np.expand_dims(actions, 1) * eps1 + np.expand_dims((1 - actions), 1) * eps)
                    not_following = np.sum(not_following, axis=1)
                    no_leaders = (1 - strengths) * (1 - p1leader) * (actions * eps1 + (1 - actions) * eps)
                    leading = strengths * (actions * eps1 + (1 - actions) * eps) 
                    b = np.sum(leading + not_following + following + no_leaders)
                    

                    leading = strategies * leading
                    focus_actions = actions * strategies
                    leader_choice = np.expand_dims(strategies, 1) * leader_choice
                    diff = np.array([[s_diff[i][j] * strategies[j] for i in range(N) if j != i] for j in range(N)])
                    fp = 1 / (1 + np.exp(-(diff)))
                    # leader_choice = (strengths * strengths) / sum(strengths * strengths)
                    # leader_choice = np.array([[leader_choice[i] * strategies[j] for i in range(N) if i != j] for j in range(N)])
                    not_following = np.expand_dims((1 - strengths) * strategies, 1) * p1leader * leader_choice * (1 - fp) * (
                        np.expand_dims(focus_actions, 1) * eps1 + np.expand_dims((1 - focus_actions), 1) * eps)
                    not_following = np.sum(not_following, axis=1)
                    following = np.expand_dims((1 - strengths) * strategies, 1) * p1leader * leader_choice * fp * (
                        leader_actions * (eps1**2 + eps**2) + (1 - leader_actions) * (2 * eps1 * eps))
                    following = np.sum(following, axis=1)
                    no_leaders = (1 - strengths) * (1 - p1leader) * (focus_actions * eps1 + (1 - focus_actions) * eps)
                    c = np.sum(leading + following + not_following + no_leaders)
                    if k > 0:
                        c /= k

                    benefit += b
                    cost += c
                        

                benefit /= combs.shape[0]
                cost /= combs.shape[0]

                # TODO consider the case in which the leaders that are chosen vote to agree on a single action and then the followers follow this action

                ## each player has prob of being s1 as k/N, prob of leading and of following chosen correspondingly among the N already calculated.

                # TODO how do you define a strong or a weak player if the strength level is not 1 or 0
                ## could be a threshold over/under 0.5
                ## could be relative the strength of the other players in the groups
                ## could be a probabilistic strategy, so act strong with a probability that is proportional to the strength              

                if benefit > M:
                    WCD[i,j,k,0] = benefit/N

                if benefit > N:
                    print('ao')

                WCD[i,j,k,1] = cost
                
    # WCD[1,0,:]=-999
    # WCD[0,N,:]=-999

    return WCD 
    
def gaussian(x, mu, sig):
    return np.exp(-((x - mu) ** 2) / (2 * (sig ** 2))) / (np.sqrt(2 * np.pi) * sig)

def beta_dist(x, a, b):
    return math.gamma(a + b) / math.gamma(a) / math.gamma(b) * (x**(a - 1)) * ((1 - x)**(b - 1))

def fermi(x, beta):
    return 1 / (1 + np.exp(-beta * x))