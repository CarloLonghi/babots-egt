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

    # set N levels of strength drawn from a prob. dist.
    # x = np.linspace(0, 1, N - 1)
    # #y = 1 - gaussian(x, 0.5, 0.5)
    # y = 1 - gaussian(x, mu, sigma)
    # y = y / sum(y) * 0.8
    # y = y[:-1]
    # strengths = np.array([0.1,])
    # for step in y:
    #     strengths = np.append(strengths, strengths[-1] + step)
    # strengths = np.append(strengths, 0.9)

    # alfa, beta = ab
    # x = np.linspace(0.00001, 0.99999, 8)
    # y = beta_dist(x, alfa, beta)
    # y -= min(y)
    # y /= max(y)
    # y = 1 - y
    # y = y / sum(y) * 0.8
    # strengths = np.array([0.1])
    # for step in y:
    #     strengths = np.append(strengths, strengths[-1] + step)

    # x = np.linspace(-2, 2, N)
    # strengths = fermi(x, beta)

    numS = int(np.round(N * pS))
    numW = int(np.round(N * (1 - pS)))
    s = 1 / (1 + np.exp(-deltaL))
    w = 1 / (1 + np.exp(deltaL))
    strengths = np.array([s,] * numS + [w,] * numW)

    s_diff = np.array([[strengths[i] - strengths[j] for j in range(N)] for i in range(N)])
    s_diff = np.array([[1/(1+np.exp(0)) for j in range(numS)] + [1/(1+np.exp(-deltaL)) for j in range(numS, numS+numW)] for i in range(numS)] +
                      [[1/(1+np.exp(deltaL)) for j in range(numS)] + [1/(1+np.exp(0)) for j in range(numS, numS + numW)] for i in range(numS, numS + numW)])
    follow_prob = np.zeros((N, N))
    for i in range(numS):
        for j in range(numS):
            follow_prob[i, j] = 1/(1+np.exp(0))
        for j in range(numS, numS + numW):
            follow_prob[i, j] = 1/(1+np.exp(-deltaL))
    for i in range(numS, numS + numW):
        for j in range(numS):
            follow_prob[i, j] = 1/(1+np.exp(deltaL))
        for j in range(numS, numS + numW):
            follow_prob[i, j] = 1/(1+np.exp(0))    

    p_leader = strengths / sum(strengths)

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
                    # actions = np.array([strengths[s] * s1[1] + (1 - strengths[s]) * s1[0] if strategies[s] == 1
                    #            else strengths[s] * s2[1] + (1 - strengths[s]) * s2[0]
                    #            for s in range(N)])
                    actions = np.array([s1[1] if strategies[s] == 1 else s2[1] for s in range(numS)] + 
                                       [s1[0] if strategies[s] == 1 else s2[0] for s in range(numS, numS + numW)])
                    
                    # leader_actions = np.expand_dims(actions, axis=1)
                    # other_actions = np.array([[actions[p] for p in range(N) if p != leader] for leader in range(N)])
                    # diff = np.array([[s_diff[leader][p] for p in range(N) if p != leader] for leader in range(N)])
                    # follow_prob = 1 / (1 + np.exp(-diff))
                    leader_choice = (strengths * strengths) / sum(strengths * strengths)
                    leader_choice = np.array([[leader_choice[i] for i in range(N) if i != j] for j in range(N)])
                    leader_actions = np.array([[actions[j] for j in range(N) if j != i] for i in range(N)])
                    diff = np.array([[s_diff[leader][p] for p in range(N) if p != leader] for leader in range(N)])
                    # follow_prob = 1 / (1 + np.exp(-diff))
                    # fp = np.array([[follow_prob[i, j] for j in range(N) if i != j] for i in range(N)])
                    fp = np.array([[follow_prob[i, j] for i in range(N) if i != j] for j in range(N)])
                    following = np.expand_dims((1 - strengths), 1) * leader_choice * fp * (
                        leader_actions * (eps1**2 + eps**2) + (1 - leader_actions) * (2 * eps1 * eps))
                    following = np.sum(following, axis=1)
                    not_following = np.expand_dims((1 - strengths), 1) * leader_choice * (1 - fp) * (
                        np.expand_dims(actions, 1) * eps1 + np.expand_dims((1 - actions), 1) * eps)
                    not_following = np.sum(not_following, axis=1)
                    leading = strengths * (actions * eps1 + (1 - actions) * eps) 
                    b = np.sum(leading + not_following + following)
                    

                    #leader_actions = np.expand_dims(actions, 1)
                    leader_actions = np.array([[actions[j] for j in range(N) if j != i] for i in range(N)])
                    leading = strategies * strengths * (actions * eps1 + (1 - actions) * eps)
                    focus_actions = actions * strategies
                    leader_choice = (strengths * strengths) / sum(strengths * strengths)
                    leader_choice = np.array([[leader_choice[i] * strategies[j] for i in range(N) if i != j] for j in range(N)])
                    diff = np.array([[s_diff[leader][p] * strategies[p] for p in range(N) if p != leader] for leader in range(N)])
                    # follow_prob = 1 / (1 + np.exp(-diff))
                    fp = np.array([[follow_prob[leader, j] * strategies[j] for leader in range(N) if j != leader] for j in range(N)])
                    not_following = np.expand_dims((1 - strengths) * strategies, 1) * leader_choice * (1 - fp) * (
                        np.expand_dims(focus_actions, 1) * eps1 + np.expand_dims((1 - focus_actions), 1) * eps)
                    not_following = np.sum(not_following, axis=1)
                    following = np.expand_dims((1 - strengths) * strategies, 1) * leader_choice * fp * (
                        leader_actions * (eps1**2 + eps**2) + (1 - leader_actions) * (2 * eps1 * eps))
                    following = np.sum(following, axis=1)
                    c = np.sum(leading + following + not_following)
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