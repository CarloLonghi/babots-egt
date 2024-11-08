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

def calcWCD(N,eps,beta,M):
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

    x = np.linspace(-2, 2, N)
    strengths = fermi(x, beta)

    s_diff = np.array([[strengths[i] - strengths[j] for j in range(N)] for i in range(N)])
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
                    actions = np.array([strengths[s] * s1[1] + (1 - strengths[s]) * s1[0] if strategies[s] == 1
                               else strengths[s] * s2[1] + (1 - strengths[s]) * s2[0]
                               for s in range(N)])
                    
                    b = np.zeros(N)
                    c = np.zeros(N)

                    leader_actions = np.expand_dims(actions, axis=1)
                    other_actions = np.array([[actions[p] for p in range(N) if p != leader] for leader in range(N)])
                    diff = np.array([[s_diff[leader][p] for p in range(N) if p != leader] for leader in range(N)])
                    follow_prob = 1 / (1 + np.exp(-diff))
                    not_following = (1 - follow_prob) * (other_actions * eps1 + (1 - other_actions) * eps)
                    not_following = np.sum(not_following, axis=1)
                    following = follow_prob * (leader_actions * (eps1**2 + eps**2) + (1 - leader_actions) * (2 * eps1 * eps))
                    following = np.sum(following, axis=1)
                    b = actions * eps1 + (1 - actions) * eps + not_following + following

                    # if k > 0:
                    #     c[s1_idx] += (actions * eps1 + (1 - actions) * eps)[s1_idx]
                    #     focus_players = [[p for p in s1_idx if p != leader] for leader in range(N)]
                    #     focus_actions = np.array([[actions[p] for p in focus_players[leader]] for leader in range(N)])
                    #     diff = np.array([[s_diff[leader][p] for p in focus_players[leader]] for leader in range(N)])
                    #     follow_prob = 1 / (1 + np.exp(-diff))
                    #     not_following = (1 - follow_prob) * (focus_actions * eps1 + (1 - focus_actions) * eps)
                    #     not_following = np.sum(not_following, axis=1)
                    #     following = follow_prob * (leader_actions * (eps1**2 + eps**2) + (1 - leader_actions) * (2 * eps1 * eps))
                    #     following = np.sum(following, axis=1)
                    #     p_cost += not_following + following
                    #     c += p_cost / k
                    

                    for leader in range(N):
                        leader_action = actions[leader]
                        p_cost = 0
                        if leader in s1_idx:
                            p_cost += leader_action * eps1 + (1 - leader_action) * eps
                        focus_players = [p for p in s1_idx if p != leader]
                        focus_actions = actions[focus_players]
                        diff = np.array([s_diff[leader][p] for p in focus_players])
                        follow_prob = 1 / (1 + np.exp(-diff))
                        not_following = sum((1 - follow_prob) * (focus_actions * eps1 + (1 - focus_actions) * eps))
                        following = sum(follow_prob * (leader_action * (eps1**2 + eps**2) + (1 - leader_action) * (2 * eps1 * eps)))
                        p_cost += not_following + following

                        if k > 0:
                            c[leader] = p_cost / k
                        
                    benefit += sum(b * p_leader)
                    cost += sum(c * p_leader)
                        

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