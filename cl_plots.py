import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

def gaussian(x, mu, sig):
    return np.exp(-((x - mu) ** 2) / (2 * (sig ** 2))) / (np.sqrt(2 * np.pi) * sig)

def fermi(x, beta):
    return 1 / (1 + np.exp(-beta * x))

file = 'results/multileader/cl/res_4strats_M0_f0_sdist_update'
data = np.load(file + '.npy')

nr = 5
nc = 2
fntsize=15

#muv=[0, 0.25, 0.5, 0.75, 1.]
betav=np.linspace(-5,5,num=50)
N = 9
eps = 0.01
eps1 = 1 - eps
rv=np.linspace(1,10,num=10)
deltaLv=[0, 1, 2, 4, 8]
pSv=np.linspace(0,1,num=50)


fig,axs=plt.subplots(nrows=nr, ncols=nc, sharex='all', sharey='all', figsize=(5,12))
fig.subplots_adjust(hspace=0.4, wspace=0.2)
nticksY=6
nticksX=3

cmap = plt.get_cmap('viridis')


for idr, r in enumerate(rv):
    i = idr // nc
    j = idr % nc
    ax=axs[i,j]
    for iddl, deltaL in enumerate(deltaLv):
        res = np.zeros((pSv.shape[0]))
        for strat in range(4):
            for idps, pS in enumerate(pSv):
                
                stratW = strat%2
                stratS = strat//2

                # set N levels of strength drawn from a prob. dist.
                numS = int(np.round(N * pS))
                numW = int(np.round(N * (1 - pS)))
                s = 1 / (1 + np.exp(-deltaL))
                w = 1 / (1 + np.exp(deltaL))
                strengths = np.array([s,] * numS + [w,] * numW)

                s_diff = np.array([[strengths[i] - strengths[j] for j in range(N)] for i in range(N)])
                p_leader = strengths / sum(strengths)

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

                # actions = np.array([strengths[s] * stratS + (1 - strengths[s]) * stratW for s in range(N)])
                actions = np.array([stratS for _ in range(numS)] + [stratW for _ in range(numW)])

                b = np.zeros(N)

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
                    
                cl = b / N

                res[idps] += cl * data[idr, iddl, idps, strat]
            

        ax.set_xticks(np.linspace(0, pSv.shape[0]-1, nticksX))
        ax.set_xticklabels(np.linspace(pSv[0],pSv[-1],nticksX), fontsize=12)
        ax.set_yticks(np.linspace(0, 1, 3))
        ax.set_yticklabels(np.linspace(0,1,3), fontsize=12)
        ax.set_ylim(0.0, 1.0)
        ax.plot(res, label='$\delta_L=%d$'%deltaL, color=cmap((iddl)/(len(deltaLv))))
        #ax.plot(res)

        if i==nr-1: ax.set_xlabel(r'$p_s$', fontsize=fntsize)
        if j==0 and i==nr//2: ax.set_ylabel(r'cooperation level', fontsize=fntsize)
        ax.text(20,1.06,"$r$=%d" % rv[idr], size=13)

legend_elements = [Line2D([], [], marker='None', label='Leader: $\Delta_l=\Delta_f$', linestyle='None')]
legend_elements += [Line2D([], [], marker='s', color=cmap((idx)/(len(deltaLv)+1)), label='%d'%deltaLv[idx],
                          markerfacecolor=cmap((idx)/(len(deltaLv)+1)), markersize=10, linestyle='None') for idx in range(len(deltaLv))]
plt.legend( loc='upper center', bbox_to_anchor=(0., -0.6),
          fancybox=True, shadow=False, ncol=7, columnspacing=0.0, handles=legend_elements,handletextpad=-0.3,fontsize=13)
plt.savefig('multileader_fig_baseline.png', bbox_inches='tight', dpi=300)

plt.show()