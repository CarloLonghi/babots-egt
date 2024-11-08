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


fig,axs=plt.subplots(nrows=nr, ncols=nc, sharex='all', sharey='all', figsize=(5,12))
fig.subplots_adjust(hspace=0.4, wspace=0.2)
nticksY=6
nticksX=3

cmap = plt.get_cmap('viridis')


for idr, r in enumerate(rv):
    i = idr // nc
    j = idr % nc
    ax=axs[i,j]

    res = np.zeros((betav.shape[0]))
    for strat in range(4):
        for idb, beta in enumerate(betav):
            
            stratW = strat%2
            stratS = strat//2

            # set N levels of strength drawn from a prob. dist.
            x = np.linspace(-2, 2, N)
            strengths = fermi(x, beta)       

            s_diff = [[strengths[i] - strengths[j] for j in range(N)] for i in range(N)]
            p_leader = strengths / sum(strengths)

            actions = [strengths[s] * stratS + (1 - strengths[s]) * stratW for s in range(N)]

            b = np.zeros(N)

            for leader in range(N):
                leader_action = actions[leader]
                other_actions = np.array([actions[p] for p in range(N) if p != leader])
                diff = np.array([s_diff[leader][p] for p in range(N) if p != leader])
                follow_prob = 1 / (1 + np.exp(-diff))
                not_following = sum((1 - follow_prob) * (other_actions * eps1 + (1 - other_actions) * eps))
                following = sum(follow_prob * (leader_action * (eps1**2 + eps**2) + (1 - leader_action) * (2 * eps1 * eps)))
                b[leader] = (leader_action * eps1 + (1 - leader_action) * eps +
                                not_following + following)
                
            cl = sum(b * p_leader)
            cl = cl / N

            res[idb] += cl * data[idr, idb, strat]
        

    ax.set_xticks(np.linspace(0, betav.shape[0]-1, nticksX))
    ax.set_xticklabels(np.linspace(betav[0],betav[-1],nticksX), fontsize=12)
    ax.set_yticks(np.linspace(0, 1, 3))
    ax.set_yticklabels(np.linspace(0,1,3), fontsize=12)
    ax.set_ylim(0.0, 1.0)
    #ax.plot(res, label='$\mu=%d$'%mu, color=cmap((idm+1)/(len(muv)+1)))
    ax.plot(res)

    if i==nr-1: ax.set_xlabel(r'$\sigma$', fontsize=fntsize)
    if j==0 and i==nr//2: ax.set_ylabel(r'cooperation level', fontsize=fntsize)
    ax.text(20,1.06,"$r$=%d" % rv[idr], size=13)

# legend_elements = [Line2D([], [], marker='None', label='Leader: $\mu$', linestyle='None')]
# legend_elements += [Line2D([], [], marker='s', color=cmap((idx)/(len(muv)+1)), label='%d'%muv[idx],
#                           markerfacecolor=cmap((idx)/(len(muv)+1)), markersize=10, linestyle='None') for idx in range(len(muv))]
# plt.legend( loc='upper center', bbox_to_anchor=(0., -0.6),
#           fancybox=True, shadow=False, ncol=7, columnspacing=0.0, handles=legend_elements,handletextpad=-0.3,fontsize=13)
plt.savefig('multileader_fig_fermi.png', bbox_inches='tight', dpi=300)

plt.show()