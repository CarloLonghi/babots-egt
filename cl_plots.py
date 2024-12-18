import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

def gaussian(x, mu, sig):
    return np.exp(-((x - mu) ** 2) / (2 * (sig ** 2))) / (np.sqrt(2 * np.pi) * sig)

def fermi(x, beta):
    return 1 / (1 + np.exp(-beta * x))

file = 'results/multileader/cl/res_4strats_M0_f0_sdist_test'
data = np.load(file + '.npy')

nr = 2
nc = 5
fntsize=15

#muv=[0, 0.25, 0.5, 0.75, 1.]
N = 9
eps = 0.01
eps1 = 1 - eps
rv=np.linspace(1,10,num=10)
pSv=np.linspace(-8, 8, num=50)
deltaLv=[0, 1, 2, 4, 8]


fig,axs=plt.subplots(nrows=nr, ncols=nc, sharex='all', sharey='all', figsize=(10,5))
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

                # strengths = np.array([0.3, 0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.7])
                # strengths = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9])
                x = np.linspace(-deltaL, deltaL, N)
                strengths = 1 / (1 + np.exp(-(x + pS)))

                s_diff = np.array([[strengths[i] - strengths[j] for j in range(N)] for i in range(N)])

                actions = np.array([strengths[s] * stratS + (1 - strengths[s]) * stratW for s in range(N)])

                b = np.zeros(N)

                others = np.array([[strengths[j] for j in range(N) if j != i] for i in range(N)])
                leader_choice = np.array([(others[i] * others[i]) / sum(others[i] * others[i]) for i in range(N)])
                #leader_choice = np.array([[leader_choice[i] for i in range(N) if i != j] for j in range(N)])
                leader_actions = np.array([[actions[j] for j in range(N) if j != i] for i in range(N)])
                diff = np.array([[s_diff[leader][p] for p in range(N) if p != leader] for leader in range(N)])
                fp = 1 / (1 + np.exp(-(diff)))
                p1leader = 1 - np.prod(1 - strengths)
                following = np.expand_dims((1 - strengths), 1) * p1leader * leader_choice * fp * (
                    leader_actions * (eps1**2 + eps**2) + (1 - leader_actions) * (2 * eps1 * eps))
                following = np.sum(following, axis=1)
                not_following = np.expand_dims((1 - strengths), 1) * p1leader * leader_choice * (1 - fp) * (
                    np.expand_dims(actions, 1) * eps1 + np.expand_dims((1 - actions), 1) * eps)
                not_following = np.sum(not_following, axis=1)
                no_leaders = (1 - strengths) * (1 - p1leader) * (actions * eps1 + (1 - actions) * eps)
                leading = strengths * (actions * eps1 + (1 - actions) * eps) 
                b = np.sum(leading + not_following + following + no_leaders)
                    
                cl = b / N

                res[idps] += cl * data[idr, iddl, idps, strat]
            

        ax.set_xticks(np.linspace(0, pSv.shape[0]-1, nticksX))
        ax.set_xticklabels(np.linspace(pSv[0],pSv[-1],nticksX), fontsize=12)
        ax.set_yticks(np.linspace(0, 1, 3))
        ax.set_yticklabels(np.linspace(0,1,3), fontsize=12)
        ax.set_ylim(0.0, 1.0)
        ax.plot(res, label='$\delta_L=%d$'%deltaL, color=cmap((iddl)/(len(deltaLv))))
        #ax.plot(res)

        if i==nr-1: ax.set_xlabel('s_center', fontsize=fntsize)
        if j==0: ax.set_ylabel(r'cooperation level', fontsize=fntsize)
        ax.text(20,1.06,"$r$=%d" % rv[idr], size=13)

legend_elements = [Line2D([], [], marker='None', label='s_width', linestyle='None')]
legend_elements += [Line2D([], [], marker='s', color=cmap((idx)/(len(deltaLv)+1)), label='%.2f'%deltaLv[idx],
                          markerfacecolor=cmap((idx)/(len(deltaLv)+1)), markersize=10, linestyle='None') for idx in range(len(deltaLv))]
plt.legend( loc='upper center', bbox_to_anchor=(-2., -0.6),
          fancybox=True, shadow=False, ncol=7, columnspacing=0.0, handles=legend_elements,handletextpad=-0.3,fontsize=13)
plt.savefig('multileader_fig_sdist_test3.png', bbox_inches='tight', dpi=300)

plt.show()