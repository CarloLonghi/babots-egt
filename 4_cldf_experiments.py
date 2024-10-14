import numpy as np
import evoEGT as evo
from heterogeneous4 import calcH, calcWCD

def coop_pF_r(r,M,N,HZ,beta,eps,pSv,deltaFv,deltaLv,f,betaF):
# Input: pFv, rv, Mv (vectors with values of pF, r, and M), N, HZ (H or Z), beta, eps
# Output: matrix with the fraction of cooperators as a function of pF and r
    if np.isscalar(HZ):
        H=calcH(N-1,HZ-1)

    MAT = np.zeros((len(deltaLv), len(deltaFv), len(pSv), 4))

    for iddl, deltaL, in enumerate(deltaLv):
        for iddf, deltaF in enumerate(deltaFv):
            for idps, pS in enumerate(pSv):
                pF=np.zeros((2,2))
                # pF[0,0] = 1/(1+np.exp(-betaF*(f+deltaF)))
                # pF[1,1] = 1/(1+np.exp(-betaF*(f-deltaF)))
                # pF[0,1] = 1/(1+np.exp(-betaF*(f+2*deltaF)))
                # pF[1,0] = 1/(1+np.exp(-betaF*(f-2*deltaF)))

                pF[0,0] = 1/(1+np.exp(-betaF*(f)))
                pF[1,1] = 1/(1+np.exp(-betaF*(f)))
                pF[0,1] = 1/(1+np.exp(-betaF*(f+deltaF)))
                pF[1,0] = 1/(1+np.exp(-betaF*(f-deltaF)))

                WCD=calcWCD(N,eps,pF,deltaL,pS,M)
                #Wgen=transfW2Wgen(WCD) # transforming to evoEGT format
                print(r,pS)
                SD,fixM = evo.Wgroup2SD(WCD,H,[r,-1.],beta,infocheck=False)
                MAT[iddl, iddf, idps, :] = SD[:,0]
    return MAT


if __name__ == "__main__":

    import time

    t0=time.time()

####### Plot heatmap #########################################
    eps=0.01 #0.01
    Z=100
    N=9
    beta=1.
    M=0
    f=0
    betaF=1
    r=5

    deltaLv=[0, 1, 2, 4]
    deltaFv=[0, 1, 2, 4, 8]
    pSv=np.linspace(0,1.,num=50)
    
    labfilenpy='results/h4/cl/res_4strats_M0_f0_r5_dldf'
    MAT=coop_pF_r(r,M,N,Z,beta,eps,pSv,deltaFv,deltaLv,f,betaF)
    np.save(labfilenpy,MAT)             # save matrix for heatmap
    print('data saved to file!')
    
#####################################################