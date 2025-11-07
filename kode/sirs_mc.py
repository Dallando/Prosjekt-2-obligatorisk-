import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

def sirs_mc(a, b, c, N=400, S0=300, I0=100, R0=0, T=120, M=200):
    dt = min(4/(a*N), 1/(b*N), 1/(c*N))
    steps = int(T/dt)
    traj_mean = np.zeros((steps+1,3))
    traj_std  = np.zeros((steps+1,3))

    for m in range(M):
        S, I, R = S0, I0, R0
        path = np.zeros((steps+1,3))
        path[0] = (S,I,R)
        for n in range(steps):
            p_SI = a*S*I/N*dt
            p_IR = b*I*dt
            p_RS = c*R*dt
            SI = rng.binomial(S, p_SI/S if S>0 else 0.0)
            IR = rng.binomial(I, p_IR/I if I>0 else 0.0)
            RS = rng.binomial(R, p_RS/R if R>0 else 0.0)
            S = S - SI + RS
            I = I + SI - IR
            R = R + IR - RS
            path[n+1] = (S,I,R)
        traj_mean += path
        traj_std  += path**2

    traj_mean /= M
    traj_std = np.sqrt(traj_std/M - traj_mean**2)
    t = np.linspace(0, steps*dt, steps+1)
    return t, traj_mean, traj_std

if __name__ == "__main__":
    t, m, s = sirs_mc(1.0, 0.25, 0.1)
    plt.plot(t, m[:,1])
    plt.fill_between(t, m[:,1]-s[:,1], m[:,1]+s[:,1], alpha=0.3)
    plt.xlabel("Tid [dager]"); plt.ylabel("Smittede")
    plt.title("SIRS Monte Carlo")
    plt.savefig("rapport/figs/sirs_mc.png")
    plt.show()
