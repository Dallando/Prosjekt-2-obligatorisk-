import numpy as np
import matplotlib.pyplot as plt

def sirs_vital_rhs(t, y, a, b, c, e, d, dI):
    S, I, R = y
    N = S + I + R
    dS = c*R - (a*S*I)/N - d*S + e*N
    dI = (a*S*I)/N - b*I - d*I - dI*I
    dR = b*I - c*R - d*R
    return np.array([dS, dI, dR])

# RK4 som i sirs_model.py
from sirs_model import rk4_step

def simulate_sirs_vital(a=1, b=0.25, c=0.1, e=2.9e-5, d=2.2e-5, dI=1e-4,
                        S0=300, I0=100, R0=0, T=365, h=0.1):
    t = np.arange(0, T+h, h)
    Y = np.zeros((len(t), 3))
    Y[0] = [S0, I0, R0]
    for n in range(len(t)-1):
        Y[n+1] = rk4_step(sirs_vital_rhs, t[n], Y[n], h, a, b, c, e, d, dI)
    return t, Y

if __name__ == "__main__":
    t,Y = simulate_sirs_vital()
    plt.plot(t,Y[:,1])
    plt.xlabel("Dager"); plt.ylabel("Smittede")
    plt.title("SIRS med vitaldynamikk")
    plt.savefig("rapport/figs/sirs_vital.png")
    plt.show()
