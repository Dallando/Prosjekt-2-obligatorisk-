
import numpy as np
import matplotlib.pyplot as plt

def sirs_rhs(t, y, a, b, c, N):
    S, I, R = y
    dS = c * R - (a * S * I) / N
    dI = (a * S * I) / N - b * I
    dR = b * I - c * R
    return np.array([dS, dI, dR])

def rk4_step(f, t, y, h, *args):
    k1 = f(t, y, *args)
    k2 = f(t + 0.5*h, y + 0.5*h*k1, *args)
    k3 = f(t + 0.5*h, y + 0.5*h*k2, *args)
    k4 = f(t + h, y + h*k3, *args)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def simulate_sirs(a, b, c, N=400, S0=300, I0=100, R0=0, T=120, h=0.1):
    t = np.arange(0, T+h, h)
    Y = np.zeros((len(t), 3))
    Y[0] = [S0, I0, R0]
    for n in range(len(t)-1):
        Y[n+1] = rk4_step(sirs_rhs, t[n], Y[n], h, a, b, c, N)
    return t, Y

if __name__ == "__main__":
    params = [
        {"a":1.0, "b":1/4, "c":0.1},
        {"a":1.0, "b":1/2, "c":0.1},
        {"a":1.0, "b":3/4, "c":0.1},
        {"a":1.0, "b":1.0, "c":0.1},
    ]
    fig, ax = plt.subplots(2, 2, figsize=(10,6))
    ax = ax.ravel()
    for i, p in enumerate(params):
        t, Y = simulate_sirs(**p)
        ax[i].plot(t, Y[:,0], label="S")
        ax[i].plot(t, Y[:,1], label="I")
        ax[i].plot(t, Y[:,2], label="R")
        ax[i].set_title(f"a={p['a']}, b={p['b']}, c={p['c']}")
        ax[i].legend(); ax[i].grid()
    plt.tight_layout()
    plt.savefig("rapport/figs/sirs_ode.png")
    plt.show()
