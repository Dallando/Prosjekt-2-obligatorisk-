
from __future__ import annotations
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    from numba import njit
    JIT = True
except Exception:
    # Kjør uten numba hvis ikke tilgjengelig
    def njit(func=None, **_):
        return (lambda f: f) if func is None else func
    JIT = False

# Grunnleggende konstanter
g = 9.81  # [m/s^2]
EPS = 1e-12   # for å unngå deling på null
H_MIN = 1e-9  # minste tillatte vannhøyde (positivitetsklipp)


# ---------- Hjelpefunksjoner for flux ----------
@njit
def _flux_F(h, u, v):
    """x-flux F(Q) = [hu, hu^2 + 1/2 g h^2, huv]^T."""
    hu = h * u
    hv = h * v
    F0 = hu
    F1 = hu * u + 0.5 * g * h * h
    F2 = hu * v
    # shape: (3, Nx, Ny)
    out = np.empty((3, h.shape[0], h.shape[1]), dtype=h.dtype)
    out[0] = F0
    out[1] = F1
    out[2] = F2
    return out

@njit
def _flux_G(h, u, v):
    """y-flux G(Q) = [hv, huv, hv^2 + 1/2 g h^2]^T."""
    hu = h * u
    hv = h * v
    G0 = hv
    G1 = hu * v
    G2 = hv * v + 0.5 * g * h * h
    out = np.empty((3, h.shape[0], h.shape[1]), dtype=h.dtype)
    out[0] = G0
    out[1] = G1
    out[2] = G2
    return out


# ---------- Randeffekter via "shift" ----------
def _shift(arr: np.ndarray, di: int, dj: int, bc: str) -> np.ndarray:
    """
    Hent nabofelt ved å "skifte" med håndtering av randen.
    arr: (..., Nx, Ny)
    di: +1 betyr i+1 (høyre), -1 betyr i-1 (venstre)
    dj: +1 betyr j+1 (opp), -1 betyr j-1 (ned)
    """
    if bc == "periodic":
        return np.roll(np.roll(arr, di, axis=-2), dj, axis=-1)

    out = np.empty_like(arr)
    out[:] = arr

    # start med en enkel kopiering (skift uten wrap)
    if di == 1:
        out[..., :-1, :] = arr[..., 1:, :]
        out[..., -1, :] = arr[..., -1, :]  # midlertidig
    elif di == -1:
        out[..., 1:, :] = arr[..., :-1, :]
        out[..., 0, :] = arr[..., 0, :]

    if dj == 1:
        tmp = out.copy()
        out[..., :, :-1] = tmp[..., :, 1:]
        out[..., :, -1] = tmp[..., :, -1]
    elif dj == -1:
        tmp = out.copy()
        out[..., :, 1:] = tmp[..., :, :-1]
        out[..., :, 0] = tmp[..., :, 0]

    if bc == "neumann":
        # Null normalgradient ≈ kopier nærmeste indre kant (gjort over)
        return out

    if bc == "dirichlet":
        # Hold kantverdier faste (det vi allerede har i arr på kanten)
        if di == 1:
            out[..., -1, :] = arr[..., -1, :]
        if di == -1:
            out[..., 0, :] = arr[..., 0, :]
        if dj == 1:
            out[..., :, -1] = arr[..., :, -1]
        if dj == -1:
            out[..., :, 0] = arr[..., :, 0]
        return out

    raise ValueError(f"Ukjent bc: {bc}")


# ---------- Tidssteg (CFL) ----------
def cfl_dt(h: np.ndarray, u: np.ndarray, v: np.ndarray, dx: float, dy: float, cfl: float = 0.45) -> float:
    wavespeed_x = np.max(np.abs(u) + np.sqrt(g * np.maximum(h, 0.0)))
    wavespeed_y = np.max(np.abs(v) + np.sqrt(g * np.maximum(h, 0.0)))
    # Beskytt mot deling på 0 hvis feltet er helt tomt (teoretisk)
    wavespeed_x = max(wavespeed_x, EPS)
    wavespeed_y = max(wavespeed_y, EPS)
    return cfl * min(dx / wavespeed_x, dy / wavespeed_y)


# ---------- Ett Lax–Friedrichs-FTCS-steg ----------
def step_lax(q: np.ndarray, dx: float, dy: float, dt: float, bc: str) -> np.ndarray:
    """
    q shape: (3, Nx, Ny) med rekkefølge [h, hu, hv].
    """
    h = q[0]
    u = np.divide(q[1], h, out=np.zeros_like(h), where=h > EPS)
    v = np.divide(q[2], h, out=np.zeros_like(h), where=h > EPS)

    # Flux i senter
    F = _flux_F(h, u, v)
    G = _flux_G(h, u, v)

    # Naboceller
    Qip = _shift(q, +1,  0, bc)  # i+1, j
    Qim = _shift(q, -1,  0, bc)  # i-1, j
    Qjp = _shift(q,  0, +1, bc)  # i, j+1
    Qjm = _shift(q,  0, -1, bc)  # i, j-1

    # Flux i naboer (beregnes fra deres h,u,v)
    hip = Qip[0]; uip = np.divide(Qip[1], hip, out=np.zeros_like(hip), where=hip > EPS); vip = np.divide(Qip[2], hip, out=np.zeros_like(hip), where=hip > EPS)
    him = Qim[0]; uim = np.divide(Qim[1], him, out=np.zeros_like(him), where=him > EPS); vim = np.divide(Qim[2], him, out=np.zeros_like(him), where=him > EPS)
    hjp = Qjp[0]; ujp = np.divide(Qjp[1], hjp, out=np.zeros_like(hjp), where=hjp > EPS); vjp = np.divide(Qjp[2], hjp, out=np.zeros_like(hjp), where=hjp > EPS)
    hjm = Qjm[0]; ujm = np.divide(Qjm[1], hjm, out=np.zeros_like(hjm), where=hjm > EPS); vjm = np.divide(Qjm[2], hjm, out=np.zeros_like(hjm), where=hjm > EPS)

    Fip = _flux_F(hip, uip, vip)
    Fim = _flux_F(him, uim, vim)
    Gjp = _flux_G(hjp, ujp, vjp)
    Gjm = _flux_G(hjm, ujm, vjm)

    # Lax–Friedrichs-oppdatering (oppgavens form)
    Qnew = 0.25 * (Qip + Qim + Qjp + Qjm) \
           - (dt / (2 * dx)) * (Fip - Fim) \
           - (dt / (2 * dy)) * (Gjp - Gjm)

    # Positivitetsklipp på h
    Qnew[0] = np.maximum(Qnew[0], H_MIN)
    return Qnew


# ---------- Initialbetingelser ----------
def init_break(Lx=100.0, Ly=100.0, Nx=201, Ny=201, h_inside=10.0, h_outside=1.0, R=10.0):
    """
    "Dam break": sirkel i sentrum med høyere vannhøyde.
    """
    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    cx, cy = Lx / 2, Ly / 2
    h = np.where((X - cx) ** 2 + (Y - cy) ** 2 <= R ** 2, h_inside, h_outside)
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    return x, y, h, u, v


def init_large(Lx=1_000_000.0, Ly=1_000_000.0, Nx=401, Ny=401, H=100.0):
    """
    Stor skala: bakgrunn H og to Gauss-topper (oppgavens 2e).
    """
    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    eta  = np.exp(-((X - Lx/2) ** 2) / (2 * (0.05 * Lx) ** 2) - ((Y - Ly/4) ** 2) / (2 * (0.05 * Ly) ** 2))
    eta += np.exp(-((X - Lx/4) ** 2) / (2 * (0.05 * Lx) ** 2) - ((Y - 3*Ly/4) ** 2) / (2 * (0.05 * Ly) ** 2))
    h = H + eta
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    return x, y, h, u, v


# ---------- Simulering ----------
def simulate_swe(h0: np.ndarray, u0: np.ndarray, v0: np.ndarray,
                 Lx: float, Ly: float, Nt: int,
                 bc: str = "periodic", cfl: float = 0.45,
                 save_every: int = 10) -> tuple[list[np.ndarray], list[float]]:
    """
    Kjør simulering og returner lagrede h-felt og tidspunkter.
    """
    assert bc in ("periodic", "neumann", "dirichlet")
    Nx, Ny = h0.shape
    dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)

    q = np.zeros((3, Nx, Ny), dtype=float)
    q[0] = np.maximum(h0, H_MIN)
    q[1] = h0 * u0
    q[2] = h0 * v0

    frames = [q[0].copy()]
    times = [0.0]
    t = 0.0

    for n in range(1, Nt + 1):
        h = q[0]
        u = np.divide(q[1], h, out=np.zeros_like(h), where=h > EPS)
        v = np.divide(q[2], h, out=np.zeros_like(h), where=h > EPS)

        dt = cfl_dt(h, u, v, dx, dy, cfl=cfl)
        q = step_lax(q, dx, dy, dt, bc=bc)
        t += dt

        if n % save_every == 0:
            frames.append(q[0].copy())
            times.append(t)

    return frames, times


# ---------- Plotting ----------
def plot_field(x, y, h, title="", fname=None):
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(x, y, h.T, shading="auto")
    plt.colorbar(label="h(x,y) [m]")
    plt.xlabel("x"); plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    if fname:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        plt.savefig(fname, dpi=150)
    plt.close()


# ---------- Hovedprogram ----------
def main():
    parser = argparse.ArgumentParser(description="2D shallow water – Lax–Friedrichs FTCS")
    parser.add_argument("--nx", type=int, default=201, help="rutenett i x")
    parser.add_argument("--ny", type=int, default=201, help="rutenett i y")
    parser.add_argument("--lx", type=float, default=100.0, help="domene-lengde x")
    parser.add_argument("--ly", type=float, default=100.0, help="domene-lengde y")
    parser.add_argument("--nt", type=int, default=600, help="antall tidsiterasjoner (lagres hver --save-every)")
    parser.add_argument("--bc", type=str, default="periodic", choices=["periodic", "neumann", "dirichlet"], help="randbetingelse")
    parser.add_argument("--problem", type=str, default="dam", choices=["dam", "large"], help="initialbetingelse")
    parser.add_argument("--save-every", type=int, default=10, help="lagre hvert n-te steg")
    parser.add_argument("--outdir", type=str, default="rapport/figs", help="mappe for figurer")
    args = parser.parse_args()

    if args.problem == "dam":
        x, y, h0, u0, v0 = init_break(Lx=args.lx, Ly=args.ly, Nx=args.nx, Ny=args.ny,
                                      h_inside=10.0, h_outside=1.0, R=0.1*min(args.lx, args.ly))
        title0 = "Dam-break initial h"
    else:
        x, y, h0, u0, v0 = init_large(Lx=args.lx*10_000, Ly=args.ly*10_000, Nx=args.nx, Ny=args.ny, H=100.0)
        title0 = "Large-scale initial h"

    plot_field(x, y, h0, title0, fname=os.path.join(args.outdir, "shallow_init.png"))

    frames, times = simulate_swe(h0, u0, v0, Lx=x[-1]-x[0], Ly=y[-1]-y[0],
                                 Nt=args.nt, bc=args.bc, save_every=args.save_every)

    # Lagre første, midt og siste ramme
    mid_idx = len(frames)//2
    plot_field(x, y, frames[0],      f"h, t={times[0]:.2f}s  ({args.bc})",  fname=os.path.join(args.outdir, f"shallow_{args.bc}_t0.png"))
    plot_field(x, y, frames[mid_idx],f"h, t={times[mid_idx]:.2f}s ({args.bc})", fname=os.path.join(args.outdir, f"shallow_{args.bc}_tmid.png"))
    plot_field(x, y, frames[-1],     f"h, t={times[-1]:.2f}s ({args.bc})", fname=os.path.join(args.outdir, f"shallow_{args.bc}_tend.png"))

    print(f"[OK] Lagret figurer i {args.outdir}. NumPy={np.__version__}, Matplotlib={plt.matplotlib.__version__}, Numba={'on' if JIT else 'off'}")


if __name__ == "__main__":
    main()
