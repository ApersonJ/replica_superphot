import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import time
import zlib
from datetime import datetime, timedelta
from pathlib import Path

import dynesty
import numpy as np
import pandas as pd
from dynesty import utils as dyfunc
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from scipy.special import ndtr, ndtri
from scipy.stats import truncnorm

PROC = Path("data/processed")
SALIDA = Path("data/nested_resultados.jsonl")
N_DRAWS = 100          # muestras posteriores que se guardan por banda
SEED = 0


PRIORS = np.array([
    [0.096, 0.058, -0.3, 0.5],      # log10(A / Fmax)
    [8.3e-3, 3.9e-3, 0.0, 0.03],    # beta
    [1.43, 0.31, 0.0, 3.5],         # log10(gamma)
    [-17.9, 9.9, -100.0, 30.0],     # t0 (dias, relativo al tiempo de referencia)
    [0.67, 0.43, -2.0, 4.0],        # log10(tau_rise)
    [1.53, 0.30, 0.0, 4.0],         # log10(tau_fall)
    [-1.66, 0.34, -3.0, -0.8],      # log10(sigma_extra / Fmax)
])
MU, SD, LO, HI = PRIORS.T
CDF_LO = ndtr((LO - MU) / SD)
CDF_HI = ndtr((HI - MU) / SD)


def _z(u):
    """Inversa de la CDF de una gaussiana truncada (equivale a truncnorm.ppf, pero vectorizada)."""
    p = CDF_LO + np.asarray(u) * (CDF_HI - CDF_LO)
    return MU + SD * ndtri(np.clip(p, 1e-12, 1 - 1e-12))


def prior_transform(u, fmax):
    z = _z(u)
    return np.array([fmax * 10 ** z[0], z[1], 10 ** z[2], z[3],
                     10 ** z[4], 10 ** z[5], fmax * 10 ** z[6]])


def modelo_banda(t, A, beta, gamma, t0, tau_rise, tau_fall):
    """Ecuacion 1 del paper, para una banda."""
    dt = t - t0
    subida = A / (1 + np.exp(-dt / tau_rise))
    plateau = 1 - beta * dt
    caida = (1 - beta * gamma) * np.exp((gamma - dt) / tau_fall)
    return subida * np.where(dt < gamma, plateau, caida)


def loglike(theta, t, f, e):
    A, beta, gamma, t0, tau_rise, tau_fall, sig = theta
    beta = max(beta, 1e-9)
    with np.errstate(all="ignore"):
        if np.exp(-gamma / tau_rise) * (1 / beta - tau_rise - gamma) > tau_rise:   # ec. B2
            return -1e30
        if gamma > (1 - beta * tau_fall) / beta:                                   # ec. B3
            return -1e30
        var = e ** 2 + sig ** 2
        modelo = modelo_banda(t, A, beta, gamma, t0, tau_rise, tau_fall)
        ll = -0.5 * np.sum((f - modelo) ** 2 / var + np.log(2 * np.pi * var))
    return ll if np.isfinite(ll) else -1e30


def ajusta_curve_fit(c, t_ref):
    """Ajuste de minimos cuadrados (el de la reproduccion), para comparar."""
    t = c["mjd"].to_numpy() - t_ref
    f = c["flujo"].to_numpy()
    e = c["flujo_err"].to_numpy()
    fm = f.max()
    p0 = [1.25 * fm, 0.0083, 27, -18, 4.7, 34]
    inf = [0.5 * fm, 1e-4, 1.0, -100, 0.01, 1.0]
    sup = [3.16 * fm, 0.03, 3162, 30, 1e4, 1e4]
    popt, _ = curve_fit(modelo_banda, t, f, p0=p0, sigma=e, bounds=(inf, sup), maxfev=20000)
    return popt


def ajusta_dynesty(c, t_ref, seed):
    t = c["mjd"].to_numpy() - t_ref
    f = c["flujo"].to_numpy()
    e = c["flujo_err"].to_numpy()
    fmax = f.max()
    sampler = dynesty.NestedSampler(
        lambda th: loglike(th, t, f, e),
        lambda u: prior_transform(u, fmax),
        ndim=7, nlive=50, bound="single", sample="rwalk",
        rstate=np.random.default_rng(seed))

    sampler.run_nested(dlogz=0.5, maxiter=5000, maxcall=150000, print_progress=False)
    res = sampler.results
    pesos = np.exp(res.logwt - res.logz[-1])
    return dyfunc.resample_equal(res.samples, pesos), int(np.sum(res.ncall))


def procesa_objeto(nombre):
    try:
        curva = pd.read_parquet(PROC / f"{nombre}.parquet")
        cr = curva[curva["banda"] == "r"]
        t_ref = cr["mjd"].to_numpy()[cr["flujo"].to_numpy().argmax()]   # maximo en r, comun a g y r
        out = {"NAME": nombre}
        for b in ["g", "r"]:
            c = curva[curva["banda"] == b]
            t_ini = time.time()
            seed = SEED + zlib.crc32(f"{nombre}{b}".encode()) % 100000
            muestras, ncall = ajusta_dynesty(c, t_ref, seed)
            q = np.percentile(muestras, [16, 50, 84], axis=0)
            idx = np.random.default_rng(seed).choice(
                len(muestras), min(N_DRAWS, len(muestras)), replace=False)
            try:
                cf = ajusta_curve_fit(c, t_ref).tolist()
            except (RuntimeError, ValueError):
                cf = None
            out[b] = {"n": int(len(c)), "seg": round(time.time() - t_ini, 1), "ncall": ncall,
                      "p16": q[0].tolist(), "med": q[1].tolist(), "p84": q[2].tolist(),
                      "curve_fit": cf, "draws": np.round(muestras[idx], 5).tolist()}
        return out
    except Exception as ex:          
        return {"NAME": nombre, "error": repr(ex)}


def orden_estratificado():
    """Orden por turnos entre clases: cualquier prefijo de la lista esta balanceado por clase."""
    X = pd.read_csv("data/features.csv", usecols=["NAME", "CLASE"])
    rng = np.random.default_rng(SEED)
    grupos = [rng.permutation(g["NAME"].to_numpy()) for _, g in X.groupby("CLASE")]
    orden = []
    for i in range(max(len(g) for g in grupos)):
        for g in grupos:
            if i < len(g):
                orden.append(g[i])
    return orden


def verifica():
    u = np.random.default_rng(1).random((2000, 7))
    lento = np.column_stack([
        truncnorm.ppf(u[:, i], (LO[i] - MU[i]) / SD[i], (HI[i] - MU[i]) / SD[i],
                      loc=MU[i], scale=SD[i]) for i in range(7)])
    print("prior rapido == truncnorm:", np.allclose(_z(u), lento))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hasta", default=None, help="hora HH:MM a la que deja de empezar curvas nuevas")
    ap.add_argument("--n_jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--max", type=int, default=None, help="procesar solo N curvas (prueba)")
    ap.add_argument("--verifica", action="store_true")
    a = ap.parse_args()
    if a.verifica:
        verifica()
        return

    limite = None
    if a.hasta:
        h, m = map(int, a.hasta.split(":"))
        ahora = datetime.now()
        limite = ahora.replace(hour=h, minute=m, second=0, microsecond=0)
        if limite <= ahora:
            limite += timedelta(days=1)
        print("hora limite:", limite.strftime("%d/%m %H:%M"), flush=True)

    hechos = set()
    if SALIDA.exists():
        with open(SALIDA) as fh:
            hechos = {json.loads(linea)["NAME"] for linea in fh if linea.strip()}
    pendientes = [n for n in orden_estratificado() if n not in hechos]
    if a.max:
        pendientes = pendientes[:a.max]
    print(f"{len(hechos)} ya hechas, {len(pendientes)} pendientes, {a.n_jobs} procesos", flush=True)

    lote = a.n_jobs * 2
    t_ini = time.time()
    n_hechas = 0
    for i in range(0, len(pendientes), lote):
        if limite and datetime.now() >= limite:
            print("hora limite alcanzada", flush=True)
            break
        nombres = pendientes[i:i + lote]
        resultados = Parallel(n_jobs=a.n_jobs)(delayed(procesa_objeto)(n) for n in nombres)
        with open(SALIDA, "a") as fh:
            for r in resultados:
                fh.write(json.dumps(r) + "\n")
        n_hechas += len(nombres)
        print(f"{n_hechas} curvas en {round((time.time() - t_ini) / 60, 1)} min", flush=True)
    print("listo", flush=True)


if __name__ == "__main__":
    main()