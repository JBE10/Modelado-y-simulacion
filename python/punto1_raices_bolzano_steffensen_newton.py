"""
Punto 1 — f(x) = 2x cos(x) - (x-2)^2
- Bolzano en [0.8, 1.4]
- Punto fijo g(x) = x - f(x) + aceleración Steffensen–Aitken (6 decimales)
- Newton–Raphson con |x_{n+1}-x_n| < 1e-8
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Callable, List

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None


def f(x: float) -> float:
    return 2.0 * x * math.cos(x) - (x - 2.0) ** 2


def df(x: float) -> float:
    return 2.0 * math.cos(x) - 2.0 * x * math.sin(x) - 2.0 * (x - 2.0)


def g_punto_fijo(x: float) -> float:
    """Punto fijo equivalente a f(x)=0: g(x)=x-f(x). En la raíz, g(ξ)=ξ."""
    return x - f(x)


def dg_punto_fijo(x: float) -> float:
    return 1.0 - df(x)


def bolzano(a: float, b: float) -> None:
    fa, fb = f(a), f(b)
    print("--- Bolzano ---")
    print(f"f({a}) = {fa:.10f}")
    print(f"f({b}) = {fb:.10f}")
    print(f"Producto f(a)·f(b) = {fa * fb:.6e} (< 0 ⇒ hay raíz en (a,b))")
    print()


@dataclass
class SteffRow:
    k: int
    x0: float
    x1: float
    x2: float
    x_acc: float
    err: float


def steffensen_aitken(
    g: Callable[[float], float],
    x0: float,
    decimales: int = 6,
    max_iter: int = 50,
) -> tuple[float, List[SteffRow]]:
    """
    Aceleración Δ² de Aitken sobre dos pasos de punto fijo:
    x0, x1=g(x0), x2=g(x1)  →  x̂ = x0 - (x1-x0)²/(x2 - 2x1 + x0)
    """
    tol = 0.5 * 10 ** (-decimales)
    rows: List[SteffRow] = []
    x = float(x0)
    for k in range(1, max_iter + 1):
        y0 = x
        y1 = g(y0)
        y2 = g(y1)
        denom = y2 - 2.0 * y1 + y0
        if abs(denom) < 1e-16:
            raise RuntimeError(f"Denominador Aitken ~ 0 en k={k}")
        x_acc = y0 - (y1 - y0) ** 2 / denom
        err = abs(x_acc - y0)
        rows.append(SteffRow(k=k, x0=y0, x1=y1, x2=y2, x_acc=x_acc, err=err))
        if err < tol:
            return x_acc, rows
        x = x_acc
    raise RuntimeError("Steffensen–Aitken no convergió")


def newton_tabla(x0: float, tol: float = 1e-8, max_iter: int = 50) -> List[dict]:
    historial: List[dict] = []
    xn = float(x0)
    fx = f(xn)
    for i in range(1, max_iter + 1):
        dfx = df(xn)
        if abs(dfx) <= sys.float_info.epsilon:
            raise RuntimeError(f"f'(x)=0 en iter {i}")
        x_next = xn - fx / dfx
        err = abs(x_next - xn)
        historial.append(
            {
                "n": i,
                "x_n": xn,
                "f(x_n)": fx,
                "f'(x_n)": dfx,
                "x_{n+1}": x_next,
                "|Δx|": err,
            }
        )
        fx_next = f(x_next)
        if err < tol:
            return historial
        xn, fx = x_next, fx_next
    raise RuntimeError("Newton no convergió")


def lipschitz_g_en_intervalo(a: float, b: float, puntos: int = 2000) -> tuple[float, float, float]:
    """Cota L = max|g'| en [a,b] (g C¹ ⇒ Lipschitz con constante L)."""
    xs = [a + (b - a) * i / (puntos - 1) for i in range(puntos)]
    derivs = [abs(dg_punto_fijo(t)) for t in xs]
    t_max = xs[int(derivs.index(max(derivs)))]
    return max(derivs), min(derivs), t_max


def main() -> None:
    a, b = 0.8, 1.4
    x0 = 1.0

    bolzano(a, b)

    L_max, L_min, t_max = lipschitz_g_en_intervalo(a, b)
    print("--- Lipschitz de g(x)=x-f(x) en [0.8, 1.4] ---")
    print("g ∈ C¹ ⇒ |g(x)-g(y)| ≤ L|x-y| con L = max|g'| en el compacto.")
    print(f"min|g'| ≈ {L_min:.6f}  max|g'| = L ≈ {L_max:.6f} (cerca de x ≈ {t_max:.4f})")
    print()

    raiz_s, filas_s = steffensen_aitken(g_punto_fijo, x0, decimales=6)
    print("--- Steffensen–Aitken (punto fijo acelerado), x0 = 1, 6 decimales ---")
    print("k   x_k      g(x_k)    g(g(x_k))   x̂ (Aitken)   |x̂-x_k|")
    for r in filas_s:
        print(
            f"{r.k:2d}  {r.x0:.9f}  {r.x1:.9f}  {r.x2:.9f}  {r.x_acc:.9f}  {r.err:.3e}"
        )
    print(f"\nRaíz (Steffensen–Aitken): {raiz_s:.10f}  f(r)={f(raiz_s):.3e}\n")

    filas_n = newton_tabla(x0, tol=1e-8)
    print("--- Newton–Raphson, x0=1, parada si |x_{n+1}-x_n| < 1e-8 ---")
    print("n   x_n           f(x_n)        f'(x_n)       x_{n+1}       |Δx|")
    for row in filas_n:
        dfxn = row["f'(x_n)"]
        print(
            f"{row['n']:2d}  {row['x_n']:.12f}  {row['f(x_n)']:.6e}  "
            f"{dfxn:.6e}  {row['x_{n+1}']:.12f}  {row['|Δx|']:.3e}"
        )
    raiz_n = filas_n[-1]["x_{n+1}"]
    print(f"\nRaíz (Newton): {raiz_n:.12f}  f(r)={f(raiz_n):.3e}\n")

    print("--- Comparación breve (para el informe) ---")
    print(f"Diferencia |r_Newton - r_Steffensen| = {abs(raiz_n - raiz_s):.3e}")
    print("Newton: suele ser más rápido (cuadrático cerca de raíz simple), requiere f'.")
    print("Steffensen–Aitken: acelera punto fijo sin derivada en la iteración; más pasos por ciclo.")

    # Gráfico (opcional si matplotlib está instalado)
    if plt is not None:
        xs = [a + (b - a) * i / 499 for i in range(500)]
        ys = [f(t) for t in xs]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(xs, ys, label=r"$f(x)=2x\cos x-(x-2)^2$")
        ax.axhline(0, color="k", linewidth=0.8)
        ax.axvline(raiz_n, color="C1", linestyle="--", label=f"Raíz ≈ {raiz_n:.8f}")
        ax.scatter([raiz_n], [0], color="C1", zorder=5)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title("Punto 1 — raíz en [0.8, 1.4]")
        ax.legend()
        ax.grid(True, alpha=0.3)
        out = __file__.replace(".py", "_grafico.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        print(f"Gráfico guardado: {out}")
    else:
        print("(matplotlib no instalado: omito PNG; pip install matplotlib para el gráfico)")


if __name__ == "__main__":
    main()
