"""
steffensen.py
=============
Método de Steffensen-Aitken (aceleración Δ²) como iterador primario.

En cada paso se realizan DOS evaluaciones de g para producir x̂_n:

    x'  = g(x_n)
    x'' = g(x')
    Δ   = x'' - 2x' + x_n
    x̂_n = x_n - (x' - x_n)² / Δ       ← siguiente iterado

Esto garantiza convergencia superlineal (≈ cuadrática) incluso cuando
la convergencia de Punto Fijo ordinario es lenta o marginal (|g'| ≈ 1).
"""

import math
from typing import Callable, Any, Optional

from utils import _res


def steffensen(
    g: Callable[[float], float],
    x0: float,
    tol: float = 1e-7,
    max_iter: int = 50,
    f: Optional[Callable[[float], float]] = None,
) -> dict[str, Any]:
    """
    Método de Steffensen-Aitken.

    Parameters
    ----------
    g        : función de iteración  x = g(x)
    x0       : semilla inicial
    tol      : tolerancia de parada  |x̂ - x_n| < tol
    max_iter : máximo de pasos Steffensen (cada uno usa 2 llamadas a g)
    f        : f(x) original (para reportar |f(raíz)|), opcional

    Returns
    -------
    dict estándar con claves: raiz, iteraciones, historial, convergio, justificacion
    Historial incluye: iter, x_n, g(x_n), g(g(x_n)), x_hat, Delta, K (ganancia), error
    """
    historial = []
    xn = float(x0)

    for i in range(1, max_iter + 1):
        # ── Dos evaluaciones de g ────────────────────────────────────────────
        try:
            x1 = float(g(xn))
        except Exception as e:
            return _res(xn, i - 1, historial, False,
                        f"Error al evaluar g(x_n) en iter {i}: {e}")

        try:
            x2 = float(g(x1))
        except Exception as e:
            return _res(x1, i - 1, historial, False,
                        f"Error al evaluar g(g(x_n)) en iter {i}: {e}")

        if not math.isfinite(x1) or not math.isfinite(x2):
            return _res(xn, i - 1, historial, False,
                        f"g produjo valor no finito en iter {i}.")

        # ── Fórmula Δ² ───────────────────────────────────────────────────────
        delta2 = x2 - 2.0 * x1 + xn          # denominador

        if abs(delta2) < 1e-14:
            # Δ² ≈ 0 → el método ya converge, tomamos x2 directamente
            x_hat = x2
        else:
            x_hat = xn - (x1 - xn) ** 2 / delta2

        if not math.isfinite(x_hat):
            return _res(xn, i - 1, historial, False,
                        f"x̂ no es finito en iter {i} (Δ²={delta2:.3e}).")

        error = abs(x_hat - xn)

        # Ganancia efectiva de Kalman (análogo): cuánto "corrijo"
        K = 1.0 - (x_hat - x2) / (xn - x2) if abs(xn - x2) > 1e-14 else float("nan")

        fila: dict[str, Any] = {
            "iter":      i,
            "x_n":       xn,
            "g(x_n)":    x1,
            "g(g(x_n))": x2,
            "Δ²":        delta2,
            "x_hat":     x_hat,
            "error":     error,
        }
        if f is not None:
            try:
                fila["|f(x_hat)|"] = abs(float(f(x_hat)))
            except Exception:
                fila["|f(x_hat)|"] = None

        historial.append(fila)

        if error < tol:
            return _res(x_hat, i, historial, True,
                        f"Convergió en {i} iteraciones de Steffensen-Aitken "
                        f"(|x̂ - x_n| = {error:.3e} < {tol:.3e}).")

        xn = x_hat   # ← x̂ es el NUEVO iterado (aquí está la diferencia)

    return _res(xn, max_iter, historial, False,
                f"No convergió en {max_iter} pasos de Steffensen-Aitken.")
