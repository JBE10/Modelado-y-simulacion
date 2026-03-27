"""
Diferenciación Numérica — Diferencias Finitas Centradas
========================================================
Tema 03A/03B — Modelado y Simulación

Fórmulas:
    1ra derivada centrada:  f'(x_i)  ≈ [f(x_{i+1}) - f(x_{i-1})] / (2h)
    2da derivada centrada:  f''(x_i) ≈ [f(x_{i+1}) - 2f(x_i) + f(x_{i-1})] / h²

    Progresiva (extremo izquierdo):
        f'(x_i)  ≈ [f(x_{i+1}) - f(x_i)] / h
        f''(x_i) ≈ [f(x_{i+2}) - 2f(x_{i+1}) + f(x_i)] / h²

    Regresiva (extremo derecho):
        f'(x_i)  ≈ [f(x_i) - f(x_{i-1})] / h
        f''(x_i) ≈ [f(x_i) - 2f(x_{i-1}) + f(x_{i-2})] / h²
"""

from typing import Optional


# ───────────────────────────────────────────────────────────────
# Core: derivadas sobre una lista de puntos (x_i, y_i)
# ───────────────────────────────────────────────────────────────

def diferencias_finitas_tabla(xs: list[float], ys: list[float]) -> dict:
    """
    Calcula la primera y segunda derivada numérica para cada punto de
    una tabla discreta usando el esquema apropiado:
      - interior  → centrada  (O(h²))
      - extremo izq → progresiva (O(h))
      - extremo der → regresiva  (O(h))

    Parámetros
    ----------
    xs : lista de valores x (equidistantes o no)
    ys : lista de valores y = f(x)

    Retorna
    -------
    dict con lista de filas, una por punto.
    """
    n = len(xs)
    if n < 2:
        raise ValueError("Se necesitan al menos 2 puntos.")
    if len(ys) != n:
        raise ValueError("xs e ys deben tener la misma longitud.")

    filas = []
    for i in range(n):
        fila: dict = {"i": i, "x_i": xs[i], "y_i": ys[i]}

        # ── Primera derivada ──────────────────────────────────
        if i == 0:
            # Progresiva
            h = xs[1] - xs[0]
            d1 = (ys[1] - ys[0]) / h
            tipo_d1 = "Progresiva"
        elif i == n - 1:
            # Regresiva
            h = xs[-1] - xs[-2]
            d1 = (ys[-1] - ys[-2]) / h
            tipo_d1 = "Regresiva"
        else:
            # Centrada ⭐
            h = xs[i + 1] - xs[i - 1]  # 2h si equidistante
            d1 = (ys[i + 1] - ys[i - 1]) / h
            tipo_d1 = "Centrada"

        fila["f'(x_i)"] = d1
        fila["tipo_d1"] = tipo_d1

        # ── Segunda derivada ─────────────────────────────────
        if i == 0:
            if n >= 3:
                h = xs[1] - xs[0]
                d2 = (ys[2] - 2 * ys[1] + ys[0]) / (h ** 2)
                tipo_d2 = "Progresiva"
            else:
                d2 = None
                tipo_d2 = "N/A"
        elif i == n - 1:
            if n >= 3:
                h = xs[-1] - xs[-2]
                d2 = (ys[-1] - 2 * ys[-2] + ys[-3]) / (h ** 2)
                tipo_d2 = "Regresiva"
            else:
                d2 = None
                tipo_d2 = "N/A"
        else:
            # Centrada ⭐
            h_fwd = xs[i + 1] - xs[i]
            h_bwd = xs[i] - xs[i - 1]
            h_avg = (h_fwd + h_bwd) / 2.0
            d2 = (ys[i + 1] - 2 * ys[i] + ys[i - 1]) / (h_avg ** 2)
            tipo_d2 = "Centrada"

        fila["f''(x_i)"] = d2
        fila["tipo_d2"] = tipo_d2
        filas.append(fila)

    return {"filas": filas, "n_puntos": n}


# ───────────────────────────────────────────────────────────────
# Core: derivadas sobre una función continua con paso h
# ───────────────────────────────────────────────────────────────

def diferencias_finitas_funcion(
    f,
    x0: float,
    h: float,
    orden: int = 1,
    exacta_d1: Optional[float] = None,
    exacta_d2: Optional[float] = None,
) -> dict:
    """
    Calcula derivadas de una función f en x0 usando diferencias centradas.

    Parámetros
    ----------
    f         : función callable f(x)
    x0        : punto donde derivar
    h         : paso
    orden     : 1 → solo 1ra derivada, 2 → solo 2da, 3 → ambas
    exacta_d1 : valor exacto de f'(x0) para calcular error (opcional)
    exacta_d2 : valor exacto de f''(x0) para calcular error (opcional)

    Fórmulas centradas usadas
    ─────────────────────────
    f'(x)  ≈ [f(x+h) - f(x-h)] / (2h)
    f''(x) ≈ [f(x+h) - 2·f(x) + f(x-h)] / h²
    """
    if h <= 0:
        raise ValueError("El paso h debe ser positivo.")

    f_xph  = f(x0 + h)
    f_x    = f(x0)
    f_xmh  = f(x0 - h)

    resultado = {
        "x0": x0,
        "h": h,
        "f(x0-h)": f_xmh,
        "f(x0)":   f_x,
        "f(x0+h)": f_xph,
        "formula_d1": "[ f(x+h) - f(x-h) ] / (2h)",
        "formula_d2": "[ f(x+h) - 2·f(x) + f(x-h) ] / h²",
    }

    # Primera derivada centrada
    d1 = (f_xph - f_xmh) / (2.0 * h)
    resultado["d1"] = d1
    if exacta_d1 is not None:
        resultado["error_d1"] = abs(d1 - exacta_d1)
        resultado["exacta_d1"] = exacta_d1

    # Segunda derivada centrada
    d2 = (f_xph - 2.0 * f_x + f_xmh) / (h ** 2)
    resultado["d2"] = d2
    if exacta_d2 is not None:
        resultado["error_d2"] = abs(d2 - exacta_d2)
        resultado["exacta_d2"] = exacta_d2

    return resultado
