"""
Diferenciación Numérica — Diferencias Finitas Centradas
========================================================
Tema 03A/03B — Modelado y Simulación

Incluye interpolación de Newton: diferencias progresivas (h constante) y
diferencias divididas (nodos arbitrarios), con LaTeX del polinomio en potencias de x.

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

import math
from fractions import Fraction
from typing import Optional


# ───────────────────────────────────────────────────────────────
# Interpolación Newton — diferencias finitas progresivas (h cte.)
# ───────────────────────────────────────────────────────────────

def _poly_trim(p: list[float], eps: float = 1e-12) -> list[float]:
    out = list(p)
    while len(out) > 1 and abs(out[-1]) < eps:
        out.pop()
    return out


def _poly_add(a: list[float], b: list[float]) -> list[float]:
    n = max(len(a), len(b))
    out = []
    for i in range(n):
        va = a[i] if i < len(a) else 0.0
        vb = b[i] if i < len(b) else 0.0
        out.append(va + vb)
    return _poly_trim(out)


def _poly_scale(p: list[float], s: float) -> list[float]:
    return _poly_trim([c * s for c in p])


def _poly_mul_x_minus_r(p: list[float], r: float) -> list[float]:
    """Multiplica p(x) por (x - r); coeficientes en base 1, x, x², …"""
    out = [0.0] * (len(p) + 1)
    for i, c in enumerate(p):
        out[i + 1] += c
        out[i] -= c * r
    return _poly_trim(out)


def _float_a_latex_coef(c: float, eps: float = 1e-10) -> str:
    """Un coeficiente positivo (valor absoluto) o con signo para término independiente."""
    if abs(c) < eps:
        return "0"
    fr = Fraction(c).limit_denominator(1_000_000)
    if abs(float(fr) - c) < 1e-8:
        num, den = fr.numerator, fr.denominator
        if den == 1:
            return str(num)
        if num < 0:
            return f"-\\frac{{{abs(num)}}}{{{den}}}"
        return f"\\frac{{{num}}}{{{den}}}"
    s = f"{c:.12g}"
    return s.replace("-", "\\text{-}") if s.startswith("-") else s


def polinomio_potencias_a_latex(coefs: list[float], var: str = "x", eps: float = 1e-9) -> str:
    """
    coefs[i] multiplica var^i. Devuelve fragmento LaTeX (sin delimitadores $$).
    Orden decreciente de grado, como en álgebra.
    """
    terms: list[str] = []
    for i in range(len(coefs) - 1, -1, -1):
        c = coefs[i]
        if abs(c) < eps:
            continue
        if i == 0:
            if not terms:
                terms.append(_float_a_latex_coef(c, eps))
            elif c >= 0:
                terms.append(" + " + _float_a_latex_coef(c, eps))
            else:
                terms.append(" - " + _float_a_latex_coef(abs(c), eps))
            continue
        mag = abs(c)
        coef_tex = _float_a_latex_coef(mag, eps)
        if i == 1:
            if abs(mag - 1.0) < eps:
                xp = var
            else:
                xp = f"{coef_tex}\\,{var}"
        else:
            if abs(mag - 1.0) < eps:
                xp = f"{var}^{{{i}}}"
            else:
                xp = f"{coef_tex}\\,{var}^{{{i}}}"
        if c < 0:
            terms.append(" - " + xp if terms else "-" + xp)
        else:
            terms.append(" + " + xp if terms else xp)
    if not terms:
        return "0"
    return "".join(terms)

def polinomio_potencias_a_python(coefs: list[float], var: str = "x", eps: float = 1e-9) -> str:
    """Devuelve la expresión del polinomio en formato de Python para usar con pow/lambdify."""
    terms: list[str] = []
    for i in range(len(coefs) - 1, -1, -1):
        c = coefs[i]
        if abs(c) < eps:
            continue
        c_str = f"{c:.15g}" # plain float formatting
        if i == 0:
            terms.append(f"{c_str}")
        elif i == 1:
            terms.append(f"{c_str}*{var}")
        else:
            terms.append(f"{c_str}*{var}**{i}")
    if not terms:
        return "0.0"
    return " + ".join(terms).replace("+ -", "- ")



def interpolacion_newton_diferencias(
    xs: list[float],
    ys: list[float],
    tol_h: float = 1e-7,
) -> dict:
    """
    Polinomio interpolante por Newton (diferencias progresivas).
    Requiere abscisas equiespaciadas: x_i = x_0 + i h.

    Retorna
    -------
    dict con x0, h, triangulo (filas), deltas (Δ^k y_0), coefs (base 1,x,x²,…),
    latex_p (expandido), latex_newton (forma en s=(x-x0)/h).
    """
    n = len(xs)
    if n != len(ys):
        raise ValueError("xs e ys deben tener la misma longitud.")
    if n < 1:
        raise ValueError("Se necesita al menos un punto.")

    x0 = xs[0]
    if n == 1:
        h = 1.0
        diff = [[float(ys[0])]]
    else:
        h = xs[1] - xs[0]
        if abs(h) < 1e-15:
            raise ValueError("El paso h entre nodos no puede ser cero.")
        for i in range(n - 1):
            hi = xs[i + 1] - xs[i]
            if abs(hi - h) > tol_h * max(1.0, abs(h)):
                raise ValueError(
                    "Los nodos x deben ser equiespaciados (mismo h en todo el intervalo)."
                )
        diff = []
        row = [float(v) for v in ys]
        diff.append(row)
        for _ in range(n - 1):
            prev = diff[-1]
            row = [prev[j + 1] - prev[j] for j in range(len(prev) - 1)]
            diff.append(row)

    deltas = [diff[k][0] for k in range(len(diff))]

    total = [0.0]
    prod = [1.0]
    for k in range(len(deltas)):
        d0 = deltas[k]
        coef = d0 / math.factorial(k) / (h ** k)
        if k == 0:
            term = [coef]
        else:
            prod = _poly_mul_x_minus_r(prod, x0 + (k - 1) * h)
            term = _poly_scale(prod, coef)
        total = _poly_add(total, term)

    coefs = total
    latex_expandido = polinomio_potencias_a_latex(coefs)
    python_poly = polinomio_potencias_a_python(coefs)

    # Forma Newton: y_0 + (Δy_0/1!) s + (Δ²y_0/2!) s(s-1) + … ,  s = (x-x_0)/h
    newton_terms: list[str] = []
    primero = True
    for k in range(len(deltas)):
        d0 = deltas[k]
        if abs(d0) < 1e-14 and k > 0:
            continue
        raw = d0 / math.factorial(k)
        if k == 0:
            newton_terms.append(_float_a_latex_coef(raw, 1e-10))
            primero = False
            continue
        mag = abs(raw)
        if k == 1:
            inner = "s" if abs(mag - 1.0) < 1e-9 else f"{_float_a_latex_coef(mag, 1e-10)}\\,s"
        else:
            sf = "s" + "".join(f"(s-{j})" for j in range(1, k))
            inner = sf if abs(mag - 1.0) < 1e-9 else f"{_float_a_latex_coef(mag, 1e-10)}\\,{sf}"
        if raw < 0:
            newton_terms.append((" - " if not primero else "-") + inner)
        else:
            newton_terms.append((" + " if not primero else "") + inner)
        primero = False
    latex_newton = "".join(newton_terms)

    # Tabla legible para DataFrame: columnas x, y, Δ^1, Δ^2, ...
    filas_tri = []
    for i in range(n):
        fila: dict = {"x": xs[i], "y": ys[i]}
        for k in range(1, n):
            if i < n - k:
                fila[f"Δ^{k}"] = diff[k][i]
            else:
                fila[f"Δ^{k}"] = None
        filas_tri.append(fila)

    return {
        "x0": x0,
        "h": h,
        "triangulo_diff": diff,
        "filas_tabla": filas_tri,
        "deltas_y0": deltas,
        "coefs_potencias": coefs,
        "latex_polinomio": latex_expandido,
        "python_polinomio": python_poly,
        "latex_newton_s": latex_newton,
        "grado": len(coefs) - 1,
        "modo": "progresivas",
    }


def interpolacion_newton_divididas(xs: list[float], ys: list[float]) -> dict:
    """
    Polinomio interpolante por Newton con diferencias divididas (nodos x distintos,
    espaciado arbitrario). Grado ≤ n con n+1 puntos.

    P(x) = f[x₀] + f[x₀,x₁](x-x₀) + f[x₀,x₁,x₂](x-x₀)(x-x₁) + …
    """
    n1 = len(xs)
    if n1 != len(ys):
        raise ValueError("xs e ys deben tener la misma longitud.")
    if n1 < 1:
        raise ValueError("Se necesita al menos un punto.")
    xs_f = [float(x) for x in xs]
    ys_f = [float(y) for y in ys]
    for i in range(n1 - 1):
        if abs(xs_f[i + 1] - xs_f[i]) < 1e-14:
            raise ValueError("Hay abscisas repetidas o demasiado cercanas; se requieren x distintos.")

    n = n1 - 1
    DD: list[list[float]] = [list(ys_f)]
    for k in range(1, n1):
        prev = DD[k - 1]
        row = []
        for j in range(n1 - k):
            num = prev[j + 1] - prev[j]
            den = xs_f[j + k] - xs_f[j]
            if abs(den) < 1e-15:
                raise ValueError("Abscisas mal condicionadas (denominador cero en divididas).")
            row.append(num / den)
        DD.append(row)

    aks = [DD[k][0] for k in range(n1)]

    total = [0.0]
    prod = [1.0]
    for k in range(n1):
        ak = aks[k]
        if k == 0:
            term = [ak]
        else:
            prod = _poly_mul_x_minus_r(prod, xs_f[k - 1])
            term = _poly_scale(prod, ak)
        total = _poly_add(total, term)

    coefs = total
    latex_expandido = polinomio_potencias_a_latex(coefs)
    python_poly = polinomio_potencias_a_python(coefs)

    # Forma Newton en factores (x-x_i) con coeficientes numéricos
    def x_shift_latex(xi: float) -> str:
        if abs(xi) < 1e-10:
            return "x"
        tx = _float_a_latex_coef(abs(xi), 1e-9)
        if xi > 0:
            return f"\\bigl(x - {tx}\\bigr)"
        return f"\\bigl(x + {tx}\\bigr)"

    newton_terms: list[str] = []
    primero = True
    for k in range(n1):
        ak = aks[k]
        if abs(ak) < 1e-14 and k > 0:
            continue
        if k == 0:
            newton_terms.append(_float_a_latex_coef(ak, 1e-10))
            primero = False
            continue
        factor = "".join(x_shift_latex(xs_f[j]) for j in range(k))
        mag = abs(ak)
        if abs(mag - 1.0) < 1e-9:
            piece = factor
        else:
            piece = f"{_float_a_latex_coef(mag, 1e-10)}\\,{factor}"
        if ak < 0:
            newton_terms.append((" - " if not primero else "-") + piece)
        else:
            newton_terms.append((" + " if not primero else "") + piece)
        primero = False
    latex_newton_factores = "".join(newton_terms)

    filas_tri = []
    for i in range(n1):
        fila: dict = {"x": xs_f[i], "y": ys_f[i]}
        for k in range(1, n1):
            key = f"div_{k}"
            if i < n1 - k:
                fila[key] = DD[k][i]
            else:
                fila[key] = None
        filas_tri.append(fila)

    # ¿equiespaciados? (solo informativo)
    h0 = xs_f[1] - xs_f[0] if n1 > 1 else 1.0
    equi = n1 <= 1 or all(
        abs((xs_f[i + 1] - xs_f[i]) - h0) < 1e-7 * max(1.0, abs(h0)) for i in range(n1 - 1)
    )

    return {
        "x0": xs_f[0],
        "h": xs_f[1] - xs_f[0] if n1 > 1 else 1.0,
        "triangulo_diff": DD,
        "filas_tabla": filas_tri,
        "deltas_y0": aks,
        "coefs_potencias": coefs,
        "latex_polinomio": latex_expandido,
        "python_polinomio": python_poly,
        "latex_newton_divididas": latex_newton_factores,
        "latex_newton_s": "",
        "grado": len(coefs) - 1,
        "modo": "divididas",
        "nodos_equiespaciados": equi,
    }


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
            d1_fwd = (ys[i + 1] - ys[i]) / h_fwd
            d1_bwd = (ys[i] - ys[i - 1]) / h_bwd
            d2 = 2.0 * (d1_fwd - d1_bwd) / (h_fwd + h_bwd)
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
