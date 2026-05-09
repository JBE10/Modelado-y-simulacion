"""
Modelo epidemiológico SIR resuelto con Runge-Kutta.
  S(t) = Susceptibles
  I(t) = Infectados
  R(t) = Recuperados

Sistema de EDOs:
  dS/dt = -β·S·I / N
  dI/dt =  β·S·I / N  - γ·I
  dR/dt =  γ·I

Parámetros:
  β  = tasa de contagio
  γ  = tasa de recuperación  (1/γ = días promedio enfermo)
  R₀ = β/γ  (número reproductivo básico)
  N  = población total (S + I + R = N, constante)
"""


def _sir_derivs(t, state, beta, gamma, N):
    """Calcula las derivadas del sistema SIR."""
    S, I, R = state
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def sir_euler(beta, gamma, N, I0, R0_init, t_max, h):
    """
    Resuelve el modelo SIR con el método de Euler.
    """
    S0 = N - I0 - R0_init
    t_vals = [0.0]
    S_vals = [S0]
    I_vals = [I0]
    R_vals = [R0_init]

    t = 0.0
    S, I, R = S0, I0, R0_init

    while t < t_max - 1e-9:
        step = min(h, t_max - t)
        derivs = _sir_derivs(t, [S, I, R], beta, gamma, N)
        S = S + step * derivs[0]
        I = I + step * derivs[1]
        R = R + step * derivs[2]
        t = t + step

        # Clamp para evitar valores negativos por errores numéricos
        S = max(S, 0.0)
        I = max(I, 0.0)
        R = max(R, 0.0)

        t_vals.append(t)
        S_vals.append(S)
        I_vals.append(I)
        R_vals.append(R)

    return {
        "metodo": "Euler",
        "t": t_vals,
        "S": S_vals,
        "I": I_vals,
        "R": R_vals,
    }


def sir_rk4(beta, gamma, N, I0, R0_init, t_max, h):
    """
    Resuelve el modelo SIR con Runge-Kutta de orden 4 (RK4 clásico).
    """
    S0 = N - I0 - R0_init
    t_vals = [0.0]
    S_vals = [S0]
    I_vals = [I0]
    R_vals = [R0_init]

    t = 0.0
    S, I, R = S0, I0, R0_init

    while t < t_max - 1e-9:
        step = min(h, t_max - t)
        state = [S, I, R]

        k1 = _sir_derivs(t, state, beta, gamma, N)
        k2 = _sir_derivs(
            t + 0.5 * step,
            [state[i] + 0.5 * step * k1[i] for i in range(3)],
            beta, gamma, N,
        )
        k3 = _sir_derivs(
            t + 0.5 * step,
            [state[i] + 0.5 * step * k2[i] for i in range(3)],
            beta, gamma, N,
        )
        k4 = _sir_derivs(
            t + step,
            [state[i] + step * k3[i] for i in range(3)],
            beta, gamma, N,
        )

        S = S + (step / 6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        I = I + (step / 6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        R = R + (step / 6.0) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        t = t + step

        S = max(S, 0.0)
        I = max(I, 0.0)
        R = max(R, 0.0)

        t_vals.append(t)
        S_vals.append(S)
        I_vals.append(I)
        R_vals.append(R)

    return {
        "metodo": "RK4",
        "t": t_vals,
        "S": S_vals,
        "I": I_vals,
        "R": R_vals,
    }


def calcular_metricas(resultado, beta, gamma, N):
    """Calcula métricas clave del resultado SIR."""
    I_vals = resultado["I"]
    t_vals = resultado["t"]
    S_vals = resultado["S"]
    R_vals = resultado["R"]

    pico_I = max(I_vals)
    idx_pico = I_vals.index(pico_I)
    dia_pico = t_vals[idx_pico]

    R0 = beta / gamma if gamma > 0 else float("inf")

    # Porcentaje total que se infectó (= R final / N)
    total_infectados_pct = (R_vals[-1] / N) * 100

    # Inmunidad de rebaño: 1 - 1/R₀
    if R0 > 1:
        umbral_rebano = (1 - 1 / R0) * 100
    else:
        umbral_rebano = 0.0

    return {
        "R0": R0,
        "pico_infectados": pico_I,
        "dia_pico": dia_pico,
        "pico_pct": (pico_I / N) * 100,
        "total_infectados_pct": total_infectados_pct,
        "umbral_rebano_pct": umbral_rebano,
    }
