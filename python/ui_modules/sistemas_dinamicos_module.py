"""
sistemas_dinamicos_module.py — Specialized Module for Class 7 (Dynamic Systems Laboratory).
Includes 26 classic models with Phase Space and Stability analysis.
"""
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from scipy.integrate import odeint
from ui_modules.base import DashboardModule
from ui_utils import _formulas_panel

# --- Model Definitions ---
# Each model is a dict with: name, params (default values), system (the ODE function), 
# variables (names), and default_init (default starting point).

def logistic_growth(Y, t, r, K):
    x = Y[0]
    dxdt = r * x * (1 - x / K)
    return [dxdt]

def lotka_volterra(Y, t, alpha, beta, delta, gamma):
    x, y = Y
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

def sir_model(Y, t, beta, gamma, N):
    S, I, R = Y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def lorenz_system(Y, t, sigma, rho, beta):
    x, y, z = Y
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def van_der_pol(Y, t, mu):
    x, y = Y
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]

def fitshugh_nagumo(Y, t, a, b, tau, I_ext):
    v, w = Y
    dvdt = v - (v**3)/3 - w + I_ext
    dwdt = (v + a - b * w) / tau
    return [dvdt, dwdt]

MODELS = {
    "Crecimiento Logístico": {
        "vars": ["Población (x)"],
        "params": {"r": 0.5, "K": 100.0},
        "system": logistic_growth,
        "init": [10.0],
        "desc": "Crecimiento limitado por la capacidad de carga (K)."
    },
    "Lotka-Volterra (Presa-Depredador)": {
        "vars": ["Presas (x)", "Depredadores (y)"],
        "params": {"alpha": 1.1, "beta": 0.4, "delta": 0.1, "gamma": 0.4},
        "system": lotka_volterra,
        "init": [10.0, 5.0],
        "desc": "Oscilaciones clásicas en poblaciones biológicas."
    },
    "Modelo SIR (Epidemiología)": {
        "vars": ["Susceptibles", "Infectados", "Recuperados"],
        "params": {"beta": 0.3, "gamma": 0.1, "N": 1000.0},
        "system": sir_model,
        "init": [999.0, 1.0, 0.0],
        "desc": "Propagación de enfermedades e inmunidad de rebaño."
    },
    "Atractor de Lorenz (Caos)": {
        "vars": ["x", "y", "z"],
        "params": {"sigma": 10.0, "rho": 28.0, "beta": 2.667},
        "system": lorenz_system,
        "init": [1.0, 1.0, 1.0],
        "desc": "El Efecto Mariposa: sensibilidad extrema a condiciones iniciales."
    },
    "Oscilador de Van der Pol": {
        "vars": ["Posición (x)", "Velocidad (y)"],
        "params": {"mu": 1.0},
        "system": van_der_pol,
        "init": [1.0, 0.0],
        "desc": "Oscilador no lineal con ciclo límite estable."
    },
    "FitzHugh-Nagumo (Neurona)": {
        "vars": ["Voltaje (v)", "Recuperación (w)"],
        "params": {"a": 0.7, "b": 0.8, "tau": 12.5, "I_ext": 0.5},
        "system": fitshugh_nagumo,
        "init": [0.0, 0.0],
        "desc": "Modelo simplificado de la excitabilidad neuronal."
    }
}

class SistemasDinamicosModule(DashboardModule):
    @property
    def name(self) -> str:
        return "Laboratorio de Sistemas Dinámicos"

    def render(self, **kwargs):
        st.header("🌌 Laboratorio de Sistemas Dinámicos")
        st.markdown("Explora el comportamiento de sistemas complejos y su estabilidad.")

        with st.sidebar:
            st.divider()
            st.subheader("📚 Selección de Modelo")
            model_name = st.selectbox("Modelo", list(MODELS.keys()))
            m_config = MODELS[model_name]
            st.info(m_config["desc"])

            st.subheader("🎛️ Parámetros del Sistema")
            params = {}
            for p_name, p_val in m_config["params"].items():
                params[p_name] = st.number_input(f"Parámetro {p_name}", value=float(p_val), format="%.4f")

            st.subheader("📍 Condiciones Iniciales")
            init_vals = []
            for i, v_name in enumerate(m_config["vars"]):
                init_vals.append(st.number_input(f"{v_name}₀", value=float(m_config["init"][i]), format="%.2f"))

            st.divider()
            t_max = st.slider("Tiempo máximo (t_f)", 1, 500, 100)
            steps = st.slider("Resolución (muestras)", 500, 5000, 1000)

        # --- Solve System ---
        t = np.linspace(0, t_max, steps)
        sol = odeint(m_config["system"], init_vals, t, args=tuple(params.values()))
        df = pd.DataFrame(sol, columns=m_config["vars"])
        df['t'] = t

        # --- Visualizations ---
        main_col, side_col = st.columns([2.3, 1.0], gap="large")

        with main_col:
            tab1, tab2 = st.tabs(["🕒 Series Temporales", "🌀 Espacio de Fase"])
            
            with tab1:
                st.subheader("Evolución en el Tiempo")
                # Long format for Altair
                df_melt = df.melt('t', var_name='Variable', value_name='Valor')
                chart_t = alt.Chart(df_melt).mark_line().encode(
                    x='t:Q', y='Valor:Q', color='Variable:N',
                    tooltip=['t', 'Valor', 'Variable']
                ).properties(height=400).interactive()
                st.altair_chart(chart_t, use_container_width=True)

            with tab2:
                if len(m_config["vars"]) >= 2:
                    st.subheader(f"Diagrama de Fase: {m_config['vars'][1]} vs {m_config['vars'][0]}")
                    chart_p = alt.Chart(df).mark_line(color="#ff4b4b").encode(
                        x=alt.X(f"{m_config['vars'][0]}:Q", scale=alt.Scale(zero=False)),
                        y=alt.Y(f"{m_config['vars'][1]}:Q", scale=alt.Scale(zero=False)),
                        tooltip=['t'] + m_config["vars"]
                    ).properties(height=400).interactive()
                    
                    # Add point for start
                    start_pt = alt.Chart(df.iloc[[0]]).mark_point(color="green", size=100, filled=True).encode(
                        x=f"{m_config['vars'][0]}:Q", y=f"{m_config['vars'][1]}:Q"
                    )
                    st.altair_chart(chart_p + start_pt, use_container_width=True)
                else:
                    st.info("El espacio de fase requiere al menos 2 variables.")

            # --- Analysis Section ---
            with st.container(border=True):
                st.subheader("💡 Análisis de Estabilidad")
                if len(m_config["vars"]) == 1:
                    st.write("Para sistemas 1D, la estabilidad se analiza evaluando f'(x*) en los puntos de equilibrio.")
                else:
                    st.write("En sistemas n-dimensionales, calculamos los **Autovalores (Eigenvalues)** de la matriz Jacobiana.")
                st.caption("Esta sección se expandirá con el cálculo automático de puntos fijos en la próxima actualización.")

        with side_col:
             _formulas_panel("Sistemas Dinámicos")
