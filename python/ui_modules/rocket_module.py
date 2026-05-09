import streamlit as st
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
from ui_modules.base import DashboardModule
from rocket_engine_solid import Rocket, RocketStage, EarthRocketPhysics, RK4Integrator, SimulationEngine

class Rocket3DModule(DashboardModule):
    @property
    def name(self):
        return "Lanzamiento Cohete Pro (SOLID)"

    def render(self):
        st.header("🚀 Mission Control (SOLID Engine)")
        st.caption("Arquitectura modular con Inyección de Dependencias y Patrón Strategy.")

        main_col, side_col = st.columns([2.3, 1.0], gap="large")
        
        with main_col:
            with st.form("rocket_solid_form"):
                st.markdown("### Configuración del Sistema")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Etapa 1**")
                    t1 = st.number_input("Empuje (N)", value=7600000, step=100000)
                    m1p = st.number_input("Masa Propelente (kg)", value=395700)
                    m1d = st.number_input("Masa Seca (kg)", value=25600)
                    md1 = st.number_input("Tasa Consumo (kg/s)", value=2500)
                with c2:
                    st.markdown("**Etapa 2**")
                    t2 = st.number_input("Empuje (N) ", value=934000, step=10000)
                    m2p = st.number_input("Masa Propelente (kg) ", value=92670)
                    m2d = st.number_input("Masa Seca (kg) ", value=3900)
                    md2 = st.number_input("Tasa Consumo (kg/s) ", value=260)
                with c3:
                    st.markdown("**Simulación**")
                    mpl = st.number_input("Carga Útil (kg)", value=5000)
                    tb = st.number_input("Vuelo Balístico (s)", value=600)
                    h = st.slider("Paso h (s)", 0.1, 2.0, 0.5)
                
                submitted = st.form_submit_button("Lanzar", type="primary")

            if submitted:
                # Dependency Injection in Action
                rocket = Rocket(payload_mass=mpl)
                rocket.add_stage(RocketStage("Etapa 1", t1, m1p, m1d, md1))
                rocket.add_stage(RocketStage("Etapa 2", t2, m2p, m2d, md2))
                
                physics = EarthRocketPhysics()
                integrator = RK4Integrator()
                engine = SimulationEngine(rocket, physics, integrator)
                
                res = engine.run(h_step=h, t_ballistic=tb)
                self._render_results(res)

    def _render_results(self, res):
        # Render metrics, 3D (Three.js) and tables...
        # (Reusing the Three.js logic from before but encapsulated here)
        m1, m2 = st.columns(2)
        m1.metric("Altitud Máxima", f"{np.max(res['z'])/1000:.1f} km")
        m2.metric("Velocidad Máxima", f"{np.max(np.sqrt(res['vx']**2 + res['vz']**2)):.0f} m/s")
        
        # ... (Rest of Three.js logic would go here)
        st.success("Simulación completada con éxito usando el motor SOLID.")
