"""
main_pro.py — New Modular Dashboard Entry Point.

Run with:
    streamlit run python/main_pro.py

Architecture: Strategy + Factory (SOLID)
Each algorithm is encapsulated in its own DashboardModule subclass.
Adding a new algorithm = create a module class + register it here.
"""
import sys
from pathlib import Path

# Ensure the python/ directory is always on sys.path regardless of CWD
_dir = Path(__file__).resolve().parent
if str(_dir) not in sys.path:
    sys.path.insert(0, str(_dir))

import streamlit as st
from ui_modules.base import ModuleFactory

# ── Import all modules ─────────────────────────────────────────────────────────
from ui_modules.biseccion_module          import BiseccionModule
from ui_modules.punto_fijo_module         import PuntoFijoModule
from ui_modules.steffensen_module         import SteffensenModule
from ui_modules.newton_raphson_module     import NewtonRaphsonModule
from ui_modules.secante_module            import SecanteModule
from ui_modules.comparar_raices_module    import CompararRaicesModule
from ui_modules.diferencias_finitas_module   import DiferenciasFinitasModule
from ui_modules.integracion_numerica_module  import IntegracionNumericaModule
from ui_modules.ecuaciones_diferenciales_module import EdoModule
from ui_modules.simulacion_sir_module     import SirModule
from ui_modules.cohete_module             import CoheteModule
from ui_modules.monte_carlo_module        import MonteCarloModule
from ui_modules.filtro_kalman_module      import KalmanModule

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Métodos Numéricos", layout="wide")


def main():
    st.title("Métodos Numéricos")
    st.caption("Funciones: sin, cos, tan, exp, log, ln, sqrt, cbrt, abs — Constantes: pi, e — Potencias: ^ o **")
    st.divider()

    # 1. Initialize Factory (Open/Closed Principle)
    factory = ModuleFactory()

    # 2. Register Modules (Strategy Pattern) — order defines sidebar listing
    factory.register_module(BiseccionModule())
    factory.register_module(PuntoFijoModule())
    factory.register_module(NewtonRaphsonModule())
    factory.register_module(SecanteModule())
    factory.register_module(SteffensenModule())
    factory.register_module(CompararRaicesModule())
    factory.register_module(DiferenciasFinitasModule())
    factory.register_module(IntegracionNumericaModule())
    factory.register_module(EdoModule())
    factory.register_module(SirModule())
    factory.register_module(CoheteModule())
    factory.register_module(MonteCarloModule())
    factory.register_module(KalmanModule())

    # 3. Sidebar
    with st.sidebar:
        st.header("Configuracion")
        options = factory.get_all_names()
        selected_name = st.selectbox("Algoritmo", options)
        st.divider()
        st.markdown("### Vista de f(x)")
        n_samples = st.slider("Muestras", min_value=200, max_value=2000, step=100, value=900)
        expand_factor = st.slider("Factor de expansion del rango", min_value=0.2, max_value=2.0, step=0.1, value=0.8)
        st.caption("Aumenta muestras para curvas mas detalladas.")
        st.divider()
        st.info("Arquitectura SOLID modular — Strategy + Factory Pattern.")

    # Persist plot config so modules can read it via session_state
    st.session_state["plot_cfg"] = {
        "n_samples": int(n_samples),
        "expand_factor": float(expand_factor),
    }

    # 4. Dispatch (Liskov Substitution Principle — all modules share the same interface)
    module = factory.get_module(selected_name)
    if module:
        module.render()
    else:
        st.error("Módulo no encontrado.")


if __name__ == "__main__":
    main()
