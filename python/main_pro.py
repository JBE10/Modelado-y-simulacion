import streamlit as st
from ui_modules.base import ModuleFactory
from ui_modules.rocket_module import Rocket3DModule
# from ui_modules.roots_module import RootsModule  <-- Other modules would be imported here

# --- Configuration ---
st.set_page_config(page_title="Numerical Lab Pro", layout="wide")

def main():
    st.title("🧪 Numerical Lab: Modular Edition")
    
    # 1. Initialize Factory (SOLID: Open/Closed Principle)
    factory = ModuleFactory()
    
    # 2. Register Modules (Strategy Pattern)
    # Adding a new algorithm is as simple as registering a new class
    factory.register_module(Rocket3DModule())
    # factory.register_module(RootsModule())
    # factory.register_module(EpidemiologyModule())

    # 3. Sidebar Selection
    with st.sidebar:
        st.header("⚙️ Configuración")
        options = factory.get_all_names()
        selected_name = st.selectbox("Seleccione Algoritmo", options)
        st.divider()
        st.info("Esta versión utiliza una arquitectura SOLID desacoplada.")

    # 4. Execution (Liskov Substitution Principle)
    # All modules follow the DashboardModule interface
    module = factory.get_module(selected_name)
    if module:
        module.render()
    else:
        st.error("Módulo no encontrado.")

if __name__ == "__main__":
    main()
