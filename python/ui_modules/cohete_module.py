"""cohete_module.py — Lanzamiento Cohete 3D UI Module (full dashboard logic)."""
import numpy as np
import pandas as pd
import streamlit as st
from ui_modules.base import DashboardModule
from ui_utils import _formulas_panel
from rocket_engine_solid import Rocket, RocketStage, EarthRocketPhysics, RK4Integrator, SimulationEngine

PRESETS = {
    "Falcon 9 (SpaceX)": {
        "stages": [
            {"name": "Etapa 1", "T": 7607000, "m_p": 395700, "m_d": 25600, "md": 2500},
            {"name": "Etapa 2", "T": 934000, "m_p": 92670, "m_d": 3900, "md": 260},
        ],
        "pay": 5000, "bal": 600,
    },
    "Apolo 11 (Misión Lunar)": {
        "stages": [
            {"name": "S-IC (Etapa 1)", "T": 34000000, "m_p": 2100000, "m_d": 130000, "md": 13000},
            {"name": "S-II (Etapa 2)", "T": 4400000, "m_p": 450000, "m_d": 36000, "md": 1100},
            {"name": "S-IVB (Etapa 3 / TLI)", "T": 1000000, "m_p": 110000, "m_d": 10000, "md": 230},
        ],
        "pay": 45000, "bal": 300000,
    },
    "Artemis 2 (SLS/Orion)": {
        "stages": [
            {"name": "Boosters + Core (Etapa 1)", "T": 39100000, "m_p": 1600000, "m_d": 293000, "md": 12000},
            {"name": "Core Stage (Etapa 2)", "T": 9000000, "m_p": 600000, "m_d": 98000, "md": 2000},
            {"name": "ICPS (Etapa 3 / TLI)", "T": 110100, "m_p": 26850, "m_d": 3800, "md": 24},
        ],
        "pay": 26520, "bal": 300000,
    },
    "Personalizado": None,
}


class CoheteModule(DashboardModule):
    @property
    def name(self) -> str:
        return "Lanzamiento Cohete 3D"

    def render(self):
        main_col, side_col = st.columns([2.3, 1.0], gap="large")
        with main_col:
            st.markdown("### 🚀 Mission Control (Versión PRO)")
            preset_choice = st.selectbox("Seleccionar Plantilla de Cohete", list(PRESETS.keys()))
            config = PRESETS[preset_choice]

            with st.form("rocket_form"):
                if config:
                    st.info(f"Configuración cargada: {preset_choice} ({len(config['stages'])} etapas)")
                    for s in config["stages"]:
                        with st.expander(f"Resumen {s['name']}"):
                            st.write(f"Empuje: {s['T']:,} N | Propelente: {s['m_p']:,} kg | Burn Rate: {s['md']} kg/s")

                c1, c2, c3 = st.columns(3)
                if not config:
                    with c1:
                        st.markdown("**Etapa 1 (Booster)**")
                        T1 = st.number_input("Empuje (N)", value=7600000, step=100000)
                        m1_prop = st.number_input("Masa Propelente (kg)", value=395700)
                        m1_dry = st.number_input("Masa Seca (kg)", value=25600)
                        m_dot1 = st.number_input("Tasa Consumo (kg/s)", value=2500)
                    with c2:
                        st.markdown("**Etapa 2 (Orbital)**")
                        T2 = st.number_input("Empuje (N) ", value=934000, step=10000)
                        m2_prop = st.number_input("Masa Propelente (kg) ", value=92670)
                        m2_dry = st.number_input("Masa Seca (kg) ", value=3900)
                        m_dot2 = st.number_input("Tasa Consumo (kg/s) ", value=260)
                    with c3:
                        st.markdown("**Carga Útil & Sim**")
                        m_payload = st.number_input("Masa Carga Útil (kg)", value=5000)
                        t_bal = st.number_input("Vuelo Balístico (s)", value=600)
                        h_step = st.slider("Paso h (s)", 0.1, 2.0, 0.5)
                else:
                    m_payload = config["pay"]
                    t_bal = config["bal"]
                    h_step = st.slider("Paso h (s)", 0.1, 2.0, 0.5)

                st.divider()
                st.markdown("**Visual y Cámaras**")
                cv1, cv2, cv3 = st.columns(3)
                num_satellites = cv1.slider("Satélites", 0, 100, 30)
                show_moon = cv2.checkbox("Incluir la Luna", value=True)
                camera_mode = cv3.radio("Modo de Cámara", ["Libre", "Cabina", "Persecución"])
                c_row2 = st.columns(2)
                playback_speed = c_row2[0].slider("Velocidad de Animación", 0.1, 5.0, 1.0, step=0.1)
                show_orbits = c_row2[1].checkbox("Mostrar Órbitas", value=True)
                run_rocket = st.form_submit_button("🚀 INICIAR LANZAMIENTO", type="primary")

            if run_rocket:
                rocket = Rocket(payload_mass=m_payload)
                if config:
                    for s in config["stages"]:
                        rocket.add_stage(RocketStage(s["name"], s["T"], s["m_p"], s["m_d"], s["md"]))
                else:
                    rocket.add_stage(RocketStage("Etapa 1", T1, m1_prop, m1_dry, m_dot1))
                    rocket.add_stage(RocketStage("Etapa 2", T2, m2_prop, m2_dry, m_dot2))

                physics = EarthRocketPhysics()
                integrator = RK4Integrator()
                engine = SimulationEngine(rocket, physics, integrator)

                try:
                    res = engine.run(h_step=h_step, t_ballistic=t_bal)
                    z_vals = res["z"]; v_vals = np.sqrt(res["vx"]**2 + res["vz"]**2)
                    z_clean = z_vals[np.isfinite(z_vals)]; v_clean = v_vals[np.isfinite(v_vals)]
                    max_alt = np.max(z_clean) / 1000 if len(z_clean) > 0 else 0
                    max_vel = np.max(v_clean) if len(v_clean) > 0 else 0

                    with st.container(border=True):
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Apogeo", f"{max_alt:,.1f} km")
                        mc2.metric("Velocidad Máxima", f"{max_vel:,.0f} m/s", f"Mach {max_vel/343:.1f}" if max_vel > 0 else "0")
                        mc3.metric("Masa Final", f"{res['m'][-1]:.0f} kg")

                    # Three.js 3D visualization
                    Re_m = 6371000.0
                    phi = res["x"] / Re_m
                    X_3d = (Re_m + res["z"]) * np.sin(phi)
                    Y_3d = np.zeros_like(X_3d)
                    Z_3d = (Re_m + res["z"]) * np.cos(phi)
                    n_total = len(X_3d)
                    max_points = 2000
                    skip_rate = max(1, n_total // max_points)
                    trajectory_data = []
                    for i in range(n_total):
                        is_critical = res["t"][i] < 60
                        if is_critical or (i % skip_rate == 0) or (i == n_total - 1):
                            trajectory_data.append({
                                "x": float(X_3d[i]), "y": float(Y_3d[i]), "z": float(Z_3d[i]),
                                "f": res["etapa"][i], "v": float(np.sqrt(res["vx"][i]**2 + res["vz"][i]**2)),
                                "t": float(res["t"][i])
                            })

                    three_js_code = f"""
<!DOCTYPE html><html><head><style>
body {{ margin: 0; background-color: #000; overflow: hidden; }}
#info {{ position: absolute; top: 10px; left: 10px; color: white;
font-family: 'Segoe UI', sans-serif; pointer-events: none;
text-shadow: 1px 1px 2px black; font-size: 13px; }}
</style></head><body>
<div id="info"><b>MISSION CONTROL</b><br>
Altitud: <span id="alt">0</span> km | Vel: <span id="vel">0</span> m/s<br>
Fase: <span id="stage">-</span></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const trajectory = {trajectory_data};
const numSats = {num_satellites};
const showMoon = {str(show_moon).lower()};
const showOrbits = {str(show_orbits).lower()};
const cameraMode = "{camera_mode}";
const playbackSpeed = {playback_speed};
let scene, camera, renderer, controls, earth, rocket, moon, rocketTrail, exhaust;
let rocketParts = [], currentStageName = "", debrisList = [], satellites = [], animIndex = 0;
const RE = 6371;
function init() {{
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 10000000);
    renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    scene.add(new THREE.AmbientLight(0xffffff, 0.2));
    const sun = new THREE.DirectionalLight(0xffffff, 1.8);
    sun.position.set(RE*20, RE*10, RE*20); scene.add(sun);
    const loader = new THREE.TextureLoader();
    earth = new THREE.Mesh(new THREE.SphereGeometry(RE, 64, 64),
        new THREE.MeshPhongMaterial({{ color: 0x224488, shininess: 10 }}));
    loader.load('https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_atmos_2048.jpg', (tex) => {{ earth.material.map = tex; earth.material.needsUpdate = true; }});
    scene.add(earth);
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(RE*1.015, 64, 64),
        new THREE.MeshBasicMaterial({{ color: 0x00aaff, transparent: true, opacity: 0.1, side: THREE.BackSide }})));
    if (showMoon) {{
        moon = new THREE.Mesh(new THREE.SphereGeometry(1737, 32, 32), new THREE.MeshPhongMaterial({{ color: 0xaaaaaa }}));
        loader.load('https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/moon_1024.jpg', (tex) => {{ moon.material.map = tex; moon.material.needsUpdate = true; }});
        moon.position.set(0, 0, 384400); scene.add(moon);
    }}
    const starsPos = [];
    for(let i=0; i<25000; i++) starsPos.push((Math.random()-0.5)*RE*600, (Math.random()-0.5)*RE*600, (Math.random()-0.5)*RE*600);
    const starsGeom = new THREE.BufferGeometry();
    starsGeom.setAttribute('position', new THREE.Float32BufferAttribute(starsPos, 3));
    scene.add(new THREE.Points(starsGeom, new THREE.PointsMaterial({{ color: 0xffffff, size: 2 }})));
    rocket = new THREE.Group();
    const stagesFound = [...new Set(trajectory.map(p => p.f))];
    stagesFound.forEach((name, i) => {{
        const part = new THREE.Group(); part.userData.name = name;
        if (name.includes("Carga") || name.includes("Orion")) {{
            const nose = new THREE.Mesh(new THREE.ConeGeometry(0.3, 1.2, 16), new THREE.MeshStandardMaterial({{ color: 0xffffff, metalness: 0.4 }}));
            nose.position.y = 2.0; part.add(nose);
        }} else {{
            const h = 2.0;
            const body = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, h, 16), new THREE.MeshStandardMaterial({{ color: 0xcccccc, metalness: 0.6 }}));
            body.position.y = (stagesFound.length - 2 - i) * h; part.add(body);
            if (i === 0) {{
                const eng = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.4, 0.5, 12), new THREE.MeshStandardMaterial({{ color: 0x111111 }}));
                eng.position.y = body.position.y - 1.25; part.add(eng);
            }}
        }}
        rocketParts.push(part); rocket.add(part);
    }});
    exhaust = new THREE.Mesh(new THREE.ConeGeometry(0.35, 3.0, 12),
        new THREE.MeshBasicMaterial({{ color: 0xffaa00, transparent: true, opacity: 0.8 }}));
    exhaust.position.y = -4.5; exhaust.rotation.x = Math.PI; rocket.add(exhaust);
    rocket.rotateX(Math.PI/2); scene.add(rocket);
    rocketTrail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({{ color: 0xffaa00, linewidth: 2 }}));
    scene.add(rocketTrail);
    if (cameraMode === "Libre") {{ camera.position.set(RE*0.1, RE*0.1, RE*1.05); camera.lookAt(0, 0, RE); }}
    for(let i=0; i<numSats; i++) {{
        const s = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.1, 0.3), new THREE.MeshStandardMaterial({{ color: 0xcccccc }}));
        const r = RE + 300 + Math.random()*2500, sp = 0.0003 + Math.random()*0.0008;
        const ang = Math.random()*Math.PI*2, inc = (Math.random()-0.5)*Math.PI*0.7;
        satellites.push({{ m: s, r: r, sp: sp, ang: ang, inc: inc }}); scene.add(s);
        if(showOrbits) {{
            const o = new THREE.Mesh(new THREE.TorusGeometry(r, 0.05, 2, 100), new THREE.MeshBasicMaterial({{ color: 0xffffff, transparent: true, opacity: 0.1 }}));
            o.rotation.x = Math.PI/2; o.rotation.y = inc; scene.add(o);
        }}
    }}
    animate();
}}
function separatePart(part, forward) {{
    if (!part || !part.parent) return;
    const worldPos = new THREE.Vector3(), worldQuat = new THREE.Quaternion();
    part.getWorldPosition(worldPos); part.getWorldQuaternion(worldQuat);
    rocket.remove(part); scene.add(part);
    part.position.copy(worldPos); part.quaternion.copy(worldQuat);
    debrisList.push({{ mesh: part, vel: forward.clone().multiplyScalar(-0.2).add(new THREE.Vector3((Math.random()-0.5)*0.05, -0.1, 0)) }});
}}
function animate() {{
    requestAnimationFrame(animate);
    const idx = Math.floor(animIndex);
    if (idx < trajectory.length) {{
        const p = trajectory[idx];
        const x = p.x/1000, y = p.y/1000, z = p.z/1000;
        rocket.position.set(x, y, z);
        let forward = new THREE.Vector3();
        if (idx > 0) {{ const prev = trajectory[idx-1]; forward.set(x-prev.x/1000, y-prev.y/1000, z-prev.z/1000).normalize(); }}
        else forward.set(x, y, z).normalize();
        rocket.lookAt(new THREE.Vector3(x, y, z).add(forward));
        if (moon) {{
            const moonAngle = (p.t / 2358720) * Math.PI * 2;
            moon.position.x = 384400 * Math.sin(moonAngle);
            moon.position.z = 384400 * Math.cos(moonAngle);
        }}
        exhaust.visible = !(p.f.includes("Carga") || p.f.includes("Orion"));
        if (exhaust.visible) exhaust.scale.set(1, 0.8 + Math.random()*0.4, 1);
        if (p.f !== currentStageName) {{
            rocketParts.forEach(part => {{
                if (part.parent === rocket && part.userData.name !== p.f && currentStageName !== "") {{
                    if (trajectory.findIndex(tp => tp.f === part.userData.name) < trajectory.findIndex(tp => tp.f === p.f))
                        separatePart(part, forward);
                }}
            }});
            currentStageName = p.f;
        }}
        const points = [];
        const start = Math.max(0, idx - 1000);
        for(let i=start; i<=idx; i++) points.push(new THREE.Vector3(trajectory[i].x/1000, trajectory[i].y/1000, trajectory[i].z/1000));
        rocketTrail.geometry.setFromPoints(points);
        document.getElementById('alt').innerText = (Math.sqrt(x*x+y*y+z*z)-RE).toFixed(1);
        document.getElementById('vel').innerText = Math.round(p.v);
        document.getElementById('stage').innerText = p.f;
        if (cameraMode === "Cabina") {{
            controls.enabled = false;
            camera.position.copy(rocket.position).addScaledVector(forward, 2.5);
            camera.lookAt(rocket.position.clone().addScaledVector(forward, 10));
        }} else if (cameraMode === "Persecución") {{
            controls.enabled = false;
            const camOffset = forward.clone().multiplyScalar(-25).add(new THREE.Vector3(0, 8, 0));
            camera.position.lerp(rocket.position.clone().add(camOffset), 0.1);
            camera.lookAt(rocket.position);
        }} else {{ controls.enabled = true; controls.update(); }}
        animIndex += playbackSpeed;
    }} else {{ controls.enabled = true; controls.update(); }}
    debrisList.forEach(d => {{ d.mesh.position.add(d.vel); d.mesh.rotation.x += 0.01; d.mesh.rotation.z += 0.005; }});
    if (moon) moon.rotation.y += 0.001;
    satellites.forEach(s => {{ s.ang += s.sp; s.m.position.set(s.r*Math.cos(s.ang), s.r*Math.sin(s.ang)*Math.sin(s.inc), s.r*Math.sin(s.ang)*Math.cos(s.inc)); }});
    earth.rotation.y += 0.0005;
    renderer.render(scene, camera);
}}
init();
</script></body></html>"""
                    st.components.v1.html(three_js_code, height=600)

                    with st.container(border=True):
                        st.markdown("### 🗺️ Mapa Orbital 2D (Cenital)")
                        tierra_2d = pd.DataFrame({"x": [0], "z": [0], "label": ["Tierra"]})
                        t_final = res["t"][-1]
                        moon_angle = (t_final / 2358720) * np.pi * 2
                        luna_2d = pd.DataFrame({"x": [384400 * np.sin(moon_angle)], "z": [384400 * np.cos(moon_angle)], "label": ["Luna"]})
                        phi_map = (res["x"] / 1000) / 6371.0
                        r_map = 6371 + (res["z"] / 1000)
                        import altair as alt
                        traj_real_2d = pd.DataFrame({"x": r_map * np.sin(phi_map), "z": r_map * np.cos(phi_map), "Fase": res["etapa"]})
                        base = alt.Chart(traj_real_2d).mark_line().encode(
                            x=alt.X("x:Q", title="X (km)", scale=alt.Scale(domain=[-450000, 450000])),
                            y=alt.Y("z:Q", title="Z (km)", scale=alt.Scale(domain=[-450000, 450000])),
                            color="Fase:N")
                        c_tierra = alt.Chart(tierra_2d).mark_point(size=200, color="blue", filled=True).encode(x="x", y="z")
                        c_luna = alt.Chart(luna_2d).mark_point(size=100, color="gray", filled=True).encode(x="x", y="z")
                        st.altair_chart((base + c_tierra + c_luna).properties(height=600), use_container_width=True)

                    with st.container(border=True):
                        st.markdown("### 📉 Evolución de Variables")
                        df_res = pd.DataFrame({
                            "Tiempo (s)": res["t"], "Altitud (m)": res["z"],
                            "Dist. Horizontal (m)": res["x"], "Vel. Vertical (m/s)": res["vz"],
                            "Vel. Horizontal (m/s)": res["vx"], "Masa (kg)": res["m"], "Fase": res["etapa"],
                        })
                        st.dataframe(df_res.iloc[::max(1, len(df_res) // 100)], hide_index=True, use_container_width=True)

                except Exception as e:
                    st.error(f"Error en la simulación: {e}")

        with side_col:
            _formulas_panel("Lanzamiento Cohete 3D")
            with st.container(border=True):
                st.markdown("### 💡 Fases de Vuelo")
                st.caption("**1. Ignición:** Todo el peso. Fuerte empuje.")
                st.caption("**2. Gravity Turn:** El cohete comienza a inclinarse para ganar velocidad horizontal.")
                st.caption("**3. Separación (MECO):** Se apaga la etapa 1 y se descarta su peso estructural.")
                st.caption("**4. Inserción:** La etapa 2 lleva el payload a velocidad orbital.")
