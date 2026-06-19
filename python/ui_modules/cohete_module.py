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
            {"name": "S-II (Etapa 2)", "T": 5150000, "m_p": 450000, "m_d": 36000, "md": 1100},
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

    def render(self, **kwargs):
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
                    
                    max_q_val = np.max(res["q"]) / 1000.0
                    max_q_idx = np.argmax(res["q"])
                    max_q_t = res["t"][max_q_idx]

                    with st.container(border=True):
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Apogeo", f"{max_alt:,.1f} km")
                        mc2.metric("Velocidad Máxima", f"{max_vel:,.0f} m/s", f"Mach {max_vel/343:.1f}" if max_vel > 0 else "0")
                        mc3.metric("Max-Q", f"{max_q_val:,.1f} kPa", f"t={max_q_t:.0f}s")
                        mc4.metric("Masa Final", f"{res['m'][-1]:.0f} kg")

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
                                "t": float(res["t"][i]), "g": float(res["g_force"][i]), "q": float(res["q"][i])
                            })

                    three_js_code = f"""
<!DOCTYPE html><html><head><style>
body {{ margin: 0; background-color: #000; overflow: hidden; }}
#info {{ position: absolute; top: 10px; left: 10px; color: white;
font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; pointer-events: none;
text-shadow: 1px 1px 3px black; font-size: 13px; z-index: 10; line-height: 1.5; }}
#controls-panel {{
    position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
    background: rgba(15, 23, 42, 0.75); backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 12px;
    padding: 8px 16px; display: flex; align-items: center; gap: 12px;
    color: white; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    z-index: 100; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
}}
.btn {{
    background: rgba(255, 255, 255, 0.1); border: 1px solid rgba(255, 255, 255, 0.2);
    color: white; padding: 6px 12px; border-radius: 6px; cursor: pointer;
    font-weight: 600; font-size: 12px; transition: all 0.2s; outline: none;
}}
.btn:hover {{ background: rgba(255, 255, 255, 0.2); border-color: rgba(255, 255, 255, 0.4); }}
.btn:active {{ transform: scale(0.95); }}
select {{
    background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(255, 255, 255, 0.2);
    color: white; padding: 6px; border-radius: 6px; font-size: 12px; cursor: pointer; outline: none;
}}
input[type="range"] {{
    -webkit-appearance: none; width: 180px; height: 6px;
    background: rgba(255, 255, 255, 0.2); border-radius: 3px; outline: none;
}}
input[type="range"]::-webkit-slider-thumb {{
    -webkit-appearance: none; appearance: none; width: 14px; height: 14px;
    border-radius: 50%; background: #00aaff; cursor: pointer; box-shadow: 0 0 5px #00aaff;
}}
</style></head><body>
<div id="info">
  <b>MISSION CONTROL</b><br>
  Altitud: <span id="alt">0</span> km<br>
  Velocidad: <span id="vel">0</span> m/s<br>
  Fuerza G: <span id="gforce">1.0</span> G<br>
  Pres. Dinámica: <span id="pres">0.0</span> kPa<br>
  Fase: <span id="stage">-</span>
</div>
<div id="controls-panel">
    <button class="btn" id="playBtn">⏸ Pause</button>
    <button class="btn" id="resetBtn">🔄 Reset</button>
    <input type="range" id="timeline" min="0" value="0">
    <span style="font-size: 11px; white-space: nowrap;">Cámara:</span>
    <select id="camSelect">
        <option value="Libre">Libre</option>
        <option value="Cabina">Cabina</option>
        <option value="Persecución">Persecución</option>
    </select>
    <span style="font-size: 11px; white-space: nowrap;">Vel:</span>
    <select id="speedSelect">
        <option value="0.2">0.2x</option><option value="0.5">0.5x</option>
        <option value="1">1.0x</option><option value="2">2.0x</option>
        <option value="5">5.0x</option><option value="10">10.0x</option>
    </select>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const trajectory = {trajectory_data};
const numSats = {num_satellites};
const showMoon = {str(show_moon).lower()};
const showOrbits = {str(show_orbits).lower()};
let cameraMode = "{camera_mode}";
let playbackSpeed = {playback_speed};
let isPaused = false;
let scene, camera, renderer, controls, earth, rocket, moon, rocketTrail, exhaust;
let rocketParts = [], currentStageName = "", debrisList = [], satellites = [], animIndex = 0, lastIdx = -1;
const RE = 6371;

function init() {{
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 10000000);
    renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, RE, 0);
    
    scene.add(new THREE.AmbientLight(0xffffff, 0.2));
    const sun = new THREE.DirectionalLight(0xffffff, 1.8); sun.position.set(RE*20, RE*10, RE*20); scene.add(sun);
    
    const loader = new THREE.TextureLoader();
    earth = new THREE.Mesh(new THREE.SphereGeometry(RE, 64, 64), new THREE.MeshPhongMaterial({{ color: 0x224488 }}));
    loader.load('https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_atmos_2048.jpg', (tex) => {{
        earth.material.map = tex;
        earth.material.color.set(0xffffff);
        earth.material.needsUpdate = true;
    }});
    scene.add(earth);
    
    const atmosGeom = new THREE.SphereGeometry(RE * 1.015, 64, 64);
    const atmosMat = new THREE.MeshBasicMaterial({{
        color: 0x00aaff,
        transparent: true,
        opacity: 0.12,
        side: THREE.BackSide
    }});
    const atmosphere = new THREE.Mesh(atmosGeom, atmosMat);
    scene.add(atmosphere);

    if (showMoon) {{
        moon = new THREE.Mesh(new THREE.SphereGeometry(1737, 32, 32), new THREE.MeshPhongMaterial({{ color: 0xaaaaaa }}));
        loader.load('https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/moon_1024.jpg', (tex) => {{
            moon.material.map = tex;
            moon.material.needsUpdate = true;
        }});
        moon.position.set(0, 0, 384400); scene.add(moon);
        
        const moonOrbitPts = [];
        for (let i = 0; i <= 128; i++) {{
            const a = (i / 128) * Math.PI * 2;
            moonOrbitPts.push(new THREE.Vector3(384400 * Math.sin(a), 0, 384400 * Math.cos(a)));
        }}
        const moonOrbitGeom = new THREE.BufferGeometry().setFromPoints(moonOrbitPts);
        const moonOrbitLine = new THREE.Line(moonOrbitGeom, new THREE.LineBasicMaterial({{ color: 0x444444, transparent: true, opacity: 0.3 }}));
        scene.add(moonOrbitLine);
    }}
    
    rocket = new THREE.Group();
    const stagesFound = [...new Set(trajectory.map(p => p.f))];
    stagesFound.forEach((name, i) => {{
        const part = new THREE.Group(); part.userData.name = name;
        const h = 2.0; const r_stage = 0.3;
        const body = new THREE.Mesh(new THREE.CylinderGeometry(r_stage, r_stage, h, 16), new THREE.MeshStandardMaterial({{ color: 0xcccccc, metalness: 0.5, roughness: 0.3 }}));
        body.position.y = (stagesFound.length - 2 - i) * h; part.add(body);
        
        if (i === stagesFound.length - 1 || name.includes("Carga") || name.includes("Payload") || name.includes("Orion")) {{
            const cone = new THREE.Mesh(new THREE.ConeGeometry(r_stage, 1.2, 16), new THREE.MeshStandardMaterial({{ color: 0xdd4444, metalness: 0.4, roughness: 0.4 }}));
            cone.position.y = (stagesFound.length - 2 - i) * h + h/2 + 0.6;
            part.add(cone);
        }}
        
        rocketParts.push(part); rocket.add(part);
    }});
    
    exhaust = new THREE.Mesh(new THREE.ConeGeometry(0.35, 3.0, 12), new THREE.MeshBasicMaterial({{ color: 0xffaa00, transparent: true, opacity: 0.8 }}));
    exhaust.position.y = -(stagesFound.length - 1) * 2.0 - 1.5; exhaust.rotation.x = Math.PI; rocket.add(exhaust);
    scene.add(rocket);
    
    rocketTrail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({{ color: 0xffaa00, linewidth: 2 }}));
    scene.add(rocketTrail);
    
    createSatellites();
    
    const timeline = document.getElementById('timeline');
    timeline.max = trajectory.length - 1;
    
    document.getElementById('playBtn').addEventListener('click', () => {{
        isPaused = !isPaused;
        document.getElementById('playBtn').innerText = isPaused ? "▶ Play" : "⏸ Pause";
    }});
    
    document.getElementById('resetBtn').addEventListener('click', () => {{
        animIndex = 0;
        resetStaging();
        updateSimulationToFrame(0);
    }});
    
    timeline.addEventListener('input', (e) => {{
        const val = parseInt(e.target.value);
        if (val < animIndex) {{
            resetStaging();
        }}
        animIndex = val;
        updateSimulationToFrame(val);
    }});
    
    document.getElementById('camSelect').addEventListener('change', (e) => {{
        cameraMode = e.target.value;
    }});
    
    document.getElementById('speedSelect').addEventListener('change', (e) => {{
        playbackSpeed = parseFloat(e.target.value);
    }});
    
    window.addEventListener('resize', onWindowResize);
    
    animate();
}}

function createSatellites() {{
    for(let i=0; i<numSats; i++) {{
        const satGeom = new THREE.BoxGeometry(20, 20, 40);
        const satMat = new THREE.MeshStandardMaterial({{ color: 0xaaaaaa }});
        const sat = new THREE.Mesh(satGeom, satMat);
        const orbitRadius = RE + 300 + Math.random() * 1500;
        const speed = 0.0005 + Math.random() * 0.001;
        const angle = Math.random() * Math.PI * 2;
        const inclination = (Math.random() - 0.5) * Math.PI * 0.6;
        
        satellites.push({{ mesh: sat, radius: orbitRadius, speed: speed, angle: angle, inc: inclination }});
        scene.add(sat);
        
        if (showOrbits) {{
            const orbitGeom = new THREE.RingGeometry(orbitRadius, orbitRadius+2, 64);
            orbitGeom.rotateX(Math.PI/2);
            orbitGeom.rotateZ(inclination);
            const orbitMat = new THREE.MeshBasicMaterial({{ color: 0x444444, side: THREE.DoubleSide, opacity: 0.15, transparent: true }});
            const orbit = new THREE.Mesh(orbitGeom, orbitMat);
            scene.add(orbit);
        }}
    }}
}}

function updateSatellites() {{
    satellites.forEach(s => {{
        s.angle += s.speed;
        const x = s.radius * Math.cos(s.angle);
        const z = s.radius * Math.sin(s.angle);
        s.mesh.position.set(x, z * Math.sin(s.inc), z * Math.cos(s.inc));
        s.mesh.lookAt(0,0,0);
    }});
}}

function separatePart(part, forward) {{
    if (!part || !part.parent) return;
    const worldPos = new THREE.Vector3();
    const worldQuat = new THREE.Quaternion();
    part.getWorldPosition(worldPos);
    part.getWorldQuaternion(worldQuat);
    rocket.remove(part);
    scene.add(part);
    part.position.copy(worldPos);
    part.quaternion.copy(worldQuat);
    const debrisVel = forward.clone().multiplyScalar(-0.5).add(
        new THREE.Vector3((Math.random()-0.5)*0.1, (Math.random()-0.5)*0.1, (Math.random()-0.5)*0.1)
    );
    debrisList.push({{ mesh: part, vel: debrisVel }});
}}

function resetStaging() {{
    debrisList.forEach(d => scene.remove(d.mesh));
    debrisList = [];
    rocketParts.forEach(part => {{
        if (part.parent !== rocket) {{
            scene.remove(part);
            rocket.add(part);
        }}
        part.position.set(0, 0, 0);
        part.rotation.set(0, 0, 0);
    }});
    currentStageName = "";
}}

function updateSimulationToFrame(idx) {{
    if (idx < 0 || idx >= trajectory.length) return;
    const p = trajectory[idx];
    const pos = new THREE.Vector3(p.x/1000, p.y/1000, p.z/1000);
    
    if (idx < lastIdx) {{
        resetStaging();
    }}
    
    if (p.f !== currentStageName) {{
        const stagesFound = [...new Set(trajectory.map(pt => pt.f))];
        const currentStageIdx = stagesFound.indexOf(p.f);
        rocketParts.forEach(part => {{
            const partStageIdx = stagesFound.indexOf(part.userData.name);
            if (part.parent === rocket && partStageIdx < currentStageIdx && currentStageIdx !== -1) {{
                let forward = new THREE.Vector3(0, 1, 0);
                if (idx > 0) {{
                    const prevP = trajectory[idx-1];
                    forward.set(pos.x - prevP.x/1000, pos.y - prevP.y/1000, pos.z - prevP.z/1000).normalize();
                }}
                separatePart(part, forward);
            }}
        }});
        currentStageName = p.f;
    }}
    
    rocket.position.copy(pos);
    let forward = new THREE.Vector3();
    if (idx > 0) {{
        const prevP = trajectory[idx-1];
        forward.set(pos.x - prevP.x/1000, pos.y - prevP.y/1000, pos.z - prevP.z/1000).normalize();
    }} else {{
        forward.copy(pos).normalize();
    }}
    if (forward.lengthSq() > 0.001) {{
        const targetPos = pos.clone().add(forward);
        rocket.lookAt(targetPos);
        rocket.rotateX(Math.PI/2);
    }}
    
    const isBallistic = p.f.includes("Carga") || p.f.includes("Payload") || p.f.includes("Orion");
    exhaust.visible = !isBallistic;
    if (exhaust.visible) {{
        exhaust.scale.set(1, 0.8 + Math.random() * 0.5, 1);
    }}
    
    const trailPoints = [];
    const step = Math.max(1, Math.floor(idx / 500));
    for (let i = 0; i <= idx; i += step) {{
        trailPoints.push(new THREE.Vector3(trajectory[i].x/1000, trajectory[i].y/1000, trajectory[i].z/1000));
    }}
    if (trailPoints.length > 0 && trailPoints[trailPoints.length-1].distanceTo(pos) > 1.0) {{
        trailPoints.push(pos.clone());
    }}
    rocketTrail.geometry.setFromPoints(trailPoints);
    
    if (showMoon && moon) {{
        const moonAngle = (p.t / 2358720) * Math.PI * 2;
        moon.position.set(384400 * Math.sin(moonAngle), 0, 384400 * Math.cos(moonAngle));
    }}
    
    const altitude = Math.sqrt(pos.x**2 + pos.y**2 + pos.z**2) - RE;
    document.getElementById('alt').innerText = altitude.toFixed(1);
    document.getElementById('vel').innerText = Math.round(p.v);
    document.getElementById('gforce').innerText = p.g.toFixed(2);
    document.getElementById('pres').innerText = (p.q / 1000).toFixed(2);
    document.getElementById('stage').innerText = p.f;
    document.getElementById('timeline').value = idx;
    
    if (cameraMode === "Cabina") {{
        controls.enabled = false;
        const camPos = pos.clone().addScaledVector(forward, 0.8);
        camera.position.copy(camPos);
        camera.lookAt(pos.clone().addScaledVector(forward, 10));
    }} else if (cameraMode === "Persecución") {{
        controls.enabled = false;
        const camOffset = forward.clone().multiplyScalar(-30).add(new THREE.Vector3(0, 10, 0));
        camera.position.copy(pos.clone().add(camOffset));
        camera.lookAt(pos);
    }} else {{
        controls.enabled = true;
        controls.target.copy(pos);
        controls.update();
    }}
    
    lastIdx = idx;
}}

function animate() {{
    requestAnimationFrame(animate);
    if (!isPaused) {{
        if (animIndex < trajectory.length - 1) {{
            animIndex = Math.min(animIndex + playbackSpeed, trajectory.length - 1);
            updateSimulationToFrame(Math.floor(animIndex));
        }} else {{
            isPaused = true;
            document.getElementById('playBtn').innerText = "▶ Play";
        }}
    }}
    debrisList.forEach(d => {{
        d.mesh.position.add(d.vel);
        d.mesh.rotation.x += 0.01;
        d.mesh.rotation.z += 0.005;
    }});
    updateSatellites();
    earth.rotation.y += 0.0002;
    if (moon) {{
        moon.rotation.y += 0.0005;
    }}
    renderer.render(scene, camera);
}}

function onWindowResize() {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
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
                        st.markdown("### 📉 Gráficos de Telemetría")
                        tab1, tab2, tab3 = st.tabs(["Altitud y Velocidad", "Fuerza G y Presión Dinámica", "Masa y Consumo"])
                        df_res = pd.DataFrame({
                            "Tiempo (s)": res["t"], "Altitud (km)": res["z"] / 1000.0,
                            "Velocidad (m/s)": np.sqrt(res["vx"]**2 + res["vz"]**2),
                            "Fuerza G (G)": res["g_force"], "Presión Dinámica (kPa)": res["q"] / 1000.0,
                            "Masa (kg)": res["m"], "Fase": res["etapa"],
                        })
                        with tab1:
                            c_alt = alt.Chart(df_res).mark_line(color="#00aaff").encode(x="Tiempo (s):Q", y="Altitud (km):Q").properties(title="Perfil de Altitud")
                            c_vel = alt.Chart(df_res).mark_line(color="#ffaa00").encode(x="Tiempo (s):Q", y="Velocidad (m/s):Q").properties(title="Perfil de Velocidad")
                            st.altair_chart(c_alt, use_container_width=True); st.altair_chart(c_vel, use_container_width=True)
                        with tab2:
                            c_g = alt.Chart(df_res).mark_line(color="#ff4d4d").encode(x="Tiempo (s):Q", y="Fuerza G (G):Q").properties(title="Fuerza G")
                            c_q = alt.Chart(df_res).mark_line(color="#10b981").encode(x="Tiempo (s):Q", y="Presión Dinámica (kPa):Q").properties(title="Presión Dinámica")
                            st.altair_chart(c_g, use_container_width=True); st.altair_chart(c_q, use_container_width=True)
                        with tab3:
                            c_mass = alt.Chart(df_res).mark_line(color="#8b5cf6").encode(x="Tiempo (s):Q", y="Masa (kg):Q").properties(title="Masa del Cohete")
                            st.altair_chart(c_mass, use_container_width=True)

                    with st.container(border=True):
                        st.markdown("### 📉 Datos Completos")
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
