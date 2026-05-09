import streamlit as st
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
from cohete_model import simular_lanzamiento

st.set_page_config(page_title="Rocket Simulation Pro", layout="wide", page_icon="🚀")

# --- Custom CSS for a dark, pro theme ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #2e3192, #1bffff);
        color: white;
        border: none;
        font-weight: bold;
    }
    .stMetric {
        background: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Mission Control: Advanced Rocket Simulation")
st.markdown("Simulation of a multi-stage rocket launch with real-time 3D visualization, Earth physics, and satellite orbits.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("🚀 Rocket Parameters")
    
    with st.expander("Etapa 1 (Booster)", expanded=True):
        T1 = st.number_input("Thrust (N)", value=7600000, step=100000, key="T1_p")
        m1_prop = st.number_input("Propellant Mass (kg)", value=395700, key="m1p_p")
        m1_dry = st.number_input("Dry Mass (kg)", value=25600, key="m1d_p")
        m_dot1 = st.number_input("Burn Rate (kg/s)", value=2500, key="md1_p")
    
    with st.expander("Etapa 2 (Orbital)", expanded=False):
        T2 = st.number_input("Thrust (N) ", value=934000, step=10000, key="T2_p")
        m2_prop = st.number_input("Propellant Mass (kg) ", value=92670, key="m2p_p")
        m2_dry = st.number_input("Dry Mass (kg) ", value=3900, key="m2d_p")
        m_dot2 = st.number_input("Burn Rate (kg/s) ", value=260, key="md2_p")
    
    with st.expander("Payload & Sim", expanded=False):
        m_payload = st.number_input("Payload Mass (kg)", value=5000, key="mp_p")
        t_bal = st.number_input("Ballistic Phase (s)", value=600, key="tb_p")
        h_step = st.slider("Physics Step (s)", 0.1, 2.0, 0.5, key="h_p")

    st.divider()
    st.header("🛰️ Satellite Config")
    num_satellites = st.slider("Number of Satellites", 0, 50, 20)
    show_trajectories = st.checkbox("Show Satellite Orbits", value=True)

# --- Simulation Logic ---
from rocket_engine_solid import Rocket, RocketStage, EarthRocketPhysics, RK4Integrator, SimulationEngine

# Builder Pattern equivalent (Manual Construction)
rocket = Rocket(payload_mass=m_payload)
rocket.add_stage(RocketStage("Etapa 1 (Ignición)", T1, m1_prop, m1_dry, m_dot1))
rocket.add_stage(RocketStage("Etapa 2 (Vuelo Orbital)", T2, m2_prop, m2_dry, m_dot2))

# Strategy Pattern Injection
physics = EarthRocketPhysics()
integrator = RK4Integrator()
engine = SimulationEngine(rocket, physics, integrator)

res = engine.run(h_step=h_step, t_ballistic=t_ballistic)

# Prepare data for Three.js
# Convert 2D trajectory to 3D Earth coordinates
Re = 6371000.0  # Earth radius in meters
x = res['x']
z = res['z']
phi = x / Re
X_3d = (Re + z) * np.sin(phi)
Y_3d = np.zeros_like(X_3d)
Z_3d = (Re + z) * np.cos(phi)

# Flatten for JS transfer
trajectory_data = []
for i in range(len(X_3d)):
    trajectory_data.append({
        'x': float(X_3d[i]),
        'y': float(Y_3d[i]),
        'z': float(Z_3d[i]),
        'fase': res['etapa'][i]
    })

# --- Metrics ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Max Altitude", f"{np.max(z)/1000:.1f} km")
m2.metric("Max Velocity", f"{np.max(np.sqrt(res['vx']**2 + res['vz']**2)):.0f} m/s")
m3.metric("Final Mass", f"{res['m'][-1]:.0f} kg")
m4.metric("Burn Duration", f"{res['t'][-1]:.0f} s")

# --- Three.js Component ---
# We'll use a CDN for textures or high-quality colors if CDN fails.
three_js_code = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ margin: 0; background-color: #000; overflow: hidden; }}
        #info {{
            position: absolute; top: 10px; left: 10px; color: white;
            font-family: 'Segoe UI', sans-serif; pointer-events: none;
            text-shadow: 1px 1px 2px black;
        }}
    </style>
</head>
<body>
    <div id="info">Altitude: <span id="alt">0</span> km | Velocity: <span id="vel">0</span> m/s</div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        const trajectory = {trajectory_data};
        const numSats = {num_satellites};
        const showOrbits = {str(show_trajectories).lower()};
        
        let scene, camera, renderer, controls;
        let earth, rocket, rocketTrail;
        let satellites = [];
        let index = 0;
        let clock = new THREE.Clock();
        
        const RE = 6371; // In KM for visualization scale (1 unit = 1 KM)
        
        function init() {{
            scene = new THREE.Scene();
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 100000);
            camera.position.set(RE * 2, RE, RE * 2);
            
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.body.appendChild(renderer.domElement);
            
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            
            // Lights
            const ambientLight = new THREE.AmbientLight(0x404040, 2);
            scene.add(ambientLight);
            
            const sunLight = new THREE.DirectionalLight(0xffffff, 1.5);
            sunLight.position.set(10000, 5000, 5000);
            scene.add(sunLight);
            
            // Earth
            const geometry = new THREE.SphereGeometry(RE, 64, 64);
            const material = new THREE.MeshPhongMaterial({{
                color: 0x156289,
                emissive: 0x072534,
                side: THREE.DoubleSide,
                flatShading: false,
                specular: 0x050505,
                shininess: 10
            }});
            
            // Try to load texture
            const loader = new THREE.TextureLoader();
            loader.load('https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_atmos_2048.jpg', (tex) => {{
                material.map = tex;
                material.color.set(0xffffff);
                material.needsUpdate = true;
            }});
            
            earth = new THREE.Mesh(geometry, material);
            scene.add(earth);
            
            // Atmosphere glow
            const atmosGeom = new THREE.SphereGeometry(RE * 1.02, 64, 64);
            const atmosMat = new THREE.MeshBasicMaterial({{
                color: 0x00aaff,
                transparent: true,
                opacity: 0.1,
                side: THREE.BackSide
            }});
            const atmosphere = new THREE.Mesh(atmosGeom, atmosMat);
            scene.add(atmosphere);

            // Stars
            const starsGeom = new THREE.BufferGeometry();
            const starsPos = [];
            for(let i=0; i<10000; i++) {{
                starsPos.push((Math.random()-0.5)*20000);
                starsPos.push((Math.random()-0.5)*20000);
                starsPos.push((Math.random()-0.5)*20000);
            }}
            starsGeom.setAttribute('position', new THREE.Float32BufferAttribute(starsPos, 3));
            const starsMat = new THREE.PointsMaterial({{ color: 0xffffff, size: 2 }});
            const stars = new THREE.Points(starsGeom, starsMat);
            scene.add(stars);
            
            // Rocket
            const rocketGeom = new THREE.ConeGeometry(50, 200, 8);
            rocketGeom.rotateX(Math.PI/2);
            const rocketMat = new THREE.MeshStandardMaterial({{ color: 0xffffff, metalness: 0.8, roughness: 0.2 }});
            rocket = new THREE.Mesh(rocketGeom, rocketMat);
            scene.add(rocket);
            
            // Rocket Trail
            const trailMat = new THREE.LineBasicMaterial({{ color: 0xffaa00, linewidth: 2 }});
            const trailGeom = new THREE.BufferGeometry();
            rocketTrail = new THREE.Line(trailGeom, trailMat);
            scene.add(rocketTrail);
            
            // Satellites
            createSatellites();
            
            animate();
        }}
        
        function createSatellites() {{
            for(let i=0; i<numSats; i++) {{
                const satGeom = new THREE.BoxGeometry(20, 20, 40);
                const satMat = new THREE.MeshStandardMaterial({{ color: 0xaaaaaa }});
                const sat = new THREE.Mesh(satGeom, satMat);
                
                // Random orbit
                const orbitRadius = RE + 400 + Math.random() * 2000;
                const speed = 0.001 + Math.random() * 0.002;
                const angle = Math.random() * Math.PI * 2;
                const inclination = (Math.random() - 0.5) * Math.PI;
                
                satellites.push({{ mesh: sat, radius: orbitRadius, speed: speed, angle: angle, inc: inclination }});
                scene.add(sat);
                
                if(showOrbits) {{
                    const orbitGeom = new THREE.RingGeometry(orbitRadius, orbitRadius+2, 64);
                    orbitGeom.rotateX(Math.PI/2);
                    orbitGeom.rotateZ(inclination);
                    const orbitMat = new THREE.MeshBasicMaterial({{ color: 0x444444, side: THREE.DoubleSide, opacity: 0.2, transparent: true }});
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
                
                // Apply inclination (simple rotation)
                s.mesh.position.set(
                    x,
                    z * Math.sin(s.inc),
                    z * Math.cos(s.inc)
                );
                s.mesh.lookAt(0,0,0);
            }});
        }}
        
        function animate() {{
            requestAnimationFrame(animate);
            
            // Rocket Simulation Playback
            if (index < trajectory.length) {{
                const pos = trajectory[index];
                const x = pos.x / 1000;
                const y = pos.y / 1000;
                const z = pos.z / 1000;
                
                rocket.position.set(x, y, z);
                
                // Point rocket in direction of velocity (tangent to path)
                if (index > 0) {{
                    const prev = trajectory[index-1];
                    const dir = new THREE.Vector3(x - prev.x/1000, y - prev.y/1000, z - prev.z/1000).normalize();
                    const target = new THREE.Vector3(x, y, z).add(dir);
                    rocket.lookAt(target);
                }}
                
                // Update Trail
                const trailPoints = [];
                for(let i=0; i<=index; i++) {{
                    trailPoints.push(new THREE.Vector3(trajectory[i].x/1000, trajectory[i].y/1000, trajectory[i].z/1000));
                }}
                rocketTrail.geometry.setFromPoints(trailPoints);
                
                // HUD
                const altitude = Math.sqrt(x*x + y*y + z*z) - RE;
                document.getElementById('alt').innerText = altitude.toFixed(1);
                
                index++;
            }} else {{
               // Loop animation or stay at end
               // index = 0; 
            }}
            
            updateSatellites();
            controls.update();
            renderer.render(scene, camera);
        }}
        
        init();
        
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>
"""

st.components.v1.html(three_js_code, height=600)

# --- Analysis Tables ---
st.divider()
st.subheader("📊 Telemetry Data")
df_res = pd.DataFrame({
    'Time (s)': res['t'],
    'Alt (km)': res['z'] / 1000,
    'Velocity (m/s)': np.sqrt(res['vx']**2 + res['vz']**2),
    'Mass (kg)': res['m'],
    'Stage': res['etapa']
})
st.dataframe(df_res.iloc[::20], use_container_width=True)
