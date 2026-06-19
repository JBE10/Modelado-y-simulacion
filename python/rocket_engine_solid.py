from abc import ABC, abstractmethod
import numpy as np

# --- 1. STRATEGY PATTERN: Numerical Integrators ---

class Integrator(ABC):
    @abstractmethod
    def step(self, f, t, y, h, *args):
        pass

class RK4Integrator(Integrator):
    def step(self, f, t, y, h, *args):
        k1 = f(t, y, *args)
        k2 = f(t + h/2, y + h/2 * k1, *args)
        k3 = f(t + h/2, y + h/2 * k2, *args)
        k4 = f(t + h, y + h * k3, *args)
        return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

class EulerIntegrator(Integrator):
    def step(self, f, t, y, h, *args):
        return y + h * f(t, y, *args)


# --- 2. DOMAIN ENTITIES: Rocket and Stages ---

class RocketStage:
    def __init__(self, name, thrust, prop_mass, dry_mass, burn_rate):
        self.name = name
        self.thrust = thrust
        self.prop_mass = prop_mass
        self.dry_mass = dry_mass
        self.burn_rate = burn_rate
        self.is_active = False

    @property
    def total_mass(self):
        return self.prop_mass + self.dry_mass

class Rocket:
    def __init__(self, payload_mass):
        self.stages = []
        self.payload_mass = payload_mass

    def add_stage(self, stage: RocketStage):
        self.stages.append(stage)

    def get_current_mass(self, active_stage_index, current_prop_mass):
        # Mass = Payload + Current Active Stage (Dry + Rem. Prop) + All future stages
        m = self.payload_mass
        for i in range(active_stage_index, len(self.stages)):
            if i == active_stage_index:
                m += self.stages[i].dry_mass + current_prop_mass
            else:
                m += self.stages[i].total_mass
        return m


# --- 3. PHYSICS MODEL (Open/Closed Principle) ---

class PhysicsModel(ABC):
    @abstractmethod
    def get_derivatives(self, t, state, thrust, burn_rate, current_mass):
        pass

    @abstractmethod
    def get_telemetry_vars(self, state, thrust, current_mass):
        pass

class EarthRocketPhysics(PhysicsModel):
    def __init__(self, cd=0.4, area=10.5):
        self.Re = 6371000.0
        self.g0 = 9.81
        self.rho0 = 1.225
        self.H = 8000.0
        self.cd = cd
        self.area = area

    def get_derivatives(self, t, state, thrust, burn_rate, current_mass):
        x, z, vx, vz = state[:4]
        
        # Clampear z para evitar overflow/inestabilidad cuando z < 0
        z_safe = max(0.0, z)
        
        # Gravity (Central Field)
        r = self.Re + z_safe
        g = self.g0 * (self.Re / r)**2
        
        # Atmosphere (Simplified exponential)
        rho = self.rho0 * np.exp(-z_safe / self.H) if z_safe < 120000 else 0.0
        
        # Drag (Always opposes velocity vector, with Mach-dependent Cd)
        v = np.sqrt(vx**2 + vz**2)
        
        # Mach and Cd calculation
        mach = v / 340.0
        if mach < 0.8:
            cd_val = 0.3
        elif mach < 1.2:
            cd_val = 0.3 + 0.3 * (mach - 0.8) / 0.4
        elif mach < 2.0:
            cd_val = 0.6 - 0.2 * (mach - 1.2) / 0.8
        else:
            cd_val = 0.25 + 0.3 / mach
            
        D = 0.5 * rho * v**2 * cd_val * self.area
        
        # Pitch Control (Gravity Turn Optimizado)
        # 90 deg al inicio, curva gradual hacia 0 deg (horizontal)
        if z_safe < 1000: # Despegue vertical inicial
            theta = np.pi / 2
        else:
            # Perfil de pitch más técnico: se inclina según la altitud para ganar velocidad orbital
            # A 150km ya debería ser casi horizontal para orbitar/escapar
            theta = np.pi / 2 * np.exp(-(z_safe-1000) / 80000.0)
            theta = max(0.02, theta) # Nunca 0 absoluto para mantener componente radial
            
        # Equations of motion in a local vertical frame (Polar-like)
        # Adding Centrifugal term to vz to allow orbital stability
        if v < 1e-6:
            dvx = (thrust * np.cos(theta)) / current_mass
            dvz = (thrust * np.sin(theta)) / current_mass - g
        else:
            # dvz includes Centrifugal acceleration (vx^2 / r)
            dvx = (thrust * np.cos(theta) - D * (vx / v)) / current_mass - (vx * vz) / r
            dvz = (thrust * np.sin(theta) - D * (vz / v)) / current_mass - g + (vx**2 / r)
            
        return np.array([vx, vz, dvx, dvz])

    def get_telemetry_vars(self, state, thrust, current_mass):
        x, z, vx, vz = state[:4]
        z_safe = max(0.0, z)
        v = np.sqrt(vx**2 + vz**2)
        
        # Gravity
        r = self.Re + z_safe
        
        # Density
        rho = self.rho0 * np.exp(-z_safe / self.H) if z_safe < 120000 else 0.0
        
        # Mach and Cd
        mach = v / 340.0
        if mach < 0.8:
            cd_val = 0.3
        elif mach < 1.2:
            cd_val = 0.3 + 0.3 * (mach - 0.8) / 0.4
        elif mach < 2.0:
            cd_val = 0.6 - 0.2 * (mach - 1.2) / 0.8
        else:
            cd_val = 0.25 + 0.3 / mach
            
        D = 0.5 * rho * v**2 * cd_val * self.area
        q_val = 0.5 * rho * v**2
        
        # Pitch
        if z_safe < 1000:
            theta = np.pi / 2
        else:
            theta = np.pi / 2 * np.exp(-(z_safe-1000) / 80000.0)
            theta = max(0.02, theta)
            
        # Felt acceleration (non-gravitational)
        if z_safe <= 0.0 and thrust == 0.0 and v < 1e-3:
            g_force = 1.0  # standing on the ground
        else:
            if v < 1e-6:
                ax_felt = (thrust * np.cos(theta)) / current_mass
                az_felt = (thrust * np.sin(theta)) / current_mass
            else:
                ax_felt = (thrust * np.cos(theta) - D * (vx / v)) / current_mass
                az_felt = (thrust * np.sin(theta) - D * (vz / v)) / current_mass
                
            a_felt = np.sqrt(ax_felt**2 + az_felt**2)
            g_force = a_felt / 9.81
            
            # If on ground, normal force provides 1G support
            if z_safe <= 0.0 and g_force < 1.0:
                g_force = 1.0
                
        return g_force, q_val, cd_val


# --- 4. FACADE / ENGINE: Orchestrates the Simulation ---

class SimulationEngine:
    def __init__(self, rocket: Rocket, physics: PhysicsModel, integrator: Integrator):
        self.rocket = rocket
        self.physics = physics
        self.integrator = integrator

    def run(self, h_step=0.5, t_ballistic=300):
        all_times = []
        all_states = []
        all_stages = []
        all_g = []
        all_q = []
        
        t = 0.0
        # Initial State: x, z, vx, vz
        state = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Iterate through stages
        for idx, stage in enumerate(self.rocket.stages):
            current_prop = float(stage.prop_mass)
            burn_time = stage.prop_mass / stage.burn_rate
            t_end = t + burn_time
            
            while t < t_end:
                dt = min(h_step, t_end - t)
                current_prop = max(0.0, current_prop) # Protect against negative mass
                m = self.rocket.get_current_mass(idx, current_prop)
                
                # We wrap the derivative to inject current mass
                def f_wrap(t_val, s_val, _thrust=stage.thrust, _br=stage.burn_rate, _m=m):
                    return self.physics.get_derivatives(t_val, s_val, _thrust, _br, _m)
                
                new_state = self.integrator.step(f_wrap, t, state, dt)
                
                # Check for numerical explosion
                if np.any(np.isnan(new_state)) or np.any(np.isinf(new_state)):
                    break
                
                state = new_state
                current_prop -= stage.burn_rate * dt
                t += dt
                
                if state[1] < 0: # Ground contact
                    state[1] = 0; state[3] = 0
                
                # Compute telemetry
                g_force, q_val, cd_val = self.physics.get_telemetry_vars(state, stage.thrust, m)
                
                all_times.append(t)
                all_states.append(list(state) + [m])
                all_stages.append(stage.name)
                all_g.append(g_force)
                all_q.append(q_val)
        
        # Ballistic phase
        t_final = t + t_ballistic
        while t < t_final:
            dt = min(h_step, t_final - t)
            m = self.rocket.payload_mass
            
            def f_bal(t_val, s_val, _m=m):
                return self.physics.get_derivatives(t_val, s_val, 0.0, 0.0, _m)
            
            new_state = self.integrator.step(f_bal, t, state, dt)
            
            if np.any(np.isnan(new_state)) or np.any(np.isinf(new_state)):
                break
                
            state = new_state
            t += dt
            
            if state[1] < 0: # Ground contact (CRITICAL FIX)
                state[1] = 0; state[3] = 0; state[2] *= 0.9 # Friction
                # Compute telemetry
                g_force, q_val, cd_val = self.physics.get_telemetry_vars(state, 0.0, m)
                # Register the final contact point and exit ballistic loop
                all_times.append(t)
                all_states.append(list(state) + [m])
                all_stages.append("Carga Útil")
                all_g.append(g_force)
                all_q.append(q_val)
                break
            
            # Compute telemetry
            g_force, q_val, cd_val = self.physics.get_telemetry_vars(state, 0.0, m)
            all_times.append(t)
            all_states.append(list(state) + [m])
            all_stages.append("Carga Útil")
            all_g.append(g_force)
            all_q.append(q_val)

        return {
            't': np.array(all_times),
            'x': np.array([s[0] for s in all_states]),
            'z': np.array([s[1] for s in all_states]),
            'vx': np.array([s[2] for s in all_states]),
            'vz': np.array([s[3] for s in all_states]),
            'm': np.array([s[4] for s in all_states]),
            'etapa': all_stages,
            'g_force': np.array(all_g),
            'q': np.array(all_q)
        }
