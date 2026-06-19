import numpy as np

class Physics:
    def __init__(self):
        self.Re = 6371000.0
        self.g0 = 9.81
        self.rho0 = 1.225
        self.H = 8000.0
        self.cd = 0.4
        self.area = 10.5
        
        self.Rmoon = 384400000.0
        self.GMmoon = 4.9048695e12
        self.Tmoon = 2358720.0

    def get_derivatives(self, t, state, thrust, burn_rate, current_mass):
        x, z, vx, vz = state[:4]
        z_safe = max(0.0, z)
        
        # Earth Gravity
        r = self.Re + z_safe
        g = self.g0 * (self.Re / r)**2
        
        # Atmosphere
        rho = self.rho0 * np.exp(-z_safe / self.H) if z_safe < 120000 else 0.0
        v = np.sqrt(vx**2 + vz**2)
        
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
        
        # Pitch
        if z_safe < 1000:
            theta = np.pi / 2
        else:
            theta = np.pi / 2 * np.exp(-(z_safe-1000) / 80000.0)
            theta = max(0.02, theta)
            
        # Moon Gravity
        phi_moon = (t / self.Tmoon) * 2.0 * np.pi
        phi = x / self.Re
        
        X = r * np.sin(phi)
        Z = r * np.cos(phi)
        
        X_moon = self.Rmoon * np.sin(phi_moon)
        Z_moon = self.Rmoon * np.cos(phi_moon)
        
        d_rocket_moon = np.sqrt((X - X_moon)**2 + (Z - Z_moon)**2)
        if d_rocket_moon > 1e-3:
            a_moon_X = self.GMmoon * (X_moon - X) / d_rocket_moon**3
            a_moon_Z = self.GMmoon * (Z_moon - Z) / d_rocket_moon**3
        else:
            a_moon_X = 0.0
            a_moon_Z = 0.0
            
        a_moon_x = a_moon_X * np.cos(phi) - a_moon_Z * np.sin(phi)
        a_moon_z = a_moon_X * np.sin(phi) + a_moon_Z * np.cos(phi)
        
        if v < 1e-6:
            dvx = (thrust * np.cos(theta)) / current_mass + a_moon_x
            dvz = (thrust * np.sin(theta)) / current_mass - g + a_moon_z
        else:
            dvx = (thrust * np.cos(theta) - D * (vx / v)) / current_mass - (vx * vz) / r + a_moon_x
            dvz = (thrust * np.sin(theta) - D * (vz / v)) / current_mass - g + (vx**2 / r) + a_moon_z
            
        return np.array([vx, vz, dvx, dvz])

def run_simulation(physics, payload_mass, stage3_prop, start_angle=0.0, t_ballistic=800000):
    def rk4_step(f, t, y, h):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        
    stages = [
        {"name": "Boosters + Core", "T": 39100000, "m_p": 1600000, "m_d": 293000, "md": 12000},
        {"name": "Core Stage", "T": 9000000, "m_p": 600000, "m_d": 98000, "md": 2000},
        {"name": "ICPS", "T": 110100, "m_p": stage3_prop, "m_d": 3800, "md": 24}
    ]
    
    t = 0.0
    state = np.array([start_angle * physics.Re, 0.0, 0.0, 0.0])
    h_step = 0.5
    
    min_d_moon = float('inf')
    max_alt = 0.0
    returned_to_earth = False
    
    # Powered stages
    for idx, stage in enumerate(stages):
        current_prop = float(stage["m_p"])
        burn_time = stage["m_p"] / stage["md"]
        t_end = t + burn_time
        while t < t_end:
            dt = min(h_step, t_end - t)
            m = payload_mass
            for i in range(idx, len(stages)):
                if i == idx:
                    m += stage["m_d"] + current_prop
                else:
                    m += stages[i]["m_p"] + stages[i]["m_d"]
                    
            def f_wrap(t_val, s_val, _thrust=stage["T"], _br=stage["md"], _m=m):
                return physics.get_derivatives(t_val, s_val, _thrust, _br, _m)
                
            state = rk4_step(f_wrap, t, state, dt)
            current_prop -= stage["md"] * dt
            t += dt
            if state[1] < 0:
                state[1] = 0; state[3] = 0
                
            max_alt = max(max_alt, state[1])
            
            # Moon distance
            phi_moon = (t / physics.Tmoon) * 2.0 * np.pi
            phi = state[0] / physics.Re
            r = physics.Re + state[1]
            X = r * np.sin(phi)
            Z = r * np.cos(phi)
            X_moon = physics.Rmoon * np.sin(phi_moon)
            Z_moon = physics.Rmoon * np.cos(phi_moon)
            d_m = np.sqrt((X - X_moon)**2 + (Z - Z_moon)**2)
            min_d_moon = min(min_d_moon, d_m)
            
    # Ballistic phase
    t_final = t + t_ballistic
    while t < t_final:
        curr_dt = 30.0 if state[1] >= 120000.0 else h_step
        dt = min(curr_dt, t_final - t)
        
        def f_bal(t_val, s_val):
            return physics.get_derivatives(t_val, s_val, 0.0, 0.0, payload_mass)
            
        state = rk4_step(f_bal, t, state, dt)
        t += dt
        
        max_alt = max(max_alt, state[1])
        
        # Check for ground contact / re-entry
        if state[1] <= 0:
            returned_to_earth = True
            break
            
        phi_moon = (t / physics.Tmoon) * 2.0 * np.pi
        phi = state[0] / physics.Re
        r = physics.Re + state[1]
        X = r * np.sin(phi)
        Z = r * np.cos(phi)
        X_moon = physics.Rmoon * np.sin(phi_moon)
        Z_moon = physics.Rmoon * np.cos(phi_moon)
        d_m = np.sqrt((X - X_moon)**2 + (Z - Z_moon)**2)
        min_d_moon = min(min_d_moon, d_m)
        
    return min_d_moon, max_alt, returned_to_earth, state, t

def search():
    physics = Physics()
    payload = 26520
    
    print("Searching for Free-Return trajectories...")
    # Scan propellants from 27000 to 31000 in steps of 500
    for prop in np.linspace(27000, 31000, 9):
        # Fine scan of 120 angles
        angles = np.linspace(0, 2*np.pi, 120)
        for ang in angles:
            min_d, max_alt, returned, final_state, sim_t = run_simulation(physics, payload, prop, start_angle=ang, t_ballistic=800000)
            
            # We want to flyby the moon (close than 40,000 km) and return to Earth (returned is True)
            if min_d < 40000000.0 and returned:
                # Let's print candidate trajectories!
                # Wait, the moon distance should be small! 384,400 km is moon orbit.
                # If we get closer than 50,000 km to the Moon, it's a real flyby.
                if min_d < 50000000.0:
                    print(f"Prop: {prop:.0f} kg | Angle: {ang:.4f} rad ({np.degrees(ang):.1f} deg) | "
                          f"Min Moon Dist: {min_d/1000.0:.1f} km | Max Alt: {max_alt/1000.0:.1f} km | "
                          f"Returned: {returned} | Flight Time: {sim_t:.0f} s")

if __name__ == "__main__":
    search()
