import math
import numpy as np
from noise import pnoise3
import matplotlib.pyplot as plt
import random

###############################################################################
# 1. Deterministic fractal class (3D) using pnoise3 for fractal sum (fBm)
###############################################################################
class DeterministicFractal3D:
    def __init__(self, global_config=None):
        self.global_config = global_config

    def sample(self, x, y, z, base_freq=0.15, octaves=4, lacunarity=2.5, gain=0.6):
        """
        A fractal sum (fBm) approach using Perlin noise pnoise3. Returns ~[-1..1].
        We'll do a manual loop for octaves to have direct control.
        """
        val = 0.0
        amplitude = 1.0
        freq = base_freq

        for _ in range(octaves):
            noise_val = pnoise3(
                freq * x,
                freq * y,
                freq * z,
                repeatx=1024, 
                repeaty=1024, 
                repeatz=1024,
                base=0, 
                octaves=1
            )
            val += amplitude * noise_val
            freq *= lacunarity
            amplitude *= gain

        # clamp final to [-1..1]
        return max(-1.0, min(1.0, val))


###############################################################################
# 2. Agent class: camera + fractal parameters + "weight" as part of the action
###############################################################################
class FractalCameraAgent:
    """
    - fractal_obj: reference to DeterministicFractal3D
    - pos3D, pitch, yaw, fov_deg
    - base_freq, octaves, lacunarity, gain, blend_weight
    - image_size
    """
    def __init__(self,
                 fractal_obj,
                 pos3D=np.array([0.0, 0.0, 0.0]),
                 pitch=0.0,
                 yaw=0.0,
                 fov_deg=60.0,
                 base_freq=0.15,
                 octaves=4,
                 lacunarity=2.5,
                 gain=0.6,
                 blend_weight=1.0,
                 image_size=50):
        
        self.fractal = fractal_obj

        self.pos3D = pos3D
        self.pitch = pitch
        self.yaw   = yaw
        self.fov_deg = fov_deg

        self.base_freq   = base_freq
        self.octaves     = octaves
        self.lacunarity  = lacunarity
        self.gain        = gain
        self.blend_weight= blend_weight

        self.image_size  = image_size
        self.sample_dist = 10.0  

    def apply_action(self, action_dict, dt=0.1,
                     action_ranges=None):
        """
        action_dict => e.g. { 'dPos': (x,y,z in [-1..1]),
                              'dPitch': float in [-1..1],
                              ... }
        We'll scale them from [-1..1] to [MinVal..MaxVal] as per 'action_ranges'.
        Then we clamp final environment variables if needed.
        """
        if not action_ranges:
            # fallback: do minimal
            action_ranges = {}

        # We'll define a helper to map [-1..1] => [low, high].
        def scale_action(val, rng):
            return self._map(val, -1.0, 1.0, rng[0], rng[1])

        if 'dPos' in action_dict:
            dPosXYZ = action_dict['dPos']
            # interpret each component in [-1..1], scale to pos_minmax
            pos_minmax = action_ranges.get('pos_minmax', [-0.1, 0.1])
            dx = scale_action(dPosXYZ[0], pos_minmax)
            dy = scale_action(dPosXYZ[1], pos_minmax)
            dz = scale_action(dPosXYZ[2], pos_minmax)
            self.pos3D += np.array([dx, dy, dz]) * dt

        if 'dPitch' in action_dict:
            pitch_minmax = action_ranges.get('pitch_minmax', [-0.05, 0.05])
            dpitch = scale_action(action_dict['dPitch'], pitch_minmax)
            self.pitch += dpitch * dt
            # clamp pitch to [-pi/2, pi/2], or something
            limit = math.pi/2 - 0.01
            self.pitch = max(-limit, min(limit, self.pitch))

        if 'dYaw' in action_dict:
            yaw_minmax = action_ranges.get('yaw_minmax', [-0.05, 0.05])
            dyaw = scale_action(action_dict['dYaw'], yaw_minmax)
            self.yaw += dyaw * dt
            # wrap yaw
            while self.yaw < 0.0:
                self.yaw += 2.0 * math.pi
            while self.yaw >= 2.0 * math.pi:
                self.yaw -= 2.0 * math.pi

        if 'dBaseFreq' in action_dict:
            bf_minmax = action_ranges.get('base_freq_minmax', [-0.01, 0.01])
            dbf = scale_action(action_dict['dBaseFreq'], bf_minmax)
            self.base_freq = max(0.01, self.base_freq + dbf * dt)

        if 'dLacunarity' in action_dict:
            lac_minmax = action_ranges.get('lacunarity_minmax', [-0.01, 0.01])
            dlac = scale_action(action_dict['dLacunarity'], lac_minmax)
            self.lacunarity = max(1.0, min(5.0, self.lacunarity + dlac * dt))

        if 'dGain' in action_dict:
            gain_minmax = action_ranges.get('gain_minmax', [-0.01, 0.01])
            dg = scale_action(action_dict['dGain'], gain_minmax)
            self.gain = max(0.0, min(1.0, self.gain + dg * dt))

        if 'dBlendWeight' in action_dict:
            bw_minmax = action_ranges.get('blend_weight_minmax', [-0.02, 0.02])
            dbw = scale_action(action_dict['dBlendWeight'], bw_minmax)
            self.blend_weight = max(0.0, min(5.0, self.blend_weight + dbw * dt))

    def render_fractal_image(self):
        """
        Renders NxN in [-1..1] using single-sample approach.
        """
        N = self.image_size
        half_fov = math.radians(self.fov_deg)*0.5
        out = np.zeros((N,N), dtype=float)

        for v in range(N):
            ndc_y = (v/(N-1))*2 -1
            for u in range(N):
                ndc_x = (u/(N-1))*2 -1

                cx = ndc_x * math.tan(half_fov)
                cy = -ndc_y*math.tan(half_fov)
                cz = 1.0
                length = math.sqrt(cx*cx + cy*cy + cz*cz)
                cx/= length
                cy/= length
                cz/= length

                # yaw
                sy = math.sin(self.yaw)
                cyw= math.cos(self.yaw)
                rx = cx*cyw + cz*sy
                rz = -cx*sy + cz*cyw
                ry = cy

                # pitch
                sp = math.sin(self.pitch)
                cp = math.cos(self.pitch)
                final_x = rx
                final_y = ry*cp - rz*sp
                final_z = ry*sp + rz*cp

                sx = self.pos3D[0]+ self.sample_dist*final_x
                sy_ =self.pos3D[1]+ self.sample_dist*final_y
                sz = self.pos3D[2] + self.sample_dist*final_z

                val = self.fractal.sample(sx, sy_, sz,
                    base_freq=self.base_freq,
                    octaves=self.octaves,
                    lacunarity=self.lacunarity,
                    gain=self.gain)
                out[v,u] = val
        return out

    def _map(self, x, in_min, in_max, out_min, out_max):
        if abs(in_max-in_min) < 1e-8:
            return out_min
        return (x-in_min)*(out_max-out_min)/(in_max-in_min) + out_min


###############################################################################
# 3. Multi-agent environment that sums agent fractal images * blend_weight
###############################################################################
class MultiAgentFractalWaveEnv:
    def __init__(self, sim_params, action_ranges):
        """
        sim_params: dictionary with environment and state ranges, e.g.:
            {
              "num_agents": 3,
              "image_size": 50,
              "agent_init": {...}, "fractal_init": {...},
              "state_ranges": {
                  "pos_range": [-5,5],
                  "pitch_range": [-1.57,1.57],
                  "yaw_range": [0,6.28318],
                  "base_freq_range": [0.01,1.0],
                  "lacunarity_range": [1,5],
                  "gain_range": [0,1],
                  "blend_weight_range": [0,5]
              }
            }

        action_ranges: dictionary for how to interpret agent actions in [-1..1]
           e.g. {
             "pos_minmax": [-0.1,0.1],
             "pitch_minmax": [-0.05,0.05],
             ...
           }
        """
        self.sim_params = sim_params
        self.action_ranges = action_ranges

        self.num_agents = sim_params.get("num_agents", 3)
        self.image_size = sim_params.get("image_size", 50)

        self.fractal_obj = DeterministicFractal3D(global_config=sim_params.get("fractal_global", None))

        self.agents = []
        self.final_wave = np.zeros((self.image_size, self.image_size), dtype=float)
        self._init_agents()

        # Optionally parse "state_ranges" for normalization
        self.state_ranges = sim_params.get("state_ranges", {
            "pos_range": [-5,5],
            "pitch_range": [-1.57,1.57],
            "yaw_range": [0, 6.28318],
            "base_freq_range": [0.01,1.0],
            "lacunarity_range": [1.0,5.0],
            "gain_range": [0.0,1.0],
            "blend_weight_range": [0.0,5.0]
        })

    def _init_agents(self):
        agent_init = self.sim_params.get("agent_init", {})
        fractal_init = self.sim_params.get("fractal_init", {})

        pos_lo, pos_hi = agent_init.get("pos_range", [-5,5])
        pitch_lo, pitch_hi = agent_init.get("pitch_range", [-0.3,0.3])
        yaw_lo, yaw_hi = agent_init.get("yaw_range", [0, 2*math.pi])
        fov_deg = agent_init.get("fov_deg", 60.0)
        sample_dist = agent_init.get("sample_dist", 10.0)

        bf_lo, bf_hi = fractal_init.get("base_freq_range", [0.15,0.2])
        lac_lo, lac_hi = fractal_init.get("lacunarity_range", [2.0,3.0])
        gain_lo, gain_hi = fractal_init.get("gain_range", [0.5,0.7])
        octaves = fractal_init.get("octaves", 4)
        blendw_lo, blendw_hi = fractal_init.get("blend_weight_range", [0.5,1.5])

        for _ in range(self.num_agents):
            px = np.random.uniform(pos_lo,pos_hi)
            py = np.random.uniform(pos_lo,pos_hi)
            pz = np.random.uniform(pos_lo,pos_hi)

            pitch= np.random.uniform(pitch_lo,pitch_hi)
            yaw  = np.random.uniform(yaw_lo,yaw_hi)

            bf   = np.random.uniform(bf_lo,bf_hi)
            lac  = np.random.uniform(lac_lo,lac_hi)
            gn   = np.random.uniform(gain_lo,gain_hi)
            bw   = np.random.uniform(blendw_lo,blendw_hi)

            agent = FractalCameraAgent(
                fractal_obj=self.fractal_obj,
                pos3D=np.array([px,py,pz]),
                pitch=pitch,
                yaw=yaw,
                fov_deg=fov_deg,
                base_freq=bf,
                octaves=octaves,
                lacunarity=lac,
                gain=gn,
                blend_weight=bw,
                image_size=self.image_size
            )
            agent.sample_dist= sample_dist
            self.agents.append(agent)

    def step(self, actions_list, dt=0.1):
        # 1) apply actions
        for i, agent in enumerate(self.agents):
            if i < len(actions_list):
                agent.apply_action(actions_list[i], dt,
                                   action_ranges=self.action_ranges)

        # 2) combine fractal images
        wave_sums   = np.zeros((self.image_size, self.image_size), dtype=float)
        weight_sums = np.zeros((self.image_size, self.image_size), dtype=float)

        for agent in self.agents:
            img = agent.render_fractal_image()
            wave_sums   += (img* agent.blend_weight)
            weight_sums += agent.blend_weight

        weight_sums = np.where(weight_sums==0, 1e-6, weight_sums)
        self.final_wave = wave_sums/ weight_sums

    def get_wave(self):
        """
        Final wave in [-1..1].
        """
        return self.final_wave

    def get_num_agents(self):
        return len(self.agents)

    def get_agent_state(self, agent_idx):
        """
        Return agent's state variables in [-1..1], referencing self.state_ranges.
        e.g. [posX, posY, posZ, pitch, yaw, baseFreq, lacunarity, gain, blendWeight].
        """
        if agent_idx<0 or agent_idx>= len(self.agents):
            return []

        st = self.agents[agent_idx]
        # parse the config
        sr = self.state_ranges

        pos_r = sr.get("pos_range", [-5,5])
        pitch_r= sr.get("pitch_range", [-1.57,1.57])
        yaw_r= sr.get("yaw_range", [0, 6.28318])
        bf_r= sr.get("base_freq_range", [0.01,1.0])
        lac_r= sr.get("lacunarity_range", [1.0,5.0])
        g_r=  sr.get("gain_range", [0.0,1.0])
        bw_r= sr.get("blend_weight_range", [0.0,5.0])

        # We'll define a local helper
        def normalize(val, minv, maxv):
            clip= max(minv, min(maxv,val))
            # map [minv..maxv] => [-1..1]
            if (maxv-minv)<1e-8:
                return 0.0
            return ( (clip- minv)/(maxv- minv))*2.0 -1.0

        px = normalize(st.pos3D[0], pos_r[0], pos_r[1])
        py = normalize(st.pos3D[1], pos_r[0], pos_r[1])
        pz = normalize(st.pos3D[2], pos_r[0], pos_r[1])
        pitchNorm= normalize(st.pitch, pitch_r[0], pitch_r[1])
        yawNorm  = normalize(st.yaw, yaw_r[0], yaw_r[1])
        bfNorm   = normalize(st.base_freq, bf_r[0], bf_r[1])
        lacNorm  = normalize(st.lacunarity, lac_r[0], lac_r[1])
        gNorm    = normalize(st.gain, g_r[0], g_r[1])
        bwNorm   = normalize(st.blend_weight, bw_r[0], bw_r[1])

        return [px, py, pz, pitchNorm, yawNorm, bfNorm, lacNorm, gNorm, bwNorm]


###############################################################################
# 4. A mock main() with random actions + 3D matplotlib
###############################################################################
def main():
    import matplotlib.animation as animation

    sim_params = {
        "num_agents": 5,
        "image_size": 50,
        "agent_init": {
            "pos_range": [-5,5],
            "pitch_range": [-0.3,0.3],
            "yaw_range": [0, 2*math.pi],
            "fov_deg": 60.0,
            "sample_dist": 10
        },
        "fractal_init": {
            "base_freq_range": [0.15,0.2],
            "lacunarity_range": [2.0, 3.0],
            "gain_range": [0.5, 0.7],
            "octaves": 4,
            "blend_weight_range": [0.5, 1.5]
        },
        "fractal_global": {
        },
        # "state_ranges" for normalizing agent state => [-1..1].
        "state_ranges": {
            "pos_range": [-5, 5],
            "pitch_range": [-1.57,1.57],
            "yaw_range": [0, 6.28318],
            "base_freq_range": [0.01,1.0],
            "lacunarity_range": [1.0,5.0],
            "gain_range": [0.0,1.0],
            "blend_weight_range": [0.0,5.0]
        }
    }

    # action in [-1..1], scaled to environment
    action_ranges = {
        "pos_minmax": [-0.1, 0.1],
        "pitch_minmax": [-0.05, 0.05],
        "yaw_minmax":   [-0.05, 0.05],
        "base_freq_minmax": [-0.01, 0.01],
        "lacunarity_minmax": [-0.01, 0.01],
        "gain_minmax": [-0.01, 0.01],
        "blend_weight_minmax": [-0.02, 0.02]
    }

    env = MultiAgentFractalWaveEnv(sim_params, action_ranges)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    N = sim_params["image_size"]
    X_vals = np.arange(N)
    Y_vals = np.arange(N)
    X_grid, Y_grid = np.meshgrid(X_vals, Y_vals)

    surface_holder = [None]

    def init_plot():
        ax.clear()
        wave = env.get_wave()  # [-1..1]
        Z = wave + 1.0
        ax.set_zlim(0,2)
        ax.set_title("Multi-Agent Fractal (Frame=0)")
        surf = ax.plot_surface(X_grid, Y_grid, Z, cmap='viridis', linewidth=0, antialiased=False)
        surface_holder[0] = surf
        return (surf,)

    def update_plot(frame):
        if surface_holder[0] is not None:
            surface_holder[0].remove()

        # random actions
        acts = []
        for _ in range(env.get_num_agents()):
            # each in [-1..1]
            dPos = np.random.uniform(-1,1,size=3)
            dPitch = random.uniform(-1,1)
            dYaw   = random.uniform(-1,1)
            dBaseFreq = random.uniform(-1,1)
            dLacunarity= random.uniform(-1,1)
            dGain= random.uniform(-1,1)
            dBlendW= random.uniform(-1,1)

            action = {
                "dPos": dPos,
                "dPitch": dPitch,
                "dYaw": dYaw,
                "dBaseFreq": dBaseFreq,
                "dLacunarity": dLacunarity,
                "dGain": dGain,
                "dBlendWeight": dBlendW
            }
            acts.append(action)

        env.step(acts, dt=0.2)

        wave = env.get_wave()
        Z = wave + 1.0
        ax.clear()
        ax.set_zlim(0,2)
        ax.set_title(f"Multi-Agent Fractal (Frame={frame})")
        surf = ax.plot_surface(X_grid, Y_grid, Z, cmap='viridis', linewidth=0, antialiased=False)
        surface_holder[0] = surf

        # For demonstration, let's print the first agent's normalized state:
        state = env.get_agent_state(0)  # e.g. agent 0
        # print("Agent0 State in [-1..1]:", state)

        return (surf,)

    ani = animation.FuncAnimation(fig, update_plot, frames=50, init_func=init_plot,
                                  blit=False, interval=400)
    plt.show()


if __name__ == "__main__":
    main()
