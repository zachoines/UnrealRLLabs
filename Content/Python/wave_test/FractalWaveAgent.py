import math
import numpy as np
from noise import pnoise3
import matplotlib.pyplot as plt

###############################################################################
# 1. Deterministic fractal class (3D)
###############################################################################
class DeterministicFractal3D:
    def __init__(self, global_config=None):
        self.global_config = global_config

    """
    Uses Perlin-based fractal summation (fBm) via pnoise3 from the 'noise' library.
    """
    def sample(self, x, y, z, base_freq=0.15, octaves=4, lacunarity=2.5, gain=0.6):
        """
        A fractal sum (fBm) approach using Perlin noise. Returns ~[-1..1].
        
        We do a manual loop over 'octaves':
          per octave i:
            val += amplitude * pnoise3(freq*x, freq*y, freq*z, ...)
            freq *= lacunarity
            amplitude *= gain
        """
        val = 0.0
        amplitude = 1.0
        freq = base_freq

        for _ in range(2):
            # We'll pass (freq*x, freq*y, freq*z) so each octave has a higher frequency.
            noise_val = pnoise3(freq * x,
                                freq * y,
                                freq * z,
                                # Repeat or base parameters can go here if you want tiling:
                                repeatx=1024, repeaty=1024, repeatz=1024,
                                base=0, 
                                octaves=1)  # We call pnoise3 with octaves=1 for each "layer"

            val += amplitude * noise_val

            freq *= lacunarity
            amplitude *= gain

        # for _ in range(octaves):
        #     layer = (math.sin(freq*x + 1.3)*math.cos(freq*y + 2.1) +
        #              math.sin(freq*y + 4.2)*math.cos(freq*z + 1.7) +
        #              math.sin(freq*z + 3.1)*math.cos(freq*x + 0.9)) / 3.0
        #     val += amplitude * layer
        #     amplitude *= gain
        #     freq *= lacunarity

        # clamp final to [-1..1]
        val = max(-1.0, min(1.0, val))
        return val

###############################################################################
# 2. Agent class: camera + fractal parameters + "weight" as part of the action
###############################################################################
class FractalCameraAgent:
    """
    Each agent has:
      - fractal_obj: a reference to DeterministicFractal3D
      - camera position/orientation (pos3D, pitch, yaw, fov)
      - fractal params: base_freq, octaves, lacunarity, gain
      - blend_weight: how strongly this agent's image affects the final wave
      - image_size: NxN resolution
    The agent's 'actions' can modify these states. 
    """

    def __init__(self,
                 fractal_obj,
                 pos3D=np.array([0,0,0], dtype=float),
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
        self.sample_dist = 10.0  # distance at which we sample the fractal

    def apply_action(self, action_dict, dt=0.1):
        """
        Each key in action_dict modifies a parameter. 
        E.g. action_dict might contain:
          - 'dPos' (np.array(3)) for movement
          - 'dPitch', 'dYaw' for camera orientation
          - 'dBaseFreq', 'dLacunarity', 'dGain'
          - 'dBlendWeight'
        We'll clamp certain values to keep them in a valid range.
        """
        if 'dPos' in action_dict:
            self.pos3D += action_dict['dPos'] * dt

        if 'dPitch' in action_dict:
            self.pitch += action_dict['dPitch'] * dt
            limit = math.pi/2 - 0.01
            self.pitch = max(-limit, min(limit, self.pitch))

        if 'dYaw' in action_dict:
            self.yaw += action_dict['dYaw'] * dt
            self.yaw %= 2*math.pi

        if 'dBaseFreq' in action_dict:
            self.base_freq += action_dict['dBaseFreq'] * dt
            self.base_freq = max(0.01, self.base_freq)

        if 'dLacunarity' in action_dict:
            self.lacunarity += action_dict['dLacunarity'] * dt
            self.lacunarity = max(1.0, min(5.0, self.lacunarity))

        if 'dGain' in action_dict:
            self.gain += action_dict['dGain'] * dt
            self.gain = max(0.0, min(1.0, self.gain))

        if 'dBlendWeight' in action_dict:
            self.blend_weight += action_dict['dBlendWeight'] * dt
            self.blend_weight = max(0.0, min(5.0, self.blend_weight))

    def render_fractal_image(self):
        """
        Render NxN perspective image in [-1..1], by single-sample approach.
        """
        N = self.image_size
        half_fov = math.radians(self.fov_deg)*0.5
        out = np.zeros((N,N), dtype=float)

        for v in range(N):
            ndc_y = (v/(N-1))*2 - 1
            for u in range(N):
                ndc_x = (u/(N-1))*2 - 1

                # pinhole direction
                cx = ndc_x * math.tan(half_fov)
                cy = -ndc_y*math.tan(half_fov)
                cz = 1.0
                length = math.sqrt(cx*cx + cy*cy + cz*cz)
                cx /= length
                cy /= length
                cz /= length

                # yaw
                sy = math.sin(self.yaw)
                cyw = math.cos(self.yaw)
                rx = cx*cyw + cz*sy
                rz = -cx*sy + cz*cyw
                ry = cy

                # pitch
                sp = math.sin(self.pitch)
                cp = math.cos(self.pitch)
                final_x = rx
                final_y = ry*cp - rz*sp
                final_z = ry*sp + rz*cp

                # sample position
                sx = self.pos3D[0] + self.sample_dist*final_x
                sy_ = self.pos3D[1] + self.sample_dist*final_y
                sz = self.pos3D[2] + self.sample_dist*final_z

                val = self.fractal.sample(sx, sy_, sz,
                                          base_freq=self.base_freq,
                                          octaves=self.octaves,
                                          lacunarity=self.lacunarity,
                                          gain=self.gain)
                out[v,u] = val
        return out


###############################################################################
# 3. Multi-agent environment that sums the agent images * weighted by agent action
###############################################################################
class MultiAgentFractalWaveEnv:
    def __init__(self, sim_params, action_ranges):
        """
        sim_params: dictionary of environment-level config
          e.g. {
            "num_agents": 3,
            "image_size": 50,
            "agent_init": {
               "pos_range": [-5,5],
               "pitch_range": [-0.3,0.3],
               "yaw_range": [0, 2*pi],
               ...
            },
            "fractal_init": {
               "base_freq_range": [0.1, 0.2],
               "octaves": 4,
               ...
            }
          }
        action_ranges: dictionary with min/max for each agent action
        """
        self.sim_params = sim_params
        self.action_ranges = action_ranges

        self.num_agents = sim_params.get("num_agents", 3)
        self.image_size = sim_params.get("image_size", 50)

        # Create the fractal object
        self.fractal_obj = DeterministicFractal3D(global_config=sim_params.get("fractal_global", None))

        # Initialize agents
        self.agents = []
        self.final_wave = np.zeros((self.image_size, self.image_size), dtype=float)
        self._init_agents()

    def _init_agents(self):
        import numpy as np
        import math
        
        agent_init = self.sim_params.get("agent_init", {})
        fractal_init = self.sim_params.get("fractal_init", {})
        
        for _ in range(self.num_agents):
            pos_lo, pos_hi = agent_init.get("pos_range", [-5,5])
            pitch_lo, pitch_hi = agent_init.get("pitch_range", [-0.3,0.3])
            yaw_lo, yaw_hi = agent_init.get("yaw_range", [0, 2*math.pi])
            fov_deg = agent_init.get("fov_deg", 60.0)
            
            base_freq_lo, base_freq_hi = fractal_init.get("base_freq_range", [0.1,0.2])
            lac_lo, lac_hi = fractal_init.get("lacunarity_range", [1.5,3.0])
            gain_lo, gain_hi = fractal_init.get("gain_range", [0.3,0.8])
            octaves = fractal_init.get("octaves", 4)
            blend_weight_lo, blend_weight_hi = fractal_init.get("blend_weight_range", [0.5,1.5])
            
            pos = np.random.uniform(pos_lo, pos_hi, size=3)
            pitch = np.random.uniform(pitch_lo, pitch_hi)
            yaw   = np.random.uniform(yaw_lo, yaw_hi)
            
            base_freq  = np.random.uniform(base_freq_lo, base_freq_hi)
            lacunarity = np.random.uniform(lac_lo, lac_hi)
            gain       = np.random.uniform(gain_lo, gain_hi)
            blend_w    = np.random.uniform(blend_weight_lo, blend_weight_hi)
            
            agent = FractalCameraAgent(
                fractal_obj=self.fractal_obj,
                pos3D=pos,
                pitch=pitch,
                yaw=yaw,
                fov_deg=fov_deg,
                base_freq=base_freq,
                octaves=octaves,
                lacunarity=lacunarity,
                gain=gain,
                blend_weight=blend_w,
                image_size=self.image_size
            )
            self.agents.append(agent)
    
    def step(self, actions_list, dt=0.1):
        """
        actions_list: list of dict, each dict has keys like 'dPos', 'dPitch', etc.
        We'll apply each agent's action, then combine images by sum of (img * blend_weight).
        """
        # 1) apply actions
        for i, agent in enumerate(self.agents):
            if i < len(actions_list):
                agent.apply_action(actions_list[i], dt)

        # 2) each agent renders NxN
        wave_sums   = np.zeros((self.image_size, self.image_size), dtype=float)
        weight_sums = np.zeros((self.image_size, self.image_size), dtype=float)

        for agent in self.agents:
            img = agent.render_fractal_image()  # in [-1..1]
            wave_sums   += (img * agent.blend_weight)
            weight_sums += agent.blend_weight

        # avoid dividing by zero
        weight_sums = np.where(weight_sums==0, 1e-6, weight_sums)
        self.final_wave = wave_sums / weight_sums
    
    def get_wave(self):
        return self.final_wave


###############################################################################
# 4. A mock main() with random actions + 3D matplotlib
###############################################################################
def main():
    import matplotlib.animation as animation

    # 4.1) The global parameter dictionary
    sim_params = {
        "num_agents": 10,
        "image_size": 50,

        "agent_init": {
            "pos_range": [-5,5],        # random agent pos in fractal space
            "pitch_range": [-0.3,0.3],
            "yaw_range": [0, 2*math.pi],
            "fov_deg": 60.0
        },
        "fractal_init": {
            "base_freq_range": [0.15, 0.2],
            "lacunarity_range": [2.0, 3.0],
            "gain_range": [0.5, 0.7],
            "octaves": 4,
            "blend_weight_range": [0.5, 1.5]
        },
        "fractal_global": {
            # Could store global seeds or constants if needed
        }
    }

    # 4.2) The agent action range dictionary (to randomize in the mock loop)
    action_ranges = {
        "dPosMin": -0.1,  "dPosMax": 0.1,
        "dPitchMin": -0.05,  "dPitchMax": 0.05,
        "dYawMin":   -0.05,  "dYawMax":   0.05,
        "dBaseFreqMin": -0.01, "dBaseFreqMax": 0.01,
        "dLacunarityMin": -0.01, "dLacunarityMax": 0.01,
        "dGainMin": -0.01, "dGainMax": 0.01,
        "dBlendWMin": -0.01, "dBlendWMax": 0.01
    }

    # 4.3) Create environment
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
        wave = env.get_wave()  # in [-1..1]
        Z = wave + 1.0
        ax.set_zlim(0,2)
        ax.set_title("Multi-Agent Fractal (Frame=0)")
        surf = ax.plot_surface(X_grid, Y_grid, Z, cmap='viridis', linewidth=0, antialiased=False)
        surface_holder[0] = surf
        return (surf,)

    def update_plot(frame):
        # remove old surface
        if surface_holder[0] is not None:
            surface_holder[0].remove()

        # Generate random actions
        acts = []
        for _ in range(env.num_agents):
            # e.g. random from the action_ranges
            # each agent can produce a dictionary with 'dPos', 'dPitch', etc.
            dPos = np.random.uniform(action_ranges["dPosMin"],
                                     action_ranges["dPosMax"],
                                     size=3)
            dPitch = np.random.uniform(action_ranges["dPitchMin"],
                                       action_ranges["dPitchMax"])
            dYaw   = np.random.uniform(action_ranges["dYawMin"],
                                       action_ranges["dYawMax"])
            dBaseFreq = np.random.uniform(action_ranges["dBaseFreqMin"],
                                          action_ranges["dBaseFreqMax"])
            dLacunarity = np.random.uniform(action_ranges["dLacunarityMin"],
                                            action_ranges["dLacunarityMax"])
            dGain = np.random.uniform(action_ranges["dGainMin"],
                                      action_ranges["dGainMax"])
            dBlendW = np.random.uniform(action_ranges["dBlendWMin"],
                                        action_ranges["dBlendWMax"])
            
            action = {
                "dPos": dPos,
                "dPitch": dPitch,
                "dYaw":   dYaw,
                "dBaseFreq": dBaseFreq,
                "dLacunarity": dLacunarity,
                "dGain": dGain,
                "dBlendWeight": dBlendW
            }
            acts.append(action)

        # step
        env.step(acts, dt=0.2)

        # re-plot
        wave = env.get_wave()  # [-1..1]
        Z = wave + 1.0         # [0..2]
        ax.clear()
        ax.set_zlim(0,2)
        ax.set_title(f"Multi-Agent Fractal (Frame={frame})")
        surf = ax.plot_surface(X_grid, Y_grid, Z, cmap='viridis', linewidth=0, antialiased=False)
        surface_holder[0] = surf
        return (surf,)

    ani = animation.FuncAnimation(fig, update_plot, frames=50, init_func=init_plot, 
                                  blit=False, interval=400)
    plt.show()

if __name__ == "__main__":
    main()
