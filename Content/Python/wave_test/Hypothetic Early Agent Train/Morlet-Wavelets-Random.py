import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation Parameters
GRID_SIZE = 100     # Size of the grid (100x100)
N_AGENTS = 10       # Number of agents
K_FRAMES = 1000     # Total number of frames in the simulation
FRAME_SKIP = 1      # Frame skip for visualization
H0 = 0              # Base height of the grid

# Morlet Wavelet Parameters
DECAY_FACTOR = 0.1  # Decay factor for the columns (0 < DECAY_FACTOR <= 1)
V_MAX = 1.0          # Maximum speed of column height change per frame
SIGMA_MIN = 2.0      # Minimum spread of the Gaussian envelope
SIGMA_MAX = 10.0     # Maximum spread of the Gaussian envelope

# Create meshgrid for grid coordinates
x = np.linspace(0, GRID_SIZE, GRID_SIZE)
y = np.linspace(0, GRID_SIZE, GRID_SIZE)
X, Y = np.meshgrid(x, y)

# Initialize agent positions randomly within the grid
agent_positions = np.random.uniform(0, GRID_SIZE, (N_AGENTS, 2))

# Initialize agent velocities (vx, vy)
agent_velocities = np.zeros((N_AGENTS, 2))

# Initialize agent wave parameters
agent_amplitudes = np.random.uniform(5.0, 10.0, N_AGENTS)
agent_wave_orientations = np.random.uniform(0, 2 * np.pi, N_AGENTS)
agent_wavenumbers = np.random.uniform(0.2, 0.5, N_AGENTS)
phase_velocity = 1.0
agent_frequencies = agent_wavenumbers * phase_velocity
agent_phases = np.random.uniform(0, 2 * np.pi, N_AGENTS)
agent_sigmas = np.random.uniform(SIGMA_MIN, SIGMA_MAX, N_AGENTS)

# Initialize grid heights
heights = np.zeros((GRID_SIZE, GRID_SIZE))

# Prepare for animation
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Function to update the plot
def update_plot(frame_number):
    global agent_positions, agent_velocities
    global agent_amplitudes, agent_wave_orientations, agent_wavenumbers, agent_frequencies, agent_phases, agent_sigmas
    global heights

    time = frame_number * FRAME_SKIP

    # Generate random actions for agents
    # Actions: [delta_vx, delta_vy, delta_amplitude, delta_wave_orientation, delta_wavenumber, delta_phase, delta_sigma]
    action_range = {
        'delta_v': (-0.5, 0.5),  # Change in velocity components
        'delta_amplitude': (-1.0, 1.0),
        'delta_wave_orientation': (-np.pi/8, np.pi/8),
        'delta_wavenumber': (-0.05, 0.05),
        'delta_phase': (-np.pi/4, np.pi/4),
        'delta_sigma': (-0.5, 0.5),
    }

    # For simplicity, generate random actions
    delta_vx = np.random.uniform(*action_range['delta_v'], N_AGENTS)
    delta_vy = np.random.uniform(*action_range['delta_v'], N_AGENTS)
    delta_amplitude = np.random.uniform(*action_range['delta_amplitude'], N_AGENTS)
    delta_wave_orientation = np.random.uniform(*action_range['delta_wave_orientation'], N_AGENTS)
    delta_wavenumber = np.random.uniform(*action_range['delta_wavenumber'], N_AGENTS)
    delta_phase = np.random.uniform(*action_range['delta_phase'], N_AGENTS)
    delta_sigma = np.random.uniform(*action_range['delta_sigma'], N_AGENTS)

    # Update agent velocities
    agent_velocities[:, 0] += delta_vx
    agent_velocities[:, 1] += delta_vy

    # Limit agent velocities to a maximum speed
    max_speed = 3.0
    speeds = np.linalg.norm(agent_velocities, axis=1)
    speed_factors = np.ones(N_AGENTS)
    speed_exceeds = speeds > max_speed
    speed_factors[speed_exceeds] = max_speed / speeds[speed_exceeds]
    agent_velocities *= speed_factors[:, np.newaxis]

    # Update agent positions
    agent_positions += agent_velocities * FRAME_SKIP

    # Keep agents within the grid boundaries
    agent_positions = np.mod(agent_positions, GRID_SIZE)

    # Update agent wave parameters
    agent_amplitudes += delta_amplitude
    agent_wave_orientations += delta_wave_orientation
    agent_wavenumbers += delta_wavenumber
    agent_phases += delta_phase
    agent_sigmas += delta_sigma

    # Keep wave parameters within reasonable ranges
    agent_amplitudes = np.clip(agent_amplitudes, 1.0, 15.0)
    agent_wave_orientations = np.mod(agent_wave_orientations, 2 * np.pi)
    agent_wavenumbers = np.clip(agent_wavenumbers, 0.1, 1.0)
    agent_phases = np.mod(agent_phases, 2 * np.pi)
    agent_sigmas = np.clip(agent_sigmas, SIGMA_MIN, SIGMA_MAX)

    # Recalculate agent frequencies based on new wavenumbers
    agent_frequencies = agent_wavenumbers * phase_velocity

    # Apply decay to current heights
    heights *= DECAY_FACTOR

    # Compute total desired change in height (delta_h_desired)
    delta_h_desired = np.zeros((GRID_SIZE, GRID_SIZE))

    for a in range(N_AGENTS):
        x_a, y_a = agent_positions[a]
        A_a = agent_amplitudes[a]
        theta_a = agent_wave_orientations[a]
        k_a = agent_wavenumbers[a]
        omega_a = agent_frequencies[a]
        phi_a = agent_phases[a]
        sigma_a = agent_sigmas[a]

        # Shift coordinates relative to agent position
        X_shifted = X - x_a
        Y_shifted = Y - y_a

        # Rotate coordinates based on wave orientation
        x_rot = X_shifted * np.cos(theta_a) + Y_shifted * np.sin(theta_a)
        y_rot = -X_shifted * np.sin(theta_a) + Y_shifted * np.cos(theta_a)

        # Compute Gaussian envelope with agent-specific sigma
        envelope = np.exp(- (x_rot ** 2 + y_rot ** 2) / (2 * sigma_a ** 2))

        # Compute the Morlet wavelet
        wave = A_a * envelope * np.cos(k_a * x_rot - omega_a * time + phi_a)

        # Accumulate the desired change in height
        delta_h_desired += wave

    # Limit the change in height to maximum speed V_MAX
    delta_h_actual = np.clip(delta_h_desired, -V_MAX, V_MAX)

    # Update the heights with the limited change
    heights += delta_h_actual

    # Clear the axis
    ax.clear()

    # Plot the surface
    surf = ax.plot_surface(X, Y, heights, cmap='viridis', edgecolor='none', rstride=1, cstride=1, linewidth=0, antialiased=False)

    # Set plot limits
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_zlim(-20, 20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')
    ax.set_title(f"Frame {frame_number * FRAME_SKIP}")

    # Adjust viewing angle for better visualization
    ax.view_init(elev=30, azim=60)

    return surf,

# Create the animation
ani = animation.FuncAnimation(fig, update_plot, frames=K_FRAMES // FRAME_SKIP, interval=50, blit=False)

# Display the animation
plt.show()
