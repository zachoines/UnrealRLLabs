import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation Parameters
GRID_SIZE = 100     # Size of the grid (100x100)
N_AGENTS = 10       # Number of agents
K_FRAMES = 1000     # Total number of frames in the simulation
FRAME_SKIP = 1      # Frame skip for visualization
H0 = 0              # Base height of the grid

# Wave Packet Parameters
DECAY_FACTOR = 0.01  # Decay factor for the columns (0 < DECAY_FACTOR <= 1)
V_MAX = 1.0          # Maximum speed of column height change per frame
SIGMA = 10.0          # Spread of the Gaussian envelope

# Create meshgrid for grid coordinates
x = np.linspace(0, GRID_SIZE, GRID_SIZE)
y = np.linspace(0, GRID_SIZE, GRID_SIZE)
X, Y = np.meshgrid(x, y)

# Initialize agent positions randomly within the grid
agent_positions = np.random.uniform(0, GRID_SIZE, (N_AGENTS, 2))

# Initialize agent directions (angles in radians) for movement
agent_directions = np.random.uniform(0, 2 * np.pi, N_AGENTS)

# Initialize agent speeds for movement
agent_speeds = np.random.uniform(1.0, 3.0, N_AGENTS)

# Initialize agent amplitudes
agent_amplitudes = np.random.uniform(5.0, 10.0, N_AGENTS)

# Initialize agent group velocities (speeds of the envelope)
agent_group_speeds = np.random.uniform(1.0, 2.0, N_AGENTS)
agent_group_directions = np.random.uniform(0, 2 * np.pi, N_AGENTS)
v_gx = agent_group_speeds * np.cos(agent_group_directions)
v_gy = agent_group_speeds * np.sin(agent_group_directions)

# Initialize agent wavenumbers (controls frequency of carrier wave)
agent_wavenumbers_x = np.random.uniform(0.2, 0.5, N_AGENTS)
agent_wavenumbers_y = np.random.uniform(0.2, 0.5, N_AGENTS)

# Initialize agent angular frequencies
agent_frequencies = np.random.uniform(0.05, 0.2, N_AGENTS)

# Initialize agent phases
agent_phases = np.random.uniform(0, 2 * np.pi, N_AGENTS)

# Initialize grid heights
heights = np.zeros((GRID_SIZE, GRID_SIZE))

# Prepare for animation
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Function to update the plot
def update_plot(frame_number):
    global agent_positions, agent_directions, agent_speeds, agent_amplitudes, agent_wavenumbers_x, agent_wavenumbers_y, agent_frequencies, agent_phases, v_gx, v_gy, heights

    time = frame_number * FRAME_SKIP

    # Update agent positions
    delta_positions = np.stack((agent_speeds * np.cos(agent_directions),
                                agent_speeds * np.sin(agent_directions)), axis=1)
    agent_positions += delta_positions * FRAME_SKIP  # Multiply by FRAME_SKIP to match visualization frames

    # Keep agents within the grid boundaries
    agent_positions = np.mod(agent_positions, GRID_SIZE)

    # Apply decay to current heights
    heights *= DECAY_FACTOR

    # Compute total desired change in height (delta_h_desired)
    delta_h_desired = np.zeros((GRID_SIZE, GRID_SIZE))

    for a in range(N_AGENTS):
        x_a, y_a = agent_positions[a]
        A_a = agent_amplitudes[a]
        kx_a = agent_wavenumbers_x[a]
        ky_a = agent_wavenumbers_y[a]
        omega_a = agent_frequencies[a]
        phi_a = agent_phases[a]
        sigma = SIGMA  # Can vary per agent if desired
        vg_x = v_gx[a]
        vg_y = v_gy[a]

        # Compute the displacement of the envelope
        x_env = X - x_a - vg_x * time
        y_env = Y - y_a - vg_y * time

        # Compute Gaussian envelope
        envelope = np.exp(- (x_env ** 2 + y_env ** 2) / (2 * sigma ** 2))

        # Compute the carrier wave
        x_carrier = X - x_a - vg_x * time
        y_carrier = Y - y_a - vg_y * time
        phase = kx_a * x_carrier + ky_a * y_carrier - omega_a * time + phi_a

        # Compute the wave packet
        wave = A_a * envelope * np.cos(phase)

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
