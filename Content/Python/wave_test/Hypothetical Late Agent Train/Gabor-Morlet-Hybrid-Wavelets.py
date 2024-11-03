import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation Parameters
GRID_SIZE = 100     # Size of the grid (100x100)
N_AGENTS = 10       # Number of agents
K_FRAMES = 1000     # Total number of frames in the simulation
FRAME_SKIP = 5      # Frame skip for visualization
H0 = 0              # Base height of the grid

# Gabor-Morlet Hybrid Wavelet Parameters
DECAY_FACTOR = 0.95  # Decay factor for the columns (0 < DECAY_FACTOR <= 1)
V_MAX = 1.0          # Maximum speed of column height change per frame
SIGMA = 5.0          # Spread of the Gaussian envelope

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

# Initialize agent wave propagation orientations
agent_orientations = np.random.uniform(0, 2 * np.pi, N_AGENTS)

# Initialize wavenumbers (k_x, k_y) for Gabor-Morlet oscillations
agent_wavenumbers_x = np.random.uniform(0.1, 0.4, N_AGENTS)
agent_wavenumbers_y = np.random.uniform(0.1, 0.4, N_AGENTS)

# Initialize angular frequencies
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
    global agent_positions, agent_directions, agent_speeds, agent_amplitudes, agent_orientations, agent_wavenumbers_x, agent_wavenumbers_y, agent_frequencies, agent_phases, heights

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
        theta_a = agent_orientations[a]
        k_x = agent_wavenumbers_x[a]
        k_y = agent_wavenumbers_y[a]
        omega_a = agent_frequencies[a]
        phi_a = agent_phases[a]
        sigma = SIGMA  # Can vary per agent if desired

        # Shift coordinates relative to agent position
        X_shifted = X - x_a
        Y_shifted = Y - y_a

        # Rotate coordinates based on wave orientation
        x_rot = X_shifted * np.cos(theta_a) + Y_shifted * np.sin(theta_a)
        y_rot = -X_shifted * np.sin(theta_a) + Y_shifted * np.cos(theta_a)

        # Compute Gaussian envelope
        envelope = np.exp(- (x_rot ** 2 + y_rot ** 2) / (2 * sigma ** 2))

        # Compute the Gabor-Morlet Hybrid Wavelet
        wave = A_a * envelope * np.cos(k_x * x_rot + k_y * y_rot - omega_a * time + phi_a)

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
