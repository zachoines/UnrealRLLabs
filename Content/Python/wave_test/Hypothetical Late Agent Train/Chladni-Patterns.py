import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation Parameters
GRID_SIZE = 100     # Size of the grid (100x100)
N_AGENTS = 10       # Number of agents (components of the Chladni pattern)
K_FRAMES = 1000     # Total number of frames in the simulation
FRAME_SKIP = 1      # Frame skip for visualization
L = GRID_SIZE       # Length of the grid (assumed square)

# Chladni Pattern Parameters
DECAY_FACTOR = 0.01  # Decay factor for the vibration amplitude
V_MAX = 1.0          # Maximum change in vibration per frame

# Create meshgrid for grid coordinates
x = np.linspace(0, L, GRID_SIZE)
y = np.linspace(0, L, GRID_SIZE)
X, Y = np.meshgrid(x, y)

# User-defined mode ranges for x and y directions
min_mode_x = 1
max_mode_x = 10
min_mode_y = 1
max_mode_y = 10

# Initialize agent mode numbers (m, n) within the specified ranges
mode_numbers_x = np.random.uniform(min_mode_x, max_mode_x + 1, N_AGENTS)  # m_i
mode_numbers_y = np.random.uniform(min_mode_y, max_mode_y + 1, N_AGENTS)  # n_i

# Initialize agent amplitudes
agent_amplitudes = np.random.uniform(5.0, 10.0, N_AGENTS)

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
    global mode_numbers_x, mode_numbers_y, agent_amplitudes, agent_frequencies, agent_phases, heights

    time = frame_number * FRAME_SKIP

    # Apply decay to current heights
    heights *= DECAY_FACTOR

    # Compute total desired change in height (delta_h_desired)
    delta_h_desired = np.zeros((GRID_SIZE, GRID_SIZE))

    # Iterate over each agent/component
    for a in range(N_AGENTS):
        m_a = mode_numbers_x[a]
        n_a = mode_numbers_y[a]
        A_a = agent_amplitudes[a]
        omega_a = agent_frequencies[a]
        phi_a = agent_phases[a]

        # Compute the Chladni wave component
        wave = A_a * np.sin(m_a * np.pi * X / L) * np.sin(n_a * np.pi * Y / L) * np.cos(omega_a * time + phi_a)

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
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_zlim(-20, 20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')
    ax.set_title(f"Chladni Pattern - Frame {frame_number * FRAME_SKIP}")

    # Adjust viewing angle for better visualization
    ax.view_init(elev=30, azim=60)

    return surf,

# Create the animation
ani = animation.FuncAnimation(fig, update_plot, frames=K_FRAMES // FRAME_SKIP, interval=50, blit=False)

# Display the animation
plt.show()
