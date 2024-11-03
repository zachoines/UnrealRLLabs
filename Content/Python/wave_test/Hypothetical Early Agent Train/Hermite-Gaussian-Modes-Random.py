import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import hermite

# Simulation Parameters
GRID_SIZE = 100     # Size of the grid (100x100)
N_AGENTS = 20       # Number of agents
K_FRAMES = 1000     # Total number of frames in the simulation
FRAME_SKIP = 1      # Frame skip for visualization
H0 = 0              # Base height of the grid

# Hermite-Gaussian Modes Parameters
DECAY_FACTOR = 0.1    # Decay factor for the columns (0 < DECAY_FACTOR <= 1)
V_MAX = 1.0            # Maximum speed of column height change per frame
SIGMA = 5.0            # Spread of the Gaussian envelope
UPDATE_INTERVAL = 5   # Number of frames after which to regenerate agent parameters

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

# Initialize grid heights
heights = np.zeros((GRID_SIZE, GRID_SIZE))

# Initialize wave parameters (set to None initially)
agent_amplitudes = None
agent_n_orders = None
agent_m_orders = None
agent_frequencies = None
agent_phases = None

# Prepare for animation
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Function to compute Hermite polynomials
def hermite_polynomial(n, x):
    Hn = hermite(n)
    return Hn(x)

# Function to initialize or update agent wave parameters
def regenerate_agent_parameters():
    global agent_amplitudes, agent_n_orders, agent_m_orders, agent_frequencies, agent_phases
    agent_amplitudes = np.random.uniform(5.0, 10.0, N_AGENTS)
    agent_n_orders = np.random.randint(0, 3, N_AGENTS)  # Orders 0 to 2
    agent_m_orders = np.random.randint(0, 3, N_AGENTS)
    agent_frequencies = np.random.uniform(0.1, 0.5, N_AGENTS)
    agent_phases = np.random.uniform(0, 2 * np.pi, N_AGENTS)

# Function to update the plot
def update_plot(frame_number):
    global agent_positions, agent_directions, agent_speeds, agent_amplitudes, agent_n_orders, agent_m_orders, agent_frequencies, agent_phases, heights

    # Regenerate agent wave parameters every UPDATE_INTERVAL frames
    if frame_number % UPDATE_INTERVAL == 0 or frame_number == 0:
        regenerate_agent_parameters()

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

    for i in range(N_AGENTS):
        x_a, y_a = agent_positions[i]
        A_a = agent_amplitudes[i]
        n_a = agent_n_orders[i]
        m_a = agent_m_orders[i]
        omega_a = agent_frequencies[i]
        phi_a = agent_phases[i]
        sigma = SIGMA  # Can vary per agent if desired

        # Shift coordinates relative to agent position
        X_shifted = X - x_a
        Y_shifted = Y - y_a

        # Normalize coordinates by sigma
        X_norm = X_shifted / sigma
        Y_norm = Y_shifted / sigma

        # Compute Hermite polynomials
        Hn = hermite_polynomial(n_a, X_norm)
        Hm = hermite_polynomial(m_a, Y_norm)

        # Compute Gaussian envelope
        envelope = np.exp(- (X_norm ** 2 + Y_norm ** 2) / 2)

        # Compute the Hermite-Gaussian mode
        wave = A_a * Hn * Hm * envelope * np.cos(omega_a * time + phi_a)

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
