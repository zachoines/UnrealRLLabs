import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation Parameters
GRID_SIZE = 100      # Size of the grid (100x100)
N_AGENTS = 5         # Number of agents (masses)
K_FRAMES = 1000      # Total number of frames in the simulation
FRAME_SKIP = 1       # Frame skip for visualization
G = 1.0              # Gravitational constant (scaled)
EPSILON = 1e-3       # Small constant to avoid division by zero
SIGMA = 5.0          # Spread of the gravitational wave
DECAY_FACTOR = 0.001  # Decay factor for grid heights

# Create meshgrid for grid coordinates
x = np.linspace(0, GRID_SIZE, GRID_SIZE)
y = np.linspace(0, GRID_SIZE, GRID_SIZE)
X, Y = np.meshgrid(x, y)

# Initialize agent masses, positions, velocities, and wave parameters
agent_masses = np.random.uniform(-1.0, 1.0, N_AGENTS)
agent_positions = np.random.uniform(0, GRID_SIZE, (N_AGENTS, 2))
agent_velocities = np.random.uniform(-0.5, 0.5, (N_AGENTS, 2))
agent_wavenumbers = np.random.uniform(0.2, 0.5, N_AGENTS)
agent_frequencies = np.random.uniform(0.05, 0.2, N_AGENTS)
agent_phases = np.random.uniform(0, 2 * np.pi, N_AGENTS)

# Initialize grid heights
heights = np.zeros((GRID_SIZE, GRID_SIZE))

# Prepare for animation
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Function to update the plot
def update_plot(frame_number):
    global agent_positions, agent_velocities, heights

    time = frame_number * FRAME_SKIP

    # Reset grid heights for this frame
    delta_h_desired = np.zeros((GRID_SIZE, GRID_SIZE))

    # Update agent positions and velocities due to gravitational forces
    for a in range(N_AGENTS):
        x_a, y_a = agent_positions[a]
        m_a = agent_masses[a]

        # Calculate gravitational forces from other agents
        force = np.zeros(2)
        for b in range(N_AGENTS):
            if a != b:
                x_b, y_b = agent_positions[b]
                m_b = agent_masses[b]

                # Calculate distance and force components
                dx = x_b - x_a
                dy = y_b - y_a
                r = np.sqrt(dx**2 + dy**2 + EPSILON)

                # Calculate gravitational force
                F_mag = G * m_a * m_b / (r**2 + EPSILON)
                F_dir = np.array([dx, dy]) / (r + EPSILON)

                # Accumulate forces
                force += F_mag * F_dir

        # Update velocities and positions based on forces
        agent_velocities[a] += force / m_a * FRAME_SKIP
        agent_positions[a] += agent_velocities[a] * FRAME_SKIP

        # Keep agents within the grid boundaries
        agent_positions[a] = np.mod(agent_positions[a], GRID_SIZE)

    # Calculate gravitational waves and update grid heights
    for a in range(N_AGENTS):
        x_a, y_a = agent_positions[a]
        A_a = agent_masses[a]  # Amplitude proportional to mass
        k_a = agent_wavenumbers[a]
        omega_a = agent_frequencies[a]
        phi_a = agent_phases[a]

        # Compute radial distance from agent to each grid point
        r = np.sqrt((X - x_a)**2 + (Y - y_a)**2 + EPSILON)

        # Calculate wave contribution from the agent
        wave = A_a * np.cos(k_a * r - omega_a * time + phi_a) / (np.sqrt(r**2 + EPSILON))
        wave *= np.exp(-r**2 / (2 * SIGMA**2))

        # Accumulate the wave contributions
        delta_h_desired += wave

    # Apply decay to current heights
    heights *= DECAY_FACTOR

    # Update the grid heights with the new contributions
    heights += delta_h_desired

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
