import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -----------------------------
# Configuration Parameters
# -----------------------------

# Grid size
grid_size = 100

# Create a meshgrid for the grid positions
x = np.linspace(0, grid_size - 1, grid_size)
y = np.linspace(0, grid_size - 1, grid_size)
X, Y = np.meshgrid(x, y)

# Number of agents
num_agents = 10

# Number of frames in the animation
num_frames = 200

# Frame skip value (agents update parameters every 'frame_skip' frames)
frame_skip = 10

# Column height constraints
min_height = -1.0
max_height = 1.0

# Define parameter ranges
A_min, A_max = 0.5, 1.5              # Amplitude range
f_min, f_max = 0.05, 0.15            # Frequency range
phi_min, phi_max = 0.0, 2 * np.pi    # Phase shift range
theta_min, theta_max = 0.0, 2 * np.pi  # Wave propagation direction range
sigma_r_min, sigma_r_max = 10.0, 30.0  # Radial decay parameter range
v_wave_min, v_wave_max = 0.5, 2.0    # Wave speed range

v_move_min, v_move_max = 0.5, 2.0    # Movement speed range

# Define cardinal directions (optional for movement)
directions = {
    'N': (0, 1),
    'NE': (1, 1),
    'E': (1, 0),
    'SE': (1, -1),
    'S': (0, -1),
    'SW': (-1, -1),
    'W': (-1, 0),
    'NW': (-1, 1)
}
direction_list = list(directions.values())

# -----------------------------
# Agent Initialization
# -----------------------------

# Initialize agents with random parameters
agents = []
for _ in range(num_agents):
    agent = {
        # Current position (floating-point for continuous movement)
        'x_i': np.random.uniform(0, grid_size - 1),
        'y_i': np.random.uniform(0, grid_size - 1),
        # Movement parameters
        'v_i': np.random.uniform(v_move_min, v_move_max),            # Velocity
        'theta_move_i': np.random.uniform(0, 2 * np.pi),              # Movement direction
        # Wave parameters
        'A_i': np.random.uniform(A_min, A_max),                      # Amplitude
        'f_i': np.random.uniform(f_min, f_max),                      # Frequency
        'phi_i': np.random.uniform(phi_min, phi_max),                # Phase shift
        'theta_wave_i': np.random.uniform(theta_min, theta_max),      # Wave propagation direction
        'sigma_r_i': np.random.uniform(sigma_r_min, sigma_r_max),    # Radial decay
        'v_wave_i': np.random.uniform(v_wave_min, v_wave_max),        # Wave speed
    }
    agents.append(agent)

# -----------------------------
# Plot Initialization
# -----------------------------

fig, ax = plt.subplots(figsize=(8, 6))
W_initial = np.zeros((grid_size, grid_size))
im = ax.imshow(W_initial, extent=(0, grid_size - 1, 0, grid_size - 1),
               origin='lower', cmap='viridis', animated=True, vmin=min_height, vmax=max_height)
plt.colorbar(im, ax=ax, label='Column Height')
ax.set_title('Interference Pattern of Agents\' Plane Waves')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')

# Initialize the scatter plot for agents' positions
x_positions = [agent['x_i'] for agent in agents]
y_positions = [agent['y_i'] for agent in agents]
agent_scatter = ax.scatter(x_positions, y_positions, c='red', marker='o', label='Agents')

# Optional: Add a legend
ax.legend(loc='upper right')

# -----------------------------
# Animation Function
# -----------------------------

def animate(frame):
    global agents  # To modify agents within the function

    # Initialize the total wave pattern for this frame
    W_total = np.zeros((grid_size, grid_size))

    for agent in agents:
        # Update agent's position based on movement parameters
        agent['x_i'] += agent['v_i'] * np.cos(agent['theta_move_i'])
        agent['y_i'] += agent['v_i'] * np.sin(agent['theta_move_i'])

        # Keep agents within the grid boundaries using modulo for wrapping
        agent['x_i'] = agent['x_i'] % grid_size
        agent['y_i'] = agent['y_i'] % grid_size

        # Extract wave parameters
        x_i = agent['x_i']
        y_i = agent['y_i']
        A_i = agent['A_i']
        f_i = agent['f_i']
        phi_i = agent['phi_i']
        theta_i = agent['theta_wave_i']
        sigma_r_i = agent['sigma_r_i']
        v_wave_i = agent['v_wave_i']

        # Calculate positional differences
        delta_x = X - x_i
        delta_y = Y - y_i

        # Calculate radial distance from agent to each point
        r_i = np.sqrt(delta_x**2 + delta_y**2)

        # Compute the plane wave function
        # Plane Wave: W = A * cos(k * (x cos(theta) + y sin(theta) - v * t) + phi) * e^(-r^2 / (2 sigma^2))
        # Here, k = 2pi * f
        k = 2 * np.pi * f_i
        argument = k * (delta_x * np.cos(theta_i) + delta_y * np.sin(theta_i) - v_wave_i * frame)
        W_i = A_i * np.cos(argument + phi_i) * np.exp(-r_i**2 / (2 * sigma_r_i**2))

        # Add the agent's wave to the total wave pattern
        W_total += W_i

    # Normalize and clip the total wave pattern
    W_total_clipped = np.clip(W_total, min_height, max_height)

    # Update the image data
    im.set_array(W_total_clipped)

    # Update agents' positions on the scatter plot
    x_positions = [agent['x_i'] for agent in agents]
    y_positions = [agent['y_i'] for agent in agents]
    agent_scatter.set_offsets(np.c_[x_positions, y_positions])

    # Update plot title with current frame number
    ax.set_title(f'Interference Pattern of Agents\' Plane Waves (Frame {frame})')

    # Update parameters every 'frame_skip' frames
    if frame % frame_skip == 0:
        for agent in agents:
            # Update movement parameters
            agent['v_i'] = np.random.uniform(v_move_min, v_move_max)
            agent['theta_move_i'] = np.random.uniform(0, 2 * np.pi)

            # Update wave parameters
            agent['A_i'] = np.random.uniform(A_min, A_max)
            agent['f_i'] = np.random.uniform(f_min, f_max)
            agent['phi_i'] = np.random.uniform(phi_min, phi_max)
            agent['theta_wave_i'] = np.random.uniform(theta_min, theta_max)
            agent['sigma_r_i'] = np.random.uniform(sigma_r_min, sigma_r_max)
            agent['v_wave_i'] = np.random.uniform(v_wave_min, v_wave_max)

    return [im, agent_scatter]

# -----------------------------
# Create and Display Animation
# -----------------------------

# Create the animation using FuncAnimation
ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=200, blit=True)

# Display the animation
plt.show()
