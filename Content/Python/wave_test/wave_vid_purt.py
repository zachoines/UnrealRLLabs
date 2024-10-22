import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Grid size
grid_size = 40

# Create a meshgrid for the grid positions
x = np.linspace(0, grid_size - 1, grid_size)
y = np.linspace(0, grid_size - 1, grid_size)
X, Y = np.meshgrid(x, y)

# Number of agents
num_agents = 10

# Number of frames in the animation
num_frames = 50

# Define cardinal directions (N, NE, E, SE, S, SW, W, NW)
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

# Initialize agents with random parameters
agents = []
for _ in range(num_agents):
    agent = {
        'x_i': np.random.uniform(0, grid_size - 1),       # Agent's x position
        'y_i': np.random.uniform(0, grid_size - 1),       # Agent's y position
        'A_i': np.random.uniform(0.5, 1.5),               # Amplitude
        'f_i': np.random.uniform(0.05, 0.15),             # Frequency
        'phi_i': np.random.uniform(0, 2 * np.pi),         # Phase shift
        'theta_i': np.random.uniform(0, 2 * np.pi),       # Direction angle
        'sigma_i': np.random.uniform(5, 15)               # Spread
    }
    agents.append(agent)

# Function to update agents each frame
def update_agents(agents):
    for agent in agents:
        # Randomly perturb parameters by a small amount
        agent['A_i'] += np.random.uniform(-0.05, 0.05)
        agent['A_i'] = np.clip(agent['A_i'], 0.5, 1.5)

        agent['f_i'] += np.random.uniform(-0.005, 0.005)
        agent['f_i'] = np.clip(agent['f_i'], 0.05, 0.15)

        agent['phi_i'] += np.random.uniform(-0.1, 0.1)
        agent['phi_i'] = agent['phi_i'] % (2 * np.pi)

        agent['theta_i'] += np.random.uniform(-0.1, 0.1)
        agent['theta_i'] = agent['theta_i'] % (2 * np.pi)

        agent['sigma_i'] += np.random.uniform(-0.5, 0.5)
        agent['sigma_i'] = np.clip(agent['sigma_i'], 5, 15)

        # Move agent in a random cardinal direction
        dx, dy = direction_list[np.random.randint(0, 8)]
        agent['x_i'] += dx
        agent['y_i'] += dy

        # Keep agents within the grid boundaries
        agent['x_i'] = np.clip(agent['x_i'], 0, grid_size - 1)
        agent['y_i'] = np.clip(agent['y_i'], 0, grid_size - 1)

# Initialize the plot
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(np.zeros((grid_size, grid_size)), extent=(0, grid_size - 1, 0, grid_size - 1),
               origin='lower', cmap='viridis', animated=True)
plt.colorbar(im, ax=ax, label='Column Height')
ax.set_title('Interference Pattern of Agents\' Waves')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')

# Function to update the plot for each frame
def animate(frame):
    # Update agents' parameters and positions
    update_agents(agents)

    # Initialize the total wave pattern
    W_total = np.zeros((grid_size, grid_size))

    # Calculate the wave function for each agent and sum them up
    for agent in agents:
        x_i = agent['x_i']
        y_i = agent['y_i']
        A_i = agent['A_i']
        f_i = agent['f_i']
        phi_i = agent['phi_i']
        theta_i = agent['theta_i']
        sigma_i = agent['sigma_i']

        # Calculate the positional offsets
        X_shifted = X - x_i
        Y_shifted = Y - y_i

        # Project positions onto the wave's direction
        R = X_shifted * np.cos(theta_i) + Y_shifted * np.sin(theta_i)

        # Calculate the distance squared from the agent's position
        distance_squared = X_shifted**2 + Y_shifted**2

        # Compute the wave function
        W_i = A_i * np.cos(2 * np.pi * f_i * R + phi_i) * np.exp(-distance_squared / (2 * sigma_i**2))

        # Add the agent's wave to the total wave pattern
        W_total += W_i

    # Update the plot
    im.set_array(W_total)

    # Optionally, plot the agents' positions
    ax.clear()
    ax.imshow(W_total, extent=(0, grid_size - 1, 0, grid_size - 1),
              origin='lower', cmap='viridis', animated=True)
    ax.set_title('Interference Pattern of Agents\' Waves (Frame {})'.format(frame))
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')

    # Plot agents' positions
    for agent in agents:
        ax.plot(agent['x_i'], agent['y_i'], 'ro')

    return [im]

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=200, blit=False)

# Display the animation
plt.show()
