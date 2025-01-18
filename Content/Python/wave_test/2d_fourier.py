import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# ------------------------------------------------------------------
#  Environment Initialization
# ------------------------------------------------------------------

def init_env(numAgents=3, N=80, K=5):
    """
    Create an 'env' dict:
      - N, K, dim = 2K
      - Each agent has:
          agentA: (dim x dim) local matrix
          row, col: current location in that matrix
      - We remove phase/freq code entirely.
      - X_mesh, Y_mesh for plotting.
    """
    env = {}
    env["numAgents"] = numAgents
    env["N"] = N
    env["K"] = K
    env["dim"] = 2 * K

    agents = []
    dim = env["dim"]
    for i in range(numAgents):
        agent_info = {
            "agentID": i,
            "agentA": np.zeros((dim, dim)),  # local matrix
            # A random location in [0..dim-1] x [0..dim-1]
            "row": np.random.randint(0, dim),
            "col": np.random.randint(0, dim)
        }
        agents.append(agent_info)

    env["agents"] = agents

    # For plotting
    x_vals = np.linspace(0, 2*np.pi, N, endpoint=False)
    y_vals = np.linspace(0, 2*np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x_vals, y_vals)
    env["X_mesh"] = X
    env["Y_mesh"] = Y

    return env

# ------------------------------------------------------------------
#  Standard Separable 2D Fourier Basis (with no agent-specific phase)
# ------------------------------------------------------------------

def generate_basis(N, K):
    """
    Build Sx, Sy each shape (N, 2K).
    We remove phase offset and freq scale. The wave index is just m in [0..K-1].
    xLine = np.linspace(0, 2π, N)
    yLine = np.linspace(0, 2π, N)

    Sx[:, m] = cos(m * xLine),   Sx[:, K+m] = sin(m * xLine)
    Sy[:, m] = cos(m * yLine),   Sy[:, K+m] = sin(m * yLine)
    """
    xLine = np.linspace(0, 2*np.pi, N, endpoint=False)
    yLine = np.linspace(0, 2*np.pi, N, endpoint=False)

    dim = 2*K
    Sx = np.zeros((N, dim))
    Sy = np.zeros((N, dim))

    for m in range(K):
        Sx[:, m]     = np.cos(m * xLine)
        Sx[:, K + m] = np.sin(m * xLine)

        Sy[:, m]     = np.cos(m * yLine)
        Sy[:, K + m] = np.sin(m * yLine)

    return Sx, Sy

def compute_heightmap(Sx, A, Sy):
    """
    G = Sx @ A @ Sy^T, shape (N, N).
    """
    return Sx @ A @ Sy.T

# ------------------------------------------------------------------
#  getAgentState(agentID, t, env)
# ------------------------------------------------------------------

def getAgentState(agentID, t, env):
    """
    Return a dict with agent's location (row, col) and maybe the entire A if needed.
    But let's keep it minimal: row, col only, plus maybe t.
    """
    agent = env["agents"][agentID]
    return {
        "agentID": agentID,
        "time":    t,
        "row":     agent["row"],
        "col":     agent["col"]
        # If you want the entire agentA in state, you can flatten it,
        # but that might be huge. We'll keep it out for demonstration.
    }

# ------------------------------------------------------------------
#  state(env, t)
# ------------------------------------------------------------------

def state(env, t):
    """
    Return list of states for all agents (each is a dict).
    """
    numAgents = env["numAgents"]
    allStates = []
    for i in range(numAgents):
        st = getAgentState(i, t, env)
        allStates.append(st)
    return allStates

# ------------------------------------------------------------------
#  getAgentActions(agentID, agentState, env)
# ------------------------------------------------------------------

def getAgentActions(agentID, agentState, env):
    """
    Each agent picks:
      1) A direction among {up, down, left, right}, with wrapping
      2) A partial update among {increment, decrement, zero, none} for the A[row,col] entry.
    """
    directions = ["up", "down", "left", "right"]
    updates    = ["inc", "dec", "zero", "none"]

    # pick direction randomly
    dir_choice = np.random.choice(directions)
    # pick update randomly
    upd_choice = np.random.choice(updates)

    action = {
        "agentID": agentID,
        "direction": dir_choice,
        "partialUpdate": upd_choice
    }
    return action

# ------------------------------------------------------------------
#  actions(env, allAgentStates)
# ------------------------------------------------------------------

def actions(env, allAgentStates):
    """
    Return a list of actions for each agent.
    """
    allActions = []
    for st in allAgentStates:
        agentID = st["agentID"]
        act = getAgentActions(agentID, st, env)
        allActions.append(act)
    return allActions

# ------------------------------------------------------------------
#  Update(agentStates, agentActions, t, env)
# ------------------------------------------------------------------

def Update(agentStates, agentActions, t, env):
    """
    Steps:
      1) For each agent's (direction, partialUpdate), we move agent location with wrap,
         then apply partialUpdate to that location in agentA.
      2) Sum all agent waves: G_total = Σ( Sx * agentA * Sy^T )
      3) clamp to [-1,1]
    """
    agents = env["agents"]
    N = env["N"]
    K = env["K"]
    dim = env["dim"]

    # Pre-build the basis Sx, Sy once, since there's no per-agent phase/freq
    Sx, Sy = generate_basis(N, K)

    # 1) Update each agent's location + partial update
    for action in agentActions:
        i = action["agentID"]
        agent = agents[i]

        # Move agent's (row,col)
        if action["direction"] == "up":
            agent["row"] = (agent["row"] - 1) % dim
        elif action["direction"] == "down":
            agent["row"] = (agent["row"] + 1) % dim
        elif action["direction"] == "left":
            agent["col"] = (agent["col"] - 1) % dim
        elif action["direction"] == "right":
            agent["col"] = (agent["col"] + 1) % dim

        # partial update at agent["row"], agent["col"]
        r, c = agent["row"], agent["col"]
        if action["partialUpdate"] == "inc":
            agent["agentA"][r,c] += 0.05
        elif action["partialUpdate"] == "dec":
            agent["agentA"][r,c] -= 0.05
        elif action["partialUpdate"] == "zero":
            agent["agentA"][r,c] = 0.0
        elif action["partialUpdate"] == "none":
            pass
        # else no-op

    # 2) Sum all agent waves
    G_total = np.zeros((N, N))
    for agent in agents:
        A_local = agent["agentA"]
        G_i = compute_heightmap(Sx, A_local, Sy)
        G_total += G_i

    # 3) clamp to [-1,1]
    np.clip(G_total, -1.0, 1.0, out=G_total)
    return G_total

# ------------------------------------------------------------------
#  Main Simulation
# ------------------------------------------------------------------

def main():
    env = init_env(numAgents=3, N=80, K=10)
    
    T = 300  
    DELAY_MS = 10  # slower for demonstration

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    frames = []
    for t in range(T):
        # gather states
        allStates = state(env, t)
        # gather actions
        allActions = actions(env, allStates)
        # update env
        G = Update(allStates, allActions, t, env)
        frames.append(G)

    X, Y = env["X_mesh"], env["Y_mesh"]

    def update_plot(frame_index):
        ax.clear()
        G = frames[frame_index]
        surf = ax.plot_surface(
            X, Y, G,
            cmap='viridis',
            linewidth=0,
            antialiased=True
        )
        ax.set_title(f"Frame {frame_index}", fontsize=10)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Height")
        ax.set_zlim(-1, 1)

    anim = FuncAnimation(fig, update_plot, frames=T, interval=DELAY_MS)
    plt.show()
    # Optionally: anim.save("discrete_fourier_agents.gif", writer="pillow", fps=10)

if __name__ == "__main__":
    main()
