import mesa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import matplotlib.patches as mpatches

# --- Custom RandomActivation class to be fully self-contained ---
class RandomActivation:
    """
    A scheduler that activates each agent once per step in a random order.
    This custom class is fully self-contained to avoid Mesa versioning issues.
    """
    def __init__(self, model):
        self.model = model
        self.agents = []
        self.steps = 0
        self.time = 0

    def add(self, agent):
        """Add an agent to the scheduler."""
        self.agents.append(agent)

    def remove(self, agent):
        """
        Remove all instances of a given agent from the schedule.
        Necessary for agent death.
        """
        while agent in self.agents:
            self.agents.remove(agent)

    def step(self):
        """Executes a step for all agents in a random order."""
        # Create a copy to iterate over, as agents can be added/removed during a step.
        agent_list = self.agents[:]
        random.shuffle(agent_list)
        for agent in agent_list:
            # Agent may have been removed since the list was copied
            if agent in self.agents:
                agent.step()
        
        self.steps += 1
        self.time += 1

    def get_agent_count(self):
        """Returns the current number of agents in the scheduler."""
        return len(self.agents)

# --- 1. Agent Definition ---
class GbmCell(mesa.Agent):
    """
    An agent representing a single Glioblastoma cell.
    """
    def __init__(self, unique_id, model, state='NPC-like'):
        # Using the stable __init__ method that works in your environment
        self.unique_id = unique_id
        self.model = model
        self.pos = None 
        self.set_state(state)
        self.time_in_state = 0

    def set_state(self, state):
        """Helper function to set state and associated properties."""
        self.state = state
        self.time_in_state = 0 # Reset timer on state change
        if self.state == 'NPC-like':
            self.proliferation_chance = 0.1
            self.stress_resistance = 0.70 # TUNED: Increased resistance for gradual decline
        elif self.state == 'MES-like':
            self.proliferation_chance = 0.02
            self.stress_resistance = 0.95 
        elif self.state == 'AC-like':
            self.proliferation_chance = 0.01
            self.stress_resistance = 0.80 # TUNED: Increased resistance for gradual decline
        elif self.state == 'OPC-like':
            self.proliferation_chance = 0.05
            self.stress_resistance = 0.75 # TUNED: Increased resistance for gradual decline

    def step(self):
        """Defines the agent's behavior at each model step."""
        self.time_in_state += 1
        
        if self.die():
            return

        self.transition()
        self.proliferate()
        self.move()

    def die(self):
        """
        Determines if the cell dies based on environmental stress and its resistance.
        Returns True if the cell died, False otherwise.
        """
        death_chance = self.model.environmental_stress * (1 - self.stress_resistance)
        if self.model.random.random() < death_chance:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
            return True
        return False

    def transition(self):
        """
        Rule for state transitions. This is the core of the "swarm intelligence".
        """
        if self.state == 'MES-like' and self.model.environmental_stress < 0.2:
            if self.model.random.random() < 0.10: 
                self.set_state('NPC-like')
                return

        if self.state == 'NPC-like':
            if self.model.random.random() < 0.01:
                self.set_state(self.model.random.choice(['AC-like', 'OPC-like']))
                return

        if self.model.random.random() < self.model.environmental_stress * (1 - self.stress_resistance):
            if self.state != 'MES-like':
                self.set_state('MES-like')
            return

        neighbors = self.model.grid.get_neighbors(self.pos, moore=True)
        if not neighbors:
            return

        neighbor_states = [n.state for n in neighbors if isinstance(n, GbmCell)]
        if not neighbor_states:
            return
        
        mes_neighbors = neighbor_states.count('MES-like')
        if (mes_neighbors / len(neighbor_states)) > 0.5:
            if self.model.random.random() < 0.2:
                if self.state != 'MES-like':
                    self.set_state('MES-like')
                return

        if self.state in ['AC-like', 'OPC-like'] and self.time_in_state > 10:
             npc_neighbors = neighbor_states.count('NPC-like')
             if (npc_neighbors / len(neighbor_states)) > 0.5:
                 if self.model.random.random() < 0.1:
                     self.set_state('NPC-like')
                     return

    def proliferate(self):
        """
        A cell may divide (create a new agent) into an empty neighboring patch.
        """
        if self.model.random.random() < self.proliferation_chance:
            possible_moves = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            empty_patches = [p for p in possible_moves if self.model.grid.is_cell_empty(p)]
            if empty_patches:
                new_pos = self.model.random.choice(empty_patches)
                new_cell = GbmCell(self.model.next_id(), self.model, state=self.state)
                self.model.grid.place_agent(new_cell, new_pos)
                self.model.schedule.add(new_cell)

    def move(self):
        """
        The cell moves to a random empty neighboring cell.
        """
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        empty_patches = [p for p in possible_steps if self.model.grid.is_cell_empty(p)]
        if empty_patches:
            new_position = self.model.random.choice(empty_patches)
            self.model.grid.move_agent(self, new_position)


# --- 2. Model Definition ---
class GbmModel(mesa.Model):
    """
    The main model class for the GBM Swarm simulation.
    """
    def __init__(self, N=50, width=20, height=20, environmental_stress=0.1):
        super().__init__()
        self.num_agents = N
        self.grid = mesa.space.SingleGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.running = True
        self.next_id_val = 0

        self.environmental_stress = environmental_stress

        # Place initial agents in a central cluster on an empty grid
        initial_states = ['NPC-like', 'AC-like', 'OPC-like']
        center_x = width // 2
        center_y = height // 2
        radius = int(np.sqrt(N)) + 1 

        for i in range(self.num_agents):
            state = self.random.choice(initial_states)
            a = GbmCell(self.next_id(), self, state=state)
            
            x = self.random.randrange(center_x - radius, center_x + radius)
            y = self.random.randrange(center_y - radius, center_y + radius)
            
            while not self.grid.is_cell_empty((x, y)):
                 x = self.random.randrange(center_x - radius, center_x + radius)
                 y = self.random.randrange(center_y - radius, center_y + radius)
            
            self.grid.place_agent(a, (x, y))
            self.schedule.add(a)

        self.datacollector = mesa.datacollection.DataCollector(
            model_reporters={"CellCounts": lambda m: self.get_state_counts()}
        )

    def next_id(self):
        """Helper to get a unique ID for a new agent."""
        self.next_id_val += 1
        return self.next_id_val

    def get_state_counts(self):
        """Helper to count cells of each type for data collection."""
        counts = {"NPC-like": 0, "AC-like": 0, "OPC-like": 0, "MES-like": 0}
        for cell in self.schedule.agents:
            if isinstance(cell, GbmCell):
                counts[cell.state] += 1
        return counts

    def step(self):
        """Advance the model by one step."""
        if self.schedule.get_agent_count() == 0:
            self.running = False
            return
        self.datacollector.collect(self)
        self.schedule.step()


# --- 3. Visualization and Execution ---
def get_agent_color(agent):
    """Assign a color to an agent based on its state."""
    if agent is None:
        return "#FFFFFF" # White for empty
    
    state_colors = {
        "NPC-like": "#2ECC71", # Green
        "AC-like": "#3498DB",  # Blue
        "OPC-like": "#F1C40F", # Yellow
        "MES-like": "#E74C3C"  # Red
    }
    return state_colors.get(agent.state, "#000000")

def run_simulation_and_visualize(steps=200, N=50, width=50, height=50, stress_schedule=None):
    """
    Runs the simulation, saves the grid evolution to a file, and shows only the cell count plot.
    """
    # --- Setup ---
    model = GbmModel(N, width, height, environmental_stress=0.1)
    
    # --- Data storage for visualization ---
    grid_history = []
    
    # --- Simulation run ---
    for i in range(steps):
        if stress_schedule and i in stress_schedule:
            model.environmental_stress = stress_schedule[i]
            print(f"Step {i}: Stress changed to {model.environmental_stress}")

        model.step()
        
        # Store grid state for animation
        grid_state = np.zeros((model.grid.width, model.grid.height, 3), dtype=np.float32)
        for x in range(model.grid.width):
            for y in range(model.grid.height):
                content = model.grid[x, y]
                agent = content
                color_hex = get_agent_color(agent)
                color_rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) / 255.0 for i in (0, 2, 4))
                grid_state[x, y] = color_rgb
        grid_history.append(grid_state)

    # --- Data Extraction ---
    results = model.datacollector.get_model_vars_dataframe()
    state_counts = results["CellCounts"].apply(pd.Series)

    # --- Create and Save Animation ---
    fig_anim, ax_anim = plt.subplots(figsize=(10, 8)) # Make figure wider
    ax_anim.set_xticks([])
    ax_anim.set_yticks([])
    im = ax_anim.imshow(grid_history[0], animated=True)

    # ADDED: Create a legend for the animation
    legend_colors = {
        "NPC-like": "#2ECC71",
        "AC-like": "#3498DB",
        "OPC-like": "#F1C40F",
        "MES-like": "#E74C3C"
    }
    patches = [mpatches.Patch(color=color, label=state) for state, color in legend_colors.items()]
    # Place legend outside the plot area
    ax_anim.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    def update_fig(j):
        im.set_array(grid_history[j])
        ax_anim.set_title(f"GBM Swarm Simulation | Step: {j}")
        return im,

    ani = animation.FuncAnimation(fig_anim, update_fig, frames=len(grid_history), interval=100, blit=True)

    try:
        output_filename = 'gbm_swarm_simulation.gif'
        print(f"Saving animation to {output_filename}... (this may take a moment)")
        # Adjust layout to make room for the legend before saving
        fig_anim.tight_layout(rect=[0, 0, 0.8, 1])
        ani.save(output_filename, writer='pillow', fps=15)
        print("Animation saved successfully.")
    except Exception as e:
        print(f"Could not save animation. Error: {e}")
    
    plt.close(fig_anim) # Close the animation figure to prevent it from showing

    # --- Create and Show Cell Count Plot ---
    fig_counts, ax_counts = plt.subplots(figsize=(12, 6))
    
    state_colors_for_plot = {
        "NPC-like": "#2ECC71", "AC-like": "#3498DB",
        "OPC-like": "#F1C40F", "MES-like": "#E74C3C"
    }
    
    for state, color in state_colors_for_plot.items():
        if state in state_counts.columns:
            ax_counts.plot(state_counts.index, state_counts[state], label=state, color=color, linewidth=2)

    total_gbm_cells = state_counts.sum(axis=1)
    ax_counts.plot(total_gbm_cells.index, total_gbm_cells, label='Total GBM Cells', color='black', linestyle='--', linewidth=2)

    if stress_schedule:
        stress_periods = sorted(stress_schedule.keys())
        start_stress = stress_periods[0]
        end_stress = stress_periods[1] if len(stress_periods) > 1 else steps
        ax_counts.axvspan(start_stress, end_stress, color='red', alpha=0.15, label='High Stress Period')

    ax_counts.set_title("Absolute Cell Counts Over Time")
    ax_counts.set_xlabel("Time Step")
    ax_counts.set_ylabel("Number of Cells")
    ax_counts.legend(loc='upper left')
    ax_counts.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_counts.set_xlim(0, steps)
    ax_counts.set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("Pandas library not found. Please install it using: pip install pandas")
        exit()

    stress_schedule = {
        51: 0.8,
        101: 0.1
    }

    run_simulation_and_visualize(
        steps=200,
        N=100,
        width=50,
        height=50,
        stress_schedule=stress_schedule
    )
