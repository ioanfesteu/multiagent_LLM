# app.py
from model import DualDriveModel
from agents import (
    NUM_AGENTS, COLOR_OK, COLOR_HUNGRY, COLOR_COLD, COLOR_HOT, 
    COLOR_FOOD, COLOR_TRAIL, WEIGHT_TEMP, WEIGHT_ENERGY
)
import shared
import threading
import time
import solara
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from api_server import run_api_server
import asyncio
import numpy as np

# ==========================================
# BACKGROUND API RUNNER
# ==========================================
# We start the Flask server in a separate thread the first time this module is loaded
# or explicitly control it. For Solara, global scope runs once on startup usually, 
# but per-user connections might trigger re-runs. We use a simple flag check.
if not hasattr(shared, 'api_thread_started'):
    t = threading.Thread(target=run_api_server, daemon=True)
    t.start()
    shared.api_thread_started = True

# ==========================================
# VISUALIZATION LOGIC
# ==========================================
def get_plot_figure(model, step_number=0, selected_id=None):
    """
    Visualization logic adapted from FEP project.
    """
    aspect_ratio = model.grid.width / model.grid.height
    fig_width = 8 * (aspect_ratio if aspect_ratio > 1 else 1)
    fig = plt.figure(figsize=(fig_width, 6))
    ax = fig.add_subplot(111)
    
    # 1. Heatmap (Temperature)
    ax.imshow(model.temperature.T, origin='lower', cmap='coolwarm', alpha=0.4, vmin=0, vmax=40)
    
    # 2. Food patches
    fx, fy, fs = [], [], []
    for x in range(model.grid.width):
        for y in range(model.grid.height):
            val = model.food[x, y]
            if val > 1.0:
                fx.append(x)
                fy.append(y)
                fs.append(min(val * 3, 150)) 
    if fx:
        ax.scatter(fx, fy, c=COLOR_FOOD, s=fs, alpha=0.6, edgecolors='green', label='Food Source')

    # 3. Social Scent Trails
    sx, sy, ss = [], [], []
    for x in range(model.grid.width):
        for y in range(model.grid.height):
            val = model.food_scent[x, y]
            if val > 0.1:
                sx.append(x)
                sy.append(y)
                ss.append(min(val * 20, 50))
    if sx:
        ax.scatter(sx, sy, c=COLOR_TRAIL, s=ss, alpha=0.8, marker='.', label='Food Trail')

    # 4. Agents
    alive_agents = [a for a in model.agents if a.is_alive]
    n_alive = len(alive_agents)

    for agent in model.agents:
        # Body & Color Logic
        x, y = agent.pos
        if not agent.is_alive:
            # Maybe show dead agents differently or not at all if removed
            continue

        c = COLOR_OK
        diff_T = agent.T_int - agent.T_pref 
        err_T_weighted = abs(diff_T) * WEIGHT_TEMP
        err_E_weighted = max(0, agent.E_crit - agent.E_int) * WEIGHT_ENERGY
        
        if err_E_weighted > err_T_weighted and err_E_weighted > 1.0:
            c = COLOR_HUNGRY
        elif err_T_weighted > err_E_weighted and err_T_weighted > 1.0:
            if diff_T > 0: c = COLOR_HOT 
            else: c = COLOR_COLD 
            
        # Highlight selected agent
        is_selected = (agent.unique_id == selected_id)
        ec = 'red' if is_selected else 'black'
        lw = 3.0 if is_selected else 1.5
        z = 20 if is_selected else 10
            
        ax.scatter(x, y, c=c, s=120, edgecolors=ec, linewidth=lw, zorder=z)
        
        # Add numeric label
        ax.text(x, y, str(agent.unique_id), color='black', fontsize=8, 
                fontweight='bold', ha='center', va='center', zorder=z+1,
                bbox=dict(boxstyle='circle,pad=0.1', facecolor='white', alpha=0.7, edgecolor='none'))

    # 5. Overlay Text
    if n_alive > 0:
        avg_E = sum(a.E_int for a in alive_agents) / n_alive
        avg_T = sum(a.T_int for a in alive_agents) / n_alive
        avg_Valence = sum(a.valence_integrated for a in alive_agents) / n_alive
    else:
        avg_E = 0; avg_T = 0; avg_Valence = 0

    textstr = '\n'.join((
        f'Step: {step_number}',
        f'Alive: {n_alive} | Dead: {model.dead_count}',
        f'Avg Energy: {avg_E:.1f}',
        f'Avg Temp: {avg_T:.1f}',
        f'Avg Mood: {avg_Valence:.2f}'
    ))

    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    ax.set_xlim(-0.5, model.grid.width-0.5)
    ax.set_ylim(-0.5, model.grid.height-0.5)
    ax.axis('off')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.close(fig)
    return fig

@solara.component
def DivergentBar(value, center, scale, color):
    """
    Renders a bar that grows from the center (50%).
    value: current value
    center: the 'zero' or ideal point
    scale: multiplier to map units to percentage (e.g. 1 unit = 25%)
    color: color of the bar
    """
    diff = value - center
    width = np.clip(abs(diff) * scale, 0, 50)
    left = 50 if diff >= 0 else 50 - width
    
    style = {
        "width": "100%",
        "height": "14px",
        "background": "#e0e0e0",
        "position": "relative",
        "border-radius": "4px",
        "overflow": "hidden",
        "margin-bottom": "8px"
    }
    
    bar_style = {
        "position": "absolute",
        "left": f"{left}%",
        "width": f"{width}%",
        "height": "100%",
        "background": color,
        "transition": "width 0.1s, left 0.1s"
    }
    
    marker_style = {
        "position": "absolute",
        "left": "50%",
        "width": "2px",
        "height": "100%",
        "background": "#333",
        "z-index": "2",
        "opacity": "0.5"
    }

    # Outer container
    with solara.Row(style=style):
        # The actual bar
        solara.HTML(tag="div", style=bar_style)
        # The center marker
        solara.HTML(tag="div", style=marker_style)

@solara.component
def AgentCard(agent, tick):
    # Energy Color Logic
    energy_color = "green" if agent.E_int > 60 else ("orange" if agent.E_int > 30 else "red")
        
    # Valence Color Logic (Mood)
    valence_color = "dodgerblue"
    if agent.valence_integrated < -0.5:
        valence_color = "red"
    elif agent.valence_integrated > 0.5:
        valence_color = "limegreen"

    with solara.Card(f"Monitoring Agent {agent.unique_id}", subtitle=f"Pos: ({agent.pos[0]}, {agent.pos[1]})", margin=1):
        with solara.Column():
            # 1. Energy
            solara.Markdown(f"**Energy (0-100):** {agent.E_int:.1f}")
            solara.ProgressLinear(value=agent.E_int, color=energy_color)
            
            # 2. Temperature (Normal Bar)
            solara.Markdown(f"**Temperature:** {agent.T_int:.1f}°C")
            # Map 0-50 to 0-100
            temp_norm = np.clip(agent.T_int * 2, 0, 100)
            solara.ProgressLinear(value=temp_norm, color="info")
            
            solara.Markdown("---")
            
            # 3. Precision (Beta)
            solara.Markdown(f"**Precision (Decision Determinism):** {agent.current_beta:.1f}")
            beta_norm = np.clip(agent.current_beta * 3.33, 0, 100)
            solara.ProgressLinear(value=beta_norm, color="purple")

            # 4. Mood (Valence - Divergent from 0.0)
            solara.Markdown(f"**Mood (Valence: -2 to 2):** {agent.valence_integrated:.2f}")
            # Range -2 to 2, Center 0. 1 unit = 25% (since 2 units = 50%)
            DivergentBar(value=agent.valence_integrated, center=0.0, scale=25.0, color=valence_color)

# ==========================================
# SOLARA PAGE
# ==========================================

# Initialize model globally so API works even if no user connects to GUI
if shared.simulation_model is None:
    shared.simulation_model = DualDriveModel()

@solara.component
def Page():
    # Use state to track ticks and trigger re-renders
    tick, set_tick = solara.use_state(0)
    is_playing, set_playing = solara.use_state(False)
    selected_agent_id, set_selected_agent_id = solara.use_state(None)
    
    def on_step():
        with shared.simulation_lock:
            if shared.simulation_model:
                shared.simulation_model.step()
        set_tick(tick + 1)

    def on_reset():
        with shared.simulation_lock:
            shared.simulation_model = DualDriveModel()
        # Force a UI update by changing tick even if it was 0
        set_tick(lambda t: 0 if t != 0 else -1)
        set_playing(False)
        set_selected_agent_id(None)
        
    def on_play():
        set_playing(not is_playing)

    # Background loop for "Play" mode
    def run_loop():
        if not is_playing:
            return
        
        async def loop():
            try:
                while True:
                    with shared.simulation_lock:
                        if shared.simulation_model:
                            shared.simulation_model.step()
                            # Check if simulation should end
                            if len(shared.simulation_model.agents) == 0:
                                set_playing(False)
                                break
                    set_tick(lambda t: t + 1)
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                pass
        
        task = asyncio.create_task(loop())
        return lambda: task.cancel()

    solara.use_effect(run_loop, [is_playing])

    # Stats Calculation
    if shared.simulation_model:
        model = shared.simulation_model
        alive_agents = [a for a in model.agents if a.is_alive]
        alive_ids = [a.unique_id for a in alive_agents]
        n_alive = len(alive_agents)
        dead_count = model.dead_count
        
        # Ensure selected agent is still alive
        if selected_agent_id and selected_agent_id not in alive_ids:
            set_selected_agent_id(None)
    else:
        alive_ids = []
        n_alive = 0
        dead_count = 0

    with solara.Sidebar():
        solara.Markdown("## 🤖 Multiagent LLM Project")
        solara.Markdown("Monitoring Station for LLM Observers.")
        
        with solara.Row():
            solara.Button("Step", on_click=on_step, color="warning")
            solara.Button("Play/Pause", on_click=on_play, color="success" if is_playing else "primary")
            solara.Button("Reset", on_click=on_reset, color="error")
            
        solara.Markdown(f"**Step:** {tick}")
        solara.Markdown(f"**Alive:** {n_alive} | **Dead:** {dead_count}")
        
        solara.Markdown("---")
        solara.Markdown("**🔍 Focused Monitoring**")
        solara.Select(label="Select Agent", value=selected_agent_id, values=alive_ids, on_value=set_selected_agent_id)
        
        solara.Markdown("---")
        solara.Markdown("**API Status:**")
        solara.Markdown("Flask Server running on port 5000")
        
        solara.Markdown("---")
        solara.Markdown("**🎨 Map Legend (Agent Status)**")
        
        def LegendItem(color, label, text_color="black"):
            with solara.Row(style={"align-items": "center", "margin-bottom": "4px"}):
                # Mimic the map marker: circle, border, number
                style = {
                    "width": "24px",
                    "height": "24px",
                    "border-radius": "50%",
                    "background-color": color,
                    "border": "1.5px solid black",
                    "display": "flex",
                    "align-items": "center",
                    "justify-content": "center",
                    "font-size": "10px",
                    "font-weight": "bold",
                    "color": text_color,
                    "margin-right": "8px"
                }
                solara.HTML(tag="div", style=style, children="7")
                solara.Markdown(label)

        LegendItem(COLOR_OK, f"**White**: Comfortable")
        LegendItem(COLOR_HUNGRY, f"**Brown**: Hungry / Low Energy", text_color="white")
        LegendItem(COLOR_COLD, f"**Blue**: Cold", text_color="white")
        LegendItem(COLOR_HOT, f"**Red**: Hot", text_color="white")
        
        solara.Markdown("---")
        solara.Markdown("**🌱 Environment**")
        solara.Markdown(f"- 🟢 **Lime**: Food Patch")
        solara.Markdown(f"- 🟧 **Orange**: Social Scent Trace")

    # Main View
    if shared.simulation_model:
        fig = get_plot_figure(shared.simulation_model, step_number=tick, selected_id=selected_agent_id)
        solara.FigureMatplotlib(fig)
        
        if selected_agent_id:
            agent = next((a for a in shared.simulation_model.agents if a.unique_id == selected_agent_id), None)
            if agent:
                solara.Markdown("### 📋 Focused Agent Telemetry")
                AgentCard(agent, tick)
        else:
            solara.Markdown("ℹ️ *Select an agent from the sidebar to view detailed telemetry.*")
    else:
        solara.Markdown("Initializing Model...")
