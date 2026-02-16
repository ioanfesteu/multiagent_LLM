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
def get_plot_figure(model, step_number=0):
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
            
        ax.scatter(x, y, c=c, s=120, edgecolors='black', linewidth=1.5, zorder=10)

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
    
    # Initialize model ONCE (or when needed)
    # Model is now initialized globally, no need for use_memo here.

    def on_step():
        with shared.simulation_lock:
            if shared.simulation_model:
                shared.simulation_model.step()
        set_tick(tick + 1)

    def on_reset():
        with shared.simulation_lock:
            shared.simulation_model = DualDriveModel()
        set_tick(0)
        set_playing(False)
        
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
        n_alive = len(alive_agents)
        dead_count = model.dead_count
    else:
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
        solara.Markdown("**API Status:**")
        solara.Markdown("Flask Server running on port 5000")
        solara.Markdown("Endpoints:")
        solara.Markdown("- `GET /api/state`")
        solara.Markdown("- `POST /api/action/drop_food`")

    # Main View
    if shared.simulation_model:
        fig = get_plot_figure(shared.simulation_model, step_number=tick)
        solara.FigureMatplotlib(fig)
    else:
        solara.Markdown("Initializing Model...")
