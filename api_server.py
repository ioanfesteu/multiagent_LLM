# api_server.py
from flask import Flask, jsonify, request
import threading
import shared
import logging

# Disable Flask default logging to avoid cluttering stdout
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

@app.route('/api/state', methods=['GET'])
def get_state():
    with shared.simulation_lock:
        if shared.simulation_model is None:
            return jsonify({"error": "Simulation not initialized"}), 503
        
        model = shared.simulation_model
        
        # Collect basic stats
        agents = model.agents
        alive_agents = [a for a in agents if a.is_alive]
        
        state = {
            "step": model.steps,
            # Check if model has schedule, if not use internal counter if any. 
            # Our DualDriveModel doesn't explicitly use a scheduler in the FEP code, 
            # but we can infer step from data collector or add a step counter.
            # Let's count alive/dead.
            "agents_alive": len(alive_agents),
            "agents_dead": model.dead_count,
            "agents_details": [
                {
                    "id": a.unique_id,
                    "x": a.pos[0],
                    "y": a.pos[1],
                    "energy": round(a.E_int, 2),
                    "temp": round(a.T_int, 2),
                    "valence": round(a.valence_integrated, 2)
                }
                for a in alive_agents
            ],
            "food_patches_summary": [
                 {"x": x, "y": y, "amount": model.food[x,y]} 
                 for x in range(model.grid.width) 
                 for y in range(model.grid.height) 
                 if model.food[x, y] > 10.0 # Only significant patches
            ]
        }
        return jsonify(state)

@app.route('/api/grid/heatmap', methods=['GET'])
def get_heatmap():
    with shared.simulation_lock:
        if shared.simulation_model is None:
            return jsonify({"error": "Simulation not initialized"}), 503
        
        # Return a simplified representation of the grid (e.g. just food or scent)
        # For bandwidth, returning the full float array might be too much.
        # Let's return non-zero food locations and maybe scent summary.
        
        model = shared.simulation_model
        
        # Simplified heatmap: list of [x, y, value] for significantly active cells
        # We can return 'food_scent' as the heatmap
        heatmap_data = []
        rows, cols = model.food_scent.shape
        for x in range(rows):
            for y in range(cols):
                val = model.food_scent[x, y]
                if val > 0.1:
                    heatmap_data.append([x, y, round(val, 2)])
        
        return jsonify({"heatmap": heatmap_data, "dims": [rows, cols]})

@app.route('/api/grid/description', methods=['GET'])
def get_description():
    with shared.simulation_lock:
        if shared.simulation_model is None:
            return jsonify({"error": "Simulation not initialized"}), 503
        
        model = shared.simulation_model
        agents = [a for a in model.agents if a.is_alive]
        
        desc = f"Simulation Step. Agents alive: {len(agents)}. \n"
        if agents:
            avg_temp = sum(a.T_int for a in agents)/len(agents)
            desc += f"Average Agent Temperature: {avg_temp:.2f}. "
        
        return jsonify({"description": desc})

@app.route('/api/action/drop_food', methods=['POST'])
def drop_food():
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    
    x = data.get('x')
    y = data.get('y')
    amount = data.get('amount', 20.0)
    
    if x is None or y is None:
        return jsonify({"error": "Missing x or y coordinates"}), 400

    with shared.simulation_lock:
        if shared.simulation_model is None:
            return jsonify({"error": "Simulation not initialized"}), 503
        
        shared.simulation_model.drop_food(int(x), int(y), float(amount))
        
    return jsonify({"status": "success", "message": f"Dropped {amount} food at ({x}, {y})"}), 200

def run_api_server():
    print("Starting Flask API on port 5000...")
    app.run(port=5000, debug=True, use_reloader=False, host='0.0.0.0')
