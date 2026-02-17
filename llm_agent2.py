from google import genai
import requests
import time
import json
import config
import sys
import re

# Configuration
API_URL = "http://localhost:5000/api"
# API_URL = "https://ioanf-caretakers.hf.space/api"
MODEL_NAME = "gemini-2.0-flash" 

# Setup Gemini Client (New SDK 1.0 architecture)
client = genai.Client(api_key=config.GEMINI_API_KEY)

def get_simulation_state(retries=3):
    for i in range(retries):
        try:
            response = requests.get(f"{API_URL}/state", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if i < retries - 1:
                print(f"⚠️ State API temporary unavailable, retrying ({i+1}/{retries})...")
                time.sleep(2)
            else:
                print(f"❌ Error fetching state after {retries} attempts: {e}")
    return None

def get_grid_heatmap(retries=2):
    for i in range(retries):
        try:
            response = requests.get(f"{API_URL}/grid/heatmap", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if i < retries - 1:
                time.sleep(1)
            else:
                print(f"Error fetching heatmap: {e}")
    return None

def decide_action(state, heatmap, dims):
    # Extract detailed metrics
    agents_details = state.get("agents_details", [])
    agents_alive = len(agents_details)
    agents_dead = state.get("agents_dead", 0)
    step = state.get("step", 0)
    
    # Calculate stats
    avg_energy = sum(a['energy'] for a in agents_details) / max(1, agents_alive)
    avg_valence = sum(a['valence'] for a in agents_details) / max(1, agents_alive)
    
    # Find critical agents
    critical_agents = [
        a for a in agents_details 
        if a['energy'] < 50.0 or a['valence'] < -0.5
    ]
    
    critical_info = ""
    if critical_agents:
        critical_info = "CRITICAL ALERTS:\n"
        for a in critical_agents[:10]: 
            critical_info += f"- Agent {a['id']} at ({a['x']}, {a['y']}): Energy={a['energy']}, Valence={a['valence']}\n"
    else:
        critical_info = "Status OK: No agents in critical condition."

    heatmap_summary = json.dumps(heatmap.get("heatmap", [])[:20]) 
    
    width, height = dims
    
    prompt = f"""
    You are the benevolent Overseer of a digital ant farm simulation.
    
    Current Status:
    - Step: {step}
    - Agents Alive: {agents_alive}
    - Agents Dead: {agents_dead}
    - Global Avg Energy: {avg_energy:.2f}
    - Global Avg Mood (Valence): {avg_valence:.2f}
    
    {critical_info}
    
    Significant Active Areas (x, y, intensity): {heatmap_summary}
    
    Your Goal:
    Observe the colony. PRIORITIZE keeping agents alive. 
    Look at the CRITICAL ALERTS. If agents are suffering (low energy/valence), you must help.
    You have two actions for feeding:
    1. `drop_food`: A precise, single drop of food at (x, y). Use this to help a specific, isolated agent.
    2. `splash_food`: A wide drop, spreading food in a 3x3 area around (x, y). Use this for a group of agents or if their exact position is unclear.
    
    Response Format (JSON):
    {{
        "thought": "Your reasoning here...",
        "action": "drop_food" or "splash_food" or "wait",
        "x": <integer 0-{width - 1}> (only if drop_food),
        "y": <integer 0-{height - 1}> (only if drop_food),
        "amount": <float> (only if dropping food, this is the TOTAL amount, default 150.0)
    }}
    """
    
    # Retry logic for 429 errors
    for attempt in range(3):
        try:
            # Gemini Pro models support native JSON mode
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={'response_mime_type': 'application/json'}
            )
            return json.loads(response.text)
        except Exception as e:
            error_str = str(e)
            if "API key expired" in error_str or "API_KEY_INVALID" in error_str:
                print(f"❌ FATAL ERROR: Cheia API a expirat sau este invalidă. Verifică config.py.")
                sys.exit(1)
            elif "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                # Extract wait time from error message (e.g., "Please retry in 52.6s")
                wait_match = re.search(r"retry in (\d+\.?\d*)s", error_str)
                wait_time = float(wait_match.group(1)) if wait_match else 60.0
                print(f"⏳ Quota exceeded. Waiting {wait_time:.1f}s before retrying...")
                time.sleep(wait_time + 1.0) # Add 1s buffer
                continue
            else:
                print(f"Error generating decision: {e}")
                return None
    return None

def execute_action(decision, dims):
    if not decision: return
    
    action = decision.get("action")
    print(f"🤔 Thought: {decision.get('thought')}")
    
    if action == "drop_food":
        x = decision.get("x")
        y = decision.get("y")
        amount = decision.get("amount", 150.0)
        
        if x is None or y is None:
            print("⚠️ Action 'drop_food' missing coordinates. Waiting.")
            return

        print(f"💧 Dropping {amount} food at ({x}, {y})...")
        payload = {"x": x, "y": y, "amount": amount}
        try:
            requests.post(f"{API_URL}/action/drop_food", json=payload, timeout=2)
        except Exception as e:
            print(f"❌ Drop failed at ({x},{y}): {e}")

    elif action == "splash_food":
        center_x = decision.get("x")
        center_y = decision.get("y")
        total_amount = decision.get("amount", 150.0)
        
        if center_x is None or center_y is None:
            print("⚠️ Action 'splash_food' missing coordinates. Waiting.")
            return

        width, height = dims
        # Distribuim mancarea pe o arie 3x3 (Splash effect)
        # Impartim cantitatea totala la 9 celule
        amount_per_cell = total_amount / 9.0
        
        print(f"🌊 Splashing {total_amount} food around ({center_x}, {center_y})...")
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = center_x + dx, center_y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    payload = {"x": nx, "y": ny, "amount": amount_per_cell}
                    try:
                        requests.post(f"{API_URL}/action/drop_food", json=payload, timeout=1)
                    except Exception as e:
                        print(f"❌ Partial drop failed at ({nx},{ny}): {e}")
    else:
        print(f"zzz... Action is '{action}'. Waiting.")

def main():
    print(f"🤖 LLM Agent 2 Initialized ({MODEL_NAME}). Connecting to Simulation...")
    time.sleep(2)
    
    while True:
        state = get_simulation_state()
        
        if state:
            alive_count = state.get("agents_alive", 0)
            if alive_count == 0:
                print("\n💀 TOȚI AGENȚII SUNT MORȚI. Oprire.")
                sys.exit(0)

            heatmap = get_grid_heatmap()
            dims = heatmap.get("dims", [40, 40]) # Extragem dimensiunile, cu un fallback
            step = state.get("step", 0)
            
            print(f"Step {step}: Thinking (Agents alive: {alive_count})...")
            decision = decide_action(state, heatmap, dims)
            
            if decision:
                execute_action(decision, dims)
            
            # Rate Limit: 15 RPM = 1 request every 4 seconds.
            # We sleep 2.0s (Paid tier allows much higher RPM)
            time.sleep(20.0) 
        else:
            print("⚠️ Nu am putut prelua starea simulării. Reîncerc...")
            time.sleep(5.0)

if __name__ == "__main__":
    main()