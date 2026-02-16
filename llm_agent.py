import google.generativeai as genai
import requests
import time
import json
import config

# Configuration
API_URL = "http://localhost:5000/api"
MODEL_NAME = "gemma-3-12b-it"  # High rate limits (30 RPM, 14.4K RPD)

# Setup Gemini
genai.configure(api_key=config.GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

def get_simulation_state():
    try:
        response = requests.get(f"{API_URL}/state")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching state: {e}")
        return None

def get_grid_heatmap():
    try:
        response = requests.get(f"{API_URL}/grid/heatmap")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching heatmap: {e}")
        return None

def decide_action(state, heatmap):
    # Construct prompt
    # Extract detailed metrics
    agents_details = state.get("agents_details", [])
    agents_alive = len(agents_details)
    agents_dead = state.get("agents_dead", 0)
    step = state.get("step", 0)
    
    # Calculate stats manually since API no longer gives averages
    avg_energy = sum(a['energy'] for a in agents_details) / max(1, agents_alive)
    avg_temp = sum(a['temp'] for a in agents_details) / max(1, agents_alive)
    avg_valence = sum(a['valence'] for a in agents_details) / max(1, agents_alive)
    
    # Find critical agents (Low Energy AND/OR Low Valence)
    # Critical Energy < 50.0, Low Valence < -0.5
    critical_agents = [
        a for a in agents_details 
        if a['energy'] < 50.0 or a['valence'] < -0.5
    ]
    
    # Format critical agents for the Prompt
    critical_info = ""
    if critical_agents:
        critical_info = "CRITICAL ALERTS:\n"
        for a in critical_agents[:10]: # Limit to top 10 to save tokens
            critical_info += f"- Agent {a['id']} at ({a['x']}, {a['y']}): Energy={a['energy']}, Valence={a['valence']}\n"
    else:
        critical_info = "Status OK: No agents in critical condition."

    heatmap_summary = json.dumps(heatmap.get("heatmap", [])[:20]) 
    
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
    Look at the CRITICAL ALERTS. If an agent has low energy (<50) or negative valence, they are suffering.
    Drop food NEAR them (their x, y coordinates) to help.
    
    Response Format (JSON only):
    {{
        "thought": "Your reasoning here...",
        "action": "drop_food" or "wait",
        "x": <integer 0-79> (only if drop_food),
        "y": <integer 0-39> (only if drop_food),
        "amount": <float> (only if drop_food, default 30.0)
    }}
    """
    
    try:
        # Gemma 3 API doesn't support response_mime_type="application/json" yet
        response = model.generate_content(prompt)
        text = response.text
        
        # Clean up markdown code blocks if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
            
        return json.loads(text.strip())
    except Exception as e:
        print(f"Error generating decision: {e}")
        return None

def execute_action(decision):
    if not decision: return
    
    action = decision.get("action")
    print(f"🤔 Thought: {decision.get('thought')}")
    
    if action == "drop_food":
        payload = {
            "x": decision.get("x", 40),
            "y": decision.get("y", 20),
            "amount": decision.get("amount", 30.0)
        }
        try:
            res = requests.post(f"{API_URL}/action/drop_food", json=payload)
            print(f"✅ Action Executed: Dropped food at ({payload['x']}, {payload['y']})")
        except Exception as e:
            print(f"❌ Action Failed: {e}")
    else:
        print("zzz... Waiting.")

def main():
    print("🤖 LLM Agent Initialized. Connecting to Simulation...")
    
    # Wait for simulation to start
    time.sleep(2)
    
    while True:
        state = get_simulation_state()
        if state:
            heatmap = get_grid_heatmap()
            # Gemma 3 has 30 RPM, so we can poll more frequently (every 2-3 seconds)
            step = state.get("step", 0)
            
            # Decide every step or every few steps
            print(f"Step {step}: Thinking...")
            decision = decide_action(state, heatmap)
            if decision:
                execute_action(decision)
            
            # 30 RPM = 1 request every 2 seconds.
            time.sleep(2.5) 
        else:
            time.sleep(1.0) 

if __name__ == "__main__":
    main()
