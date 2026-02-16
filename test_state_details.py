import requests
import json

try:
    response = requests.get("http://localhost:5000/api/state")
    if response.status_code == 200:
        data = response.json()
        print("✅ API Response OK")
        
        # Check for new fields
        if "agents_details" in data:
            print(f"✅ 'agents_details' found. Count: {len(data['agents_details'])}")
            if data['agents_details']:
                first_agent = data['agents_details'][0]
                print(f"   First agent sample: {first_agent}")
                required = ['id', 'energy', 'temp', 'valence']
                if all(k in first_agent for k in required):
                     print("✅ Agent details structure correct.")
                else:
                     print("❌ Missing fields in agent details.")
        else:
            print("❌ 'agents_details' missing from response.")
            
        # Check backward compatibility (optional but good to know)
        if "total_energy" not in data: # We removed it
            print("ℹ️ 'total_energy' removed as expected.")
            
    else:
        print(f"❌ API Error: {response.status_code}")

except Exception as e:
    print(f"❌ Connection failed: {e}")
