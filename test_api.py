import requests
import time
import json

BASE_URL = "http://localhost:5000/api"

def test_api():
    print("Testing API...")
    
    # 1. Get State
    try:
        response = requests.get(f"{BASE_URL}/state")
        if response.status_code == 200:
            print("✅ GET /state success")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ GET /state failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    # 2. Get Heatmap
    try:
        response = requests.get(f"{BASE_URL}/grid/heatmap")
        if response.status_code == 200:
            print("✅ GET /grid/heatmap success")
            print(f"Heatmap data len: {len(response.json().get('heatmap', []))}")
        else:
            print(f"❌ GET /grid/heatmap failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")

    # 3. Drop Food
    try:
        payload = {"x": 40, "y": 20, "amount": 50}
        response = requests.post(f"{BASE_URL}/action/drop_food", json=payload)
        if response.status_code == 200:
            print("✅ POST /action/drop_food success")
            print(response.json())
        else:
            print(f"❌ POST /action/drop_food failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    test_api()
