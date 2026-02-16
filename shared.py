# shared.py
import threading

# Global simulation model instance
simulation_model = None

# Lock to ensure thread safety when accessing the model
# (Flask responding to API vs Solara/Loop stepping the model)
simulation_lock = threading.Lock()
