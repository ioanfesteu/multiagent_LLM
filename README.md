---
title: Caretakers
emoji: 🤖
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
---

# Multiagent LLM: Observers & Caretakers

## Abstract

This project implements a multi-agent simulation environment designed to explore the interaction between autonomous agents and an external Large Language Model (LLM) acting as an observer and caretaker. Built upon the principles of Active Inference and allostatic regulation, the simulation features agents that navigate a dynamic environment, managing their internal physiological states (energy and temperature). The system provides a dual interface: a **Solara-based GUI** for human observation and control, and a **Flask API** that allows an external LLM agent (powered by **Google Gemma 3**) to monitor the simulation state and intervene by providing resources (food) when necessary. This setup enables research into hybrid human-AI-agent systems and the emergent behaviors of AI caretakers.

> **Note:** This project is an evolution of the [multiagent_FEP](https://github.com/ioanfesteu/multiagent_FEP) project, extending it with external LLM agency capabilities.

## 🚀 Features

*   **Autonomous Agents**: Agents driven by active inference, balancing energy and temperature needs.
*   **Dual Interface**:
    *   **Human UI**: Real-time visualization, heatmaps, and controls via [Solara](https://solara.dev/).
    *   **LLM API**: RESTful API for external agents to query state and execute actions.
*   **AI Caretaker**: An external Python agent (`llm_agent.py`) that uses **Google Gemma 3 (12B)** to reason about the simulation and autonomously intervene.
*   **Real-time Metrics**: Live tracking of agent survival, energy levels, and spatial activity.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/ioanfesteu/multiagent_LLM.git
    cd multiagent_LLM
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Key**:
    Create a file named `config.py` in the root directory and add your Google Gemini API key:
    ```python
    # config.py
    GEMINI_API_KEY = "YOUR_API_KEY_HERE"
    ```

## 🎮 Usage

To run the full system, you will need two terminal windows.

### 1. Start the Simulation Server (Human UI)
This command launches the Solara dashboard and the background Flask API.

```bash
solara run app.py
```
*   **GUI**: Open your browser at [http://localhost:8765](http://localhost:8765).
*   **API**: The Flask server starts automatically on port `5000`.

### 2. Start the LLM Agent (AI Caretaker)
In a separate terminal, launch the autonomous agent.

```bash
python llm_agent.py
```
The agent will:
1.  Connect to the simulation API.
2.  Observe the state (every ~2.5 seconds).
3.  Use **Gemma 3** to decide whether to intervene.
4.  Drop food if it deems necessary to save agents or stimulate the colony.

## 🏗️ Architecture

*   **`model.py`**: The core Mesa simulation logic (environment, time stepping).
*   **`agents.py`**: The `AllostaticAgent` class implementing active inference reasoning.
*   **`app.py`**: The main entry point. Orchestrates the Solara UI and starts the Flask API thread.
*   **`api_server.py`**: A Flask application exposing endpoints (`/api/state`, `/api/action/drop_food`) for external interaction.
*   **`llm_agent.py`**: The client script that bridges the simulation API with the Google Generative AI SDK (Gemma 3).
*   **`shared.py`**: Thread-safe state management singleton to synchronize the UI and API.

## 🤝 Contributing

This project is open for experimentation with different LLM prompts, agent behaviors, and environmental dynamics. Feel free to fork and submit pull requests!

## 📜 License

[MIT License](LICENSE)
