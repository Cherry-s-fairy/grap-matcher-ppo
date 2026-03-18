# Role
You are a Senior Research Engineer specializing in Deep Reinforcement Learning (DRL) and Distributed Systems (UAV Cloud-Edge Coordination).

# Task
Help me scaffold a Python research project for a paper. The core idea is "Resource-Aware Task Graph Shaping and Matching in Heterogeneous UAV Swarms." 

# Key Innovation to Implement
A feedback loop where the Task Graph generation phase perceives the Resource Graph (UAV status) and previous matching performance (latency, success rate) to optimize the Task Graph's topology (granularity/dependency) for better scheduling.

# Technical Requirements
1. Framework: OpenAI Gymnasium for the RL environment.
2. Graph Processing: PyTorch Geometric (PyG) or NetworkX for handling Task Graphs (DAG) and Resource Graphs.
3. RL Algorithm: Use a Placeholder for PPO or a GNN-based Actor-Critic model.
4. Heterogeneity: UAVs must have different CPU, Bandwidth, and Battery attributes.

# Project Structure Request
Please generate the following file structure and core code:
1. `env/uav_env.py`: A Gymnasium environment that handles the state (Task Graph + Resource Graph) and step logic.
2. `models/gnn_policy.py`: A GNN-based feature extractor (using GraphConv or SageConv) to embed the two graphs.
3. `core/task_shaping.py`: The logic for "Adaptive Task Shaping" (how task nodes are modified based on feedback).
4. `configs/params.py`: Configuration for UAV heterogeneity and simulation constants.
5. `main.py`: A simple training loop to demonstrate the flow.

# Specific Logic Focus
Ensure the 'State' includes the results of the previous matching (Reschedule count, Latency) so the agent can learn to 'shape' the next task graph better.