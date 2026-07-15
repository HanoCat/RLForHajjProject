# APBC-RL 

(Adaptive Pedestrian Barrier Control using Reinforcement Learning) is a framework for optimizing pedestrian barrier configurations in JuPedSim crowd simulations. It combines image-based crowd initialization, adaptive barrier control, and reinforcement learning to learn effective barrier configurations under different crowd conditions.


![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![JuPedSim](https://img.shields.io/badge/JuPedSim-Latest-green)
![Conda](https://img.shields.io/badge/Conda-Environment-brightgreen)

This repo for the paper: "APBC-RL: Adaptive Pedestrian Barrier Control Framework for Crowd Simulations Based on Reinforcement Learning".


---

## 📝 Overview

Crowd barriers are widely used to regulate pedestrian movement during large public events such as pilgrimages. However, fixed barrier layouts often create unnecessary stopping and local congestion even when they successfully control the overall crowd flow. Finding better barrier configurations is difficult because there is no dataset that provides the best barrier placement for a given crowd scene. We address this challenge by introducing a simulation-driven reinforcement learning framework that learns barrier configurations without requiring labelled training data. Starting from a segmented crowd scene we construct a realistic simulation environment that reproduces the observed pedestrian distribution. This environment is then used to train a Soft Actor-Critic (SAC) agent that continuously adjusts barrier positions to improve pedestrian movement. Unlike supervised approaches the proposed framework allows the agent to explore barrier configurations that have never been observed in real data. We evaluate the learned policy under unseen numbers of pedestrians. The learned policy consistently outperforms the existing manually designed barrier configuration by reducing pedestrian stops while maintaining effective flow regulation. These results demonstrate that combining computer vision crowd simulation and reinforcement learning provides an effective framework for optimizing crowd control infrastructure.

### ✨ Key Features

- **Modular reinforcement learning framework** – easily replace the RL algorithm (e.g., SAC, PPO, TD3, etc.).
- **Flexible crowd simulation** – supports different JuPedSim crowd models (currently CollisionFreeSpeedModelV2).
- **Adaptive multi-barrier control** – continuous optimization of multiple barrier pairs.
- **Scenario exploration** – configure agents, barriers, and simulation settings without changing the source code.
- **Sequential and parallel simulation pipelines** – accelerate RL training by running JuPedSim simulations in parallel across CPU cores while the RL policy trains on the GPU.
- **Built-in evaluation** – quantitative metrics, interactive HTML visualization, and qualitative analysis.
- **Generalization experiments** – evaluate trained policies on unseen crowd scenarios.

<div align="center">

<img src="rendered_outputs\Overview_method.png" width="900"/>

**Figure 1.** Overview of the proposed RL-based adaptive barrier control framework.

</div>

---


---

## Installation

Clone the repository

```bash
git clone https://github.com/USERNAME/RLForHajjProject.git
cd RLForHajjProject
```

### Create Conda environment

```bash
conda create -n rl_hajj python=3.11
conda activate rl_hajj
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training

Two training implementations are provided. 

Our experimental setup: 

The proposed framework was trained on a Paperspace cloud workstation with NVIDIA RTX A4000 GPUs (16 GB) and 12 CPU cores. Parallel crowd simulations were used during training, and 100 episodes were completed in approximately 4 h 44 min.
### Option 1 — Standard Training

Recommended for a single workstation.

Resources

- 1 CPU
- 1 GPU

```bash
cd runners
python train_RL.py
```

---

### Option 2 — Parallel Training

Designed for multi-core machines.

Resources

- 1 GPU
- N CPU cores

The environment rollouts are executed in parallel while the neural network is updated on a single GPU.

```bash
cd runners
python train_RL_parallel.py
```

---

## Evaluation

Evaluate a trained policy

```bash
cd runners
python evaluate_GT_parallel.py
python evaluate_policy_parallel.py
```

---

## Qualitative Results

The following animations compare different barrier control strategies under various crowd densities.

| Number of Agents |          Ground Truth Configuration          |                   RL Policy Configuration                   |
|:----------------:|:--------------------------------------------:|:-----------------------------------------------------------:|
|        10        | ![](rendered_outputs/gif/gt_10_agents.gif)  |        ![](rendered_outputs/gif/rl_agent_303_10.gif)        |
|       300        |![](rendered_outputs/gif/gt_300_agents.gif)  |       ![](rendered_outputs/gif/rl_agent_303_300.gif)        |
|       1500       |![](rendered_outputs/gif/gt_1500_agents.gif) |       ![](rendered_outputs/gif/rl_agent_303_1500.gif)       |

---

| Number of Agents |              All Open Barrier              |                 All Closed Barrier                  |
|:----------------:|:------------------------------------------:|:--------------------------------------------:|
|        10        |  ![](rendered_outputs/gif/allopen_10.gif)  |  ![](rendered_outputs/gif/allclosed_10.gif)  |
|       300        | ![](rendered_outputs/gif/allopen_300.gif)  | ![](rendered_outputs/gif/allclosed_300.gif)  |
|       1500       | ![](rendered_outputs/gif/allopen_1500.gif) | ![](rendered_outputs/gif/allclosed_1500.gif) |

---

## Paper

If you use this repository, please cite

```bibtex
Coming soon.
```

The accompanying short paper can be found here:

```
paper/RL_Crowd_Barrier_Control.pdf
```

---

## Contact

**Alhanouf Alolyan**

📧 Email: hano.alolyan@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/hano-alolyan

🐙 GitHub: https://github.com/HanoCat