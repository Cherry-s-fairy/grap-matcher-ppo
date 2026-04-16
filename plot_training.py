import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False


def ema_smooth(values, alpha=0.05):
    """Exponential moving average smoothing."""
    smoothed = np.zeros_like(values, dtype=float)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed

results_dir = 'results'
seeds = [0, 1, 2, 3, 4]
methods = ['rl_node', 'rl_global']
colors = {'rl_node': '#1565C0', 'rl_global': '#C62828'}
labels = {'rl_node': 'RL-Node', 'rl_global': 'RL-Global'}

fig, ax = plt.subplots(figsize=(10, 5))

for method in methods:
    all_rewards = []
    steps = None
    for seed in seeds:
        path = f'{results_dir}/{method}_seed{seed}_trainlog.json'
        with open(path) as f:
            data = json.load(f)
        s = [d['step'] for d in data]
        r = [d['reward'] for d in data]
        if steps is None:
            steps = np.array(s)
        all_rewards.append(r)

    all_rewards = np.array(all_rewards)
    mean_r = all_rewards.mean(axis=0)
    smoothed = ema_smooth(mean_r, alpha=0.05)

    ax.plot(steps / 1e6, smoothed, color=colors[method], linewidth=2.2, label=labels[method])

ax.set_xlabel('Episode', fontsize=13)
ax.set_ylabel('Reward', fontsize=13)
ax.legend(fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.25, linestyle='--')
ax.set_xlim(0, 2.0)
plt.tight_layout()
plt.savefig('training_reward_curve.png', dpi=150, bbox_inches='tight')
print('Saved: training_reward_curve.png')
plt.show()
