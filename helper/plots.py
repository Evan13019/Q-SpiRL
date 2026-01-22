
import os
import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, w=100):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_metrics(rewards, losses, final_dists, goal_flags, collision_flags, oob_flags, timeout_flags, static_collision_flags, dynamic_collision_flags, episode_lengths, plot_dir):
    episodes_arr = np.arange(len(rewards))
    # rewards
    plt.figure(figsize=(10, 4))
    plt.plot(episodes_arr, rewards, alpha=0.3, label="Reward")
    if len(rewards) > 100:
        ma = moving_average(rewards, w=100)
        plt.plot(np.arange(len(ma)), ma, label="Reward (MA100)")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Episode rewards")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "rewards.png"), dpi=150)
    plt.close()

    # loss
    plt.figure(figsize=(10, 4))
    plt.plot(episodes_arr, losses, alpha=0.7, label="Loss")
    if len(losses) > 100:
        ma = moving_average(losses, w=100)
        plt.plot(np.arange(len(ma)), ma, label="Loss (MA100)")
    plt.xlabel("Episode")
    plt.ylabel("Avg loss")
    plt.title("Episode loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "loss.png"), dpi=150)
    plt.close()

    # final dist
    plt.figure(figsize=(10, 4))
    plt.plot(episodes_arr, final_dists, alpha=0.7, label="Final dist")
    if len(final_dists) > 100:
        ma = moving_average(final_dists, w=100)
        plt.plot(np.arange(len(ma)), ma, label="Final dist (MA100)")
    plt.xlabel("Episode")
    plt.ylabel("Final distance")
    plt.title("Final distance to goal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "finaldist.png"), dpi=150)
    plt.close()

    # outcome rates
    w = 10
    plt.figure(figsize=(10, 4))
    goal_rate = moving_average(goal_flags, w=w)
    coll_rate = moving_average(collision_flags, w=w)
    oob_rate = moving_average(oob_flags, w=w)
    timeout_rate = moving_average(timeout_flags, w=w)
    x_ma = np.arange(len(goal_rate))
    plt.plot(x_ma, goal_rate, label="Goal rate")
    plt.plot(x_ma, coll_rate, label="Collision rate")
    plt.plot(x_ma, oob_rate, label="Out-of-bounds rate", linestyle="--")
    plt.plot(x_ma, timeout_rate, label="Timeout rate")
    plt.ylim(-0.05, 1.05)
    plt.xlabel(f"Episode (MA window={w})")
    plt.ylabel("Frequency")
    plt.title("Episode outcomes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "outcomes.png"), dpi=150)
    plt.close()

    # NEW: Collision breakdown
    plt.figure(figsize=(10, 4))
    static_coll_rate = moving_average(static_collision_flags, w=w)
    dynamic_coll_rate = moving_average(dynamic_collision_flags, w=w)
    x_ma = np.arange(len(static_coll_rate))
    plt.plot(x_ma, static_coll_rate, label="Static obstacle collisions")
    plt.plot(x_ma, dynamic_coll_rate, label="Dynamic obstacle collisions")
    plt.ylim(-0.05, 1.05)
    plt.xlabel(f"Episode (MA window={w})")
    plt.ylabel("Frequency")
    plt.title("Collision breakdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "collision_breakdown.png"), dpi=150)
    plt.close()

    # Episode lengths
    plt.figure(figsize=(10, 4))
    plt.plot(episodes_arr, episode_lengths, alpha=0.3, label="Episode length")
    if len(episode_lengths) > 100:
        ma = moving_average(episode_lengths, w=100)
        plt.plot(np.arange(len(ma)), ma, label="Length (MA100)")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Episode length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "episode_lengths.png"), dpi=150)
    plt.close()

    print(f"\n  [Plots] Saved to {plot_dir}")

print(f"{'='*110}\n")