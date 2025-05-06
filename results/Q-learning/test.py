import pickle
import csv
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":

    # === Paths ===
    run_dir = "./"  # Change this if your files are elsewhere
    q_table_path = os.path.join(run_dir, "q_table.pkl")
    metrics_path = os.path.join(run_dir, "metrics.csv")

    # === Load Q-table ===
    with open(q_table_path, "rb") as f:
        Q = pickle.load(f)

    print(f"[✓] Loaded Q-table with {len(Q)} entries")

    # === Load win rate metrics ===
    episodes = []
    win_rates = []

    with open(metrics_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            win_rates.append(float(row["win_rate"]))

    # === Plot ===
    plt.figure()
    plt.plot(episodes, win_rates, alpha=0.4, label="Raw Win Rate")

    if len(win_rates) >= 5:
        smoothed = [sum(win_rates[i:i+5])/5 for i in range(len(win_rates)-4)]
        plt.plot(episodes[4:], smoothed, linewidth=2, label="5-Eval MA")

    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("Q-Learning Win Rate Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
