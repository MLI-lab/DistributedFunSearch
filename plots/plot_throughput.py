"""Plot throughput (iterations/hour) vs number of samplers from W&B runs."""

import wandb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Configuration
WANDB_ENTITY = "franziweindel-technical-university-of-munich"
WANDB_PROJECT = "Throughput"

# Ratios to plot: (samplers, evaluators) pairs
# Each ratio (e.g., 1:5) means 1 sampler : 5 evaluators
# The multipliers are 1x, 2x, 3x, ... up to 10x
RATIOS = [
    (1, 5),   # 1:5 ratio
    (1, 10),  # 1:10 ratio
    (1, 15),  # 1:15 ratio
]

# Gray color for spines and ticks
GRAY = "#888888"

# Plot style settings with mathtext (LaTeX-like without requiring LaTeX installation)
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",  # Computer Modern (LaTeX-like)
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.formatter.use_mathtext": True,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.75,
    "axes.edgecolor": GRAY,
    "axes.labelcolor": "black",
    "xtick.color": GRAY,
    "ytick.color": GRAY,
    "lines.linewidth": 1.5,
    "figure.figsize": (8, 5),
})


def fetch_throughput_data():
    """Fetch throughput data from W&B runs."""
    api = wandb.Api()

    # Get all runs from the project
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")

    # Organize data by ratio
    data = {ratio: {"samplers": [], "mean": [], "std": []} for ratio in RATIOS}

    for run in runs:
        name = run.name

        # Parse run name: throughput_ratio_X_Y_TIMESTAMP where X:Y is the sampler:evaluator ratio
        # e.g., "throughput_ratio_2_10_20251211_151029" -> (2, 10)
        if not name.startswith("throughput_ratio_"):
            continue

        try:
            # Extract the ratio from the name
            # Remove prefix and split by underscore
            parts = name.replace("throughput_ratio_", "").split("_")
            # First two parts are samplers and evaluators
            if len(parts) >= 2:
                num_samplers = int(parts[0])
                num_evaluators = int(parts[1])

                # Determine which base ratio this belongs to
                for base_ratio in RATIOS:
                    base_s, base_e = base_ratio
                    # Check if this run matches the ratio (allowing for multipliers)
                    # e.g., (2, 10) matches base ratio (1, 5) because 2/10 = 1/5
                    if num_evaluators * base_s == num_samplers * base_e:
                        # Get metrics from summary
                        mean = run.summary.get("iterations_per_hour_mean")
                        std = run.summary.get("iterations_per_hour_std")

                        if mean is not None:
                            data[base_ratio]["samplers"].append(num_samplers)
                            data[base_ratio]["mean"].append(mean)
                            data[base_ratio]["std"].append(std if std is not None else 0)
                            print(f"  Found: {name} -> {num_samplers}:{num_evaluators} (ratio 1:{base_e})")
                        break
        except (ValueError, IndexError) as e:
            print(f"Skipping run {name}: {e}")
            continue

    # Sort data by number of samplers for each ratio
    for ratio in RATIOS:
        if data[ratio]["samplers"]:
            sorted_indices = np.argsort(data[ratio]["samplers"])
            data[ratio]["samplers"] = [data[ratio]["samplers"][i] for i in sorted_indices]
            data[ratio]["mean"] = [data[ratio]["mean"][i] for i in sorted_indices]
            data[ratio]["std"] = [data[ratio]["std"][i] for i in sorted_indices]

    return data


def plot_throughput(data, output_path="throughput.pdf"):
    """Create throughput plot with error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Colors from green to yellow to orange (matching reference plot)
    colors = [
        "#7F8002",  # dark olive green (for 1:5)
        "#86AE74",  # sage green (for 1:10)
        "#C3BE0F",  # yellow-green (for 1:15)
        "#FF954F",  # orange (for 1:20)
    ]
    markers = ["o", "s", "^", "D", "v"]

    for i, ratio in enumerate(RATIOS):
        if not data[ratio]["samplers"]:
            print(f"No data for ratio {ratio[0]}:{ratio[1]}")
            continue

        samplers = data[ratio]["samplers"]
        means = data[ratio]["mean"]
        stds = data[ratio]["std"]

        label = r"$1:{0}$".format(ratio[1])

        ax.errorbar(
            samplers, means, yerr=stds,
            label=label,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            markersize=8,
            capsize=4,
            capthick=1.5,
            linewidth=1.5,
            elinewidth=1.5,
        )

    ax.set_xlabel(r"Number of LLMs")
    ax.set_ylabel(r"Iterations per Hour")

    # Make tick labels black (while tick marks stay gray from rcParams)
    ax.tick_params(axis="both", colors=GRAY, labelcolor="black")

    # Set x-axis to show integer ticks from 1 to 10
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.5, 10.5)

    # Scientific notation for y-axis (×10^4)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: r"${:.0f}$".format(x / 1e4)))
    ax.annotate(r"$\times 10^4$", xy=(0, 1.02), xycoords="axes fraction",
                fontsize=10, ha="left", va="bottom", color="black")

    # Gray grid
    ax.grid(True, alpha=0.3, color=GRAY)

    # Legend with black text
    legend = ax.legend(loc="upper left", framealpha=0.9)
    for text in legend.get_texts():
        text.set_color("black")

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")

    plt.show()


def main():
    print("Fetching throughput data from W&B...")
    data = fetch_throughput_data()

    # Print summary
    for ratio in RATIOS:
        print(f"\nRatio 1:{ratio[1]}:")
        if data[ratio]["samplers"]:
            for s, m, std in zip(data[ratio]["samplers"], data[ratio]["mean"], data[ratio]["std"]):
                print(f"  {s} samplers: {m:.0f} ± {std:.0f} iter/hour")
        else:
            print("  No data found")

    print("\nGenerating plot...")
    plot_throughput(data, output_path="throughput.pdf")


if __name__ == "__main__":
    main()
