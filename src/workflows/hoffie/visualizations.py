import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
import json


def plot_ubi1_optimization_results():
    values = [0.999738872051239, 0.9341765642166138, 0.8823096752166748, 0.8051742315292358, 0.04912063106894493]
    runs = ["max\n(0.9997)", "max_nat\n(0.9342)", "reference\n(0.8823)", "min_nat\n(0.8052)", "min\n(0.0491)"]
    # bar plot showing the values for each run, maximizations in green, the reference in blue, and minimizations in red
    # all labels need to be bigger
    colors = ["green", "green", "blue", "red", "red"]
    plt.figure(figsize=(8,6))
    plt.bar(runs, values, color=colors)
    plt.ylabel("Predicted Expression Level", fontsize=14)
    plt.title("Optimized Expression Levels for T03", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # add a legend, that explains the colors, with green for maximizations, blue for reference, and red for minimizations
    legend_elements = [Patch(facecolor='green', label='Maximizations'),
                          Patch(facecolor='blue', label='Reference'),
                          Patch(facecolor='red', label='Minimizations')]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    plt.savefig(os.path.join(os.path.dirname(__file__), "data", "ubi", "optimization_results.png"), bbox_inches='tight')

def plot_ZmEXPA6_optimization_results():
    values = [0.13099, 0.25229, 0.99391]
    runs = ["reference\n(0.1310)", "max_nat\n(0.2523)", "max\n(0.9939)"]
    colors = ["blue", "green", "green"]
    plt.figure(figsize=(8,6))
    plt.bar(runs, values, color=colors)
    plt.ylabel("Predicted Expression Level", fontsize=14)
    plt.title("Optimized Expression Levels for ZmEXPA6", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    legend_elements = [Patch(facecolor='green', label='Maximizations'),
                          Patch(facecolor='blue', label='Reference')]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=12)
    plt.savefig(os.path.join(os.path.dirname(__file__), "data", "zea", "ZmEXPA6_optimization_results.png"), bbox_inches='tight')


if __name__ == "__main__":
    # plot_ubi1_optimization_results()
    plot_ZmEXPA6_optimization_results()