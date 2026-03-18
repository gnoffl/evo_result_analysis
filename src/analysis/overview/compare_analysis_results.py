import matplotlib.pyplot as plt
import os
import argparse

FOLDERS = [
    ("analysis/multi_mutation_less_generations_unsorted", "multi_mutation", "less_generations", "unsorted", "normal_mutation_rate"),
    ("analysis/multi_mutation_less_generations_unsorted_lower_mutation_rate", "multi_mutation", "less_generations", "unsorted", "lower_mutation_rate"),
    ("analysis/multi_mutation_same_generation_unsorted", "multi_mutation", "same_generations", "unsorted", "normal_mutation_rate"),
    ("analysis/single_mutation_less_generations_unsorted", "single_mutation", "less_generations", "unsorted", "normal_mutation_rate"),
    ("analysis/single_mutation_less_generations_unsorted_lower_mutation_rate", "single_mutation", "less_generations", "unsorted", "lower_mutation_rate"),
    ("analysis/single_mutation_same_generation_sorted", "single_mutation", "same_generations", "sorted", "normal_mutation_rate"),
    ("analysis/single_mutation_same_generation_unsorted", "single_mutation", "same_generations", "unsorted", "normal_mutation_rate"),
    ("analysis/ara_msr_max_generations", "multi_mutation", "same_generations", "sorted", "normal_mutation_rate"),
]


def compare_conditions(index: int) -> None:
    plt.clf()
    groups = {}
    for folder, *args in FOLDERS:
        groups.setdefault(args[index], []).append(get_accuracy(folder))
    
    labels = []
    values = []
    
    for condition, accuracies in groups.items():
        # Create a bar plot for each condition
        labels.append(condition)
        values.append(sum(accuracies) / len(accuracies))  # Average accuracy for the condition
    plt.bar(labels, values, label=f"Condition {index + 1}")
    # set y-axis lower limit to min(values)
    plt.ylim(bottom=min(values) * 0.99, top=1.0)
    plt.legend()
    plt.xlabel("Conditions")
    plt.ylabel("Accuracy")
    plt.title("Comparison of Analysis Results")
    plt.savefig(f"analysis/comparison_results_{index}.png", bbox_inches='tight')

        


def get_accuracy(folder_path: str) -> float:
    summary_file = [file for file in os.listdir(folder_path) if file.startswith("summary") and file.endswith(".txt")][0]
    summary_file_path = os.path.join(folder_path, summary_file)
    with open(summary_file_path, 'r') as file:
        lines = file.readlines()
        fitness = float(lines[1].split(": ")[1].strip())
    return fitness


def compare_all_results():
    # create bar plot where each folder is represented by a single bar.
    # x-axis lables are the concatenated args of the folder
    # y-axis is the accuracy
    accuracies = []
    labels = []
    for folder, *args in FOLDERS:
        accuracy = get_accuracy(folder)
        accuracies.append(accuracy)
        labels.append(" ".join(args))
    # sort restults by concatenated args
    sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i])
    accuracies = [accuracies[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]

    # set scale such that minimal shown value is min(accuracies) - 0.1 and maximal shown value is max(accuracies) + 0.1
    plt.barh(labels, accuracies)
    plt.xlabel("Accuracy")
    plt.title("Comparison of Analysis Results")
    plt.xlim(min(accuracies) * 0.99, 1.0)
    plt.savefig("analysis/comparison_results_sorted.png", bbox_inches='tight')


def main():
    max_acc = 0
    max_args = ()
    for folder, *args in FOLDERS:
        accuracy = get_accuracy(folder)
        print(accuracy)
        if accuracy > max_acc:
            max_acc = accuracy
            max_args = args

    print(f"Max accuracy: {max_acc}")
    print(f"Args for max accuracy: {max_args}")
    compare_all_results()
    for i in range(4):
        compare_conditions(i)

if __name__ == "__main__":
    main()