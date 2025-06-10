import matplotlib.pyplot as plt
import seaborn as sns

def plot_intent_distribution(dataset, split_name="train"):
    split = dataset[split_name]  # Access the split explicitly

    intent_labels = split["intent"]
    label_names = split.features["intent"].names

    label_counts = {}
    for label in intent_labels:
        label_name = label_names[label]
        label_counts[label_name] = label_counts.get(label_name, 0) + 1

    sorted_items = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    labels, counts = zip(*sorted_items)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=counts, y=labels, palette="viridis")
    plt.title(f"Intent Distribution ({split_name})")
    plt.xlabel("Count")
    plt.ylabel("Intent")
    plt.tight_layout()
    plt.show()
