# @title Packages

import pandas as pd  # type: ignore
import matplotlib.pyplot as plt # type: ignore
from pathlib import Path

# @title Train Plot

data_frames = []
for i in range(1, 51):
    file_path = Path(f'/home/jonaskao/Code/v4/train_plot_seed_{i}.csv')
    df = pd.read_csv(file_path)
    df['videoname'] += f'_{i}'
    data_frames.append(df)

data = pd.concat(data_frames, ignore_index=True)

aggregated_data = data.groupby('videoname').agg(
    predicted_label_sum=('predicted_train_label', 'sum'),
    label=('label', 'first'),
    total_count=('predicted_train_label', 'size')  
).reset_index()

aggregated_data['normalized_predicted_label_sum'] = aggregated_data['predicted_label_sum'] / aggregated_data['total_count']

aggregated_data['binned_sum'] = pd.cut(aggregated_data['normalized_predicted_label_sum'], bins=[i * 0.05 for i in range(21)], labels=[f'{i*5}-{(i+1)*5}%' for i in range(20)], include_lowest=True)

label_data = {
    0: aggregated_data[aggregated_data['label'] == 0],
    1: aggregated_data[aggregated_data['label'] == 1]
}

label_counts = {
    label: (df['binned_sum'].value_counts(normalize=True).sort_index() * 100)  # Convert counts to percentages
    for label, df in label_data.items()
}

plt.figure(figsize=(8, 4.5))
color_palette = ['#4c78a8', '#f58518'] 

for idx, (label, counts) in enumerate(label_counts.items()):
    plt.plot(
        counts.index, 
        counts.values, 
        marker='o', 
        markersize=5, 
        linewidth=1.5, 
        label=f'Label {label}',
        color=color_palette[idx]
    )

plt.xlabel('Normalized Predicted Label Sum Binned (Predicted Label Sum / Total Count)', fontsize=10)
plt.ylabel('Percentage (%)', fontsize=10)
plt.title('Distribution of Binned Normalized Predicted Label Sum in Training Set (50 Times)', fontweight='bold', fontsize=12)

plt.xticks(rotation=30, fontsize=8)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

plt.ylim(-2, 102)
plt.legend(title="Labels", title_fontsize=10, fontsize=8, frameon=False, loc="upper center")  # Place legend in upper right

plt.tight_layout()
plt.show()

# @title Test Plot

data_frames = []
for i in range(1, 51):
    file_path = Path(f'/home/jonaskao/Code/v4/test_plot_seed_{i}.csv')
    df = pd.read_csv(file_path)
    df['videoname'] += f'_{i}'
    data_frames.append(df)

data = pd.concat(data_frames, ignore_index=True)

aggregated_data = data.groupby('videoname').agg(
    predicted_label_sum=('predicted_test_label', 'sum'),
    label=('label', 'first'),
    total_count=('predicted_test_label', 'size')  
).reset_index()

aggregated_data['normalized_predicted_label_sum'] = aggregated_data['predicted_label_sum'] / aggregated_data['total_count']

aggregated_data['binned_sum'] = pd.cut(aggregated_data['normalized_predicted_label_sum'], bins=[i * 0.05 for i in range(21)], labels=[f'{i*5}-{(i+1)*5}%' for i in range(20)], include_lowest=True)

label_data = {
    0: aggregated_data[aggregated_data['label'] == 0],
    1: aggregated_data[aggregated_data['label'] == 1]
}

label_counts = {
    label: (df['binned_sum'].value_counts(normalize=True).sort_index() * 100)  # Convert counts to percentages
    for label, df in label_data.items()
}

plt.figure(figsize=(8, 4.5))
color_palette = ['#4c78a8', '#f58518'] 

for idx, (label, counts) in enumerate(label_counts.items()):
    plt.plot(
        counts.index, 
        counts.values, 
        marker='o', 
        markersize=5, 
        linewidth=1.5, 
        label=f'Label {label}',
        color=color_palette[idx]
    )

plt.xlabel('Normalized Predicted Label Sum Binned (Predicted Label Sum / Total Count)', fontsize=10)
plt.ylabel('Percentage (%)', fontsize=10)
plt.title('Distribution of Binned Normalized Predicted Label Sum in Testing Set (50 Times)', fontweight='bold', fontsize=12)

plt.xticks(rotation=30, fontsize=8)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

plt.ylim(-2, 27)
plt.legend(title="Labels", title_fontsize=10, fontsize=8, frameon=False, loc="upper center")

plt.tight_layout()
plt.show()
