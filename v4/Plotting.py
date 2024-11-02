# @title Packages

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# @title Train Plot 

data_frames = []
for i in range(1, 31):
    file_path = Path(f'/home/jonaskao/Code/v4/train_plot_seed_{i}.csv')
    df = pd.read_csv(file_path)
    df['videoname'] += f'_{i}'
    data_frames.append(df)

data = pd.concat(data_frames, ignore_index=True)

aggregated_data = data.groupby('videoname').agg(
    predicted_label_sum=('predicted_train_label', 'sum'),
    label=('label', 'first')
).reset_index()

label_data = {
    0: aggregated_data[aggregated_data['label'] == 0],
    1: aggregated_data[aggregated_data['label'] == 1]
}

label_counts = {
    label: df['predicted_label_sum'].value_counts().sort_index()
    for label, df in label_data.items()
}

plt.figure(figsize=(8, 4.5))

color_palette = ['#4c78a8', '#f58518'] 

for idx, (label, counts) in enumerate(label_counts.items()):
    plt.plot(
        counts.index, 
        counts.values, 
        marker='o', 
        markersize=3, 
        linewidth=1.3, 
        label=f'Label {label}',
        color=color_palette[idx]  
    )

plt.xlabel('Predicted Label Sum', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.title('Distribution of Predicted Label Sum in Training Set (30 Times)', fontweight='bold', fontsize=12)

plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

plt.legend(title="Labels", title_fontsize=10, fontsize=8, frameon=False)

plt.tight_layout()
plt.show()

# @title Test Plot 

data_frames = []
for i in range(1, 31):
    file_path = Path(f'/home/jonaskao/Code/v4/test_plot_seed_{i}.csv')
    df = pd.read_csv(file_path)
    df['videoname'] += f'_{i}'
    data_frames.append(df)

data = pd.concat(data_frames, ignore_index=True)

aggregated_data = data.groupby('videoname').agg(
    predicted_label_sum=('predicted_test_label', 'sum'),
    label=('label', 'first')
).reset_index()

label_data = {
    0: aggregated_data[aggregated_data['label'] == 0],
    1: aggregated_data[aggregated_data['label'] == 1]
}

label_counts = {
    label: df['predicted_label_sum'].value_counts().sort_index()
    for label, df in label_data.items()
}

plt.figure(figsize=(8, 4.5))

color_palette = ['#4c78a8', '#f58518'] 

for idx, (label, counts) in enumerate(label_counts.items()):
    plt.plot(
        counts.index, 
        counts.values, 
        marker='o', 
        markersize=3, 
        linewidth=1.3, 
        label=f'Label {label}',
        color=color_palette[idx]  
    )

plt.xlabel('Predicted Label Sum', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.title('Distribution of Predicted Label Sum in Testing Set (30 Times)', fontweight='bold', fontsize=12)

plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

plt.legend(title="Labels", title_fontsize=10, fontsize=8, frameon=False)

plt.tight_layout()
plt.show()
