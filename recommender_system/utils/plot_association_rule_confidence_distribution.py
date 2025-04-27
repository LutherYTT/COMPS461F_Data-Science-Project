import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_files = [
    './combined_csv_01.csv',
    './combined_csv_05.csv',
    './combined_csv_1.csv',
    './combined_csv_5.csv',
    './combined_csv_10.csv'
]
labels = ['0.1%', '0.5%', '1%', '5%', '10%']

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

bins = np.linspace(0, 1, 21)

colors = ['blue', 'orange', 'green', 'red', 'purple']

# Plot for Coarse_Class_Confidence
for file, label, color in zip(csv_files, labels, colors):
    df = pd.read_csv(file)
    counts, bin_edges = np.histogram(df['Coarse_Class_Confidence'], bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    axs[0].plot(bin_centers, counts, label=label, color=color, marker='o')

axs[0].set_title('Distribution of Coarse Class Confidence')
axs[0].set_xlabel('Confidence')
axs[0].set_ylabel('Frequency')
axs[0].legend(title='Noise Level')
axs[0].grid(True)

# Plot for Fine_Class_Confidence
for file, label, color in zip(csv_files, labels, colors):
    df = pd.read_csv(file)
    counts, bin_edges = np.histogram(df['Fine_Class_Confidence'], bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    axs[1].plot(bin_centers, counts, label=label, color=color, marker='o')

axs[1].set_title('Distribution of Fine Class Confidence')
axs[1].set_xlabel('Confidence')
axs[1].set_ylabel('Frequency')
axs[1].legend(title='Noise Level')
axs[1].grid(True)

plt.figtext(0.5, 0.95, 'Confidence Distributions for Different Noise Levels', ha='center', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('confidence_distributions_lines.png')
plt.show()
