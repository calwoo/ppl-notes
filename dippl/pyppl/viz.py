"""
Visualization of standard random variables.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-pastel")

def viz(rv):
    # Run function repeatedly to collect samples.
    results = []
    for _ in range(1000):
        results.append(rv())
    # Organize samples by result.
    labels = list(set(results))
    counts = [0] * len(labels)
    for res in results:
        for jdx in range(len(counts)):
            if res == labels[jdx]:
                counts[jdx] += 1
    counts = list(map(lambda x: x/1000, counts))
    # Plot bar chart.
    plt.bar(labels, counts, align="center")
    plt.xticks(np.arange(len(labels)), labels)
    plt.show()

    