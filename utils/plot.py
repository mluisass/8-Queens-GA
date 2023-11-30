import numpy as np
import matplotlib.pyplot as plt

def plot_graph(y, title, x_label, y_label):
    mean = np.mean(y)
    std = np.std(y)

    plt.figure(figsize=(12, 4))
    plt.plot(y, color='blue', linewidth=2)
    plt.axhline(y=mean, color='r', linestyle='--')
    plt.axhline(y=mean + std, color='r', linestyle='--')
    plt.axhline(y=mean - std, color='r', linestyle='--')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()