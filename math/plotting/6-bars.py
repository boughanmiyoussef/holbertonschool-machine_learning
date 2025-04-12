#!/usr/bin/env python3
"""Module to plot a stacked bar graph of fruit quantities."""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Plot a stacked bar graph of fruit quantities per person."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    fruits = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    # Plot stacked bars
    bottom = np.zeros(3)
    for i, fruit_row in enumerate(fruit):
        plt.bar(people, fruit_row, width=0.5, label=fruits[i],
                color=colors[i], bottom=bottom)
        bottom += fruit_row

    # Set y-axis range and labels
    plt.ylim(0, 80)
    plt.ylabel('Quantity of Fruit')

    # Add title
    plt.title('Number of Fruit per Person')

    # Add legend
    plt.legend()

    # Display the plot
    plt.show()
