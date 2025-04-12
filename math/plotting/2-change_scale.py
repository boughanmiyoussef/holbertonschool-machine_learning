#!/usr/bin/env python3
"""Module to plot exponential decay of C-14 with logarithmic y-axis."""

import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """Plot exponential decay of C-14 with logarithmic y-axis scaling."""
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(x, y)

    plt.yscale('log')

    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of C-14')

    plt.xlim(0, 28650)

    plt.show()
