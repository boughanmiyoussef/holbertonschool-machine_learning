#!/usr/bin/env python3
"""This module plots exponential decay of C-14 and Ra-226."""

import numpy as np
import matplotlib.pyplot as plt


def two():
    """
    Plot exponential decay of C-14 and Ra-226.

    C-14 is plotted with a red dashed line.
    Ra-226 is plotted with a green solid line.
    The plot shows the fraction of each element remaining over time.
    """
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(x, y1, 'r--', label='C-14')
    plt.plot(x, y2, 'g-', label='Ra-226')

    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of Radioactive Elements')
    plt.xlim(0, 20000)
    plt.ylim(0, 1)
    plt.legend(loc='upper right')

    plt.show()
