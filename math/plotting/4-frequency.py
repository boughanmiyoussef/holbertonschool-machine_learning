#!/usr/bin/env python3
"""Plotting a histogram of student grades for Project A."""

import numpy as np
import matplotlib.pyplot as plt


def main():
    """
    Generate random student grades and plot a histogram.

    - Title: 'Project A'
    - X-axis: 'Grades'
    - Y-axis: 'Number of Students'
    - Bins every 10 units
    - Bars outlined in black
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    bins = np.arange(0, 110, 10)
    plt.hist(student_grades, bins=bins, edgecolor='black')
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.xticks(bins)
    plt.show()


if __name__ == "__main__":
    main()
