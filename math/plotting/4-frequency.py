#!/usr/bin/env python3
"""Module that plots a histogram of student grades."""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Generate and plot a histogram of student grades for a project.

    - The x-axis is labeled 'Grades'
    - The y-axis is labeled 'Number of Students'
    - The x-axis has bins every 10 units
    - The bars are outlined in black
    - The title is 'Project A'
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    plt.figure(figsize=(6.4, 4.8))
    bins = np.arange(0, 110, 10)
    plt.hist(student_grades, bins=bins, edgecolor='black')
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.xticks(bins)
    plt.show()
