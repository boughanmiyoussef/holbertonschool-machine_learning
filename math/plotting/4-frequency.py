#!/usr/bin/env python3
"""Plot a histogram of student grades."""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Plot histogram of student grades with black edges."""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.show()
    