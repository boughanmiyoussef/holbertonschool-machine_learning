#!/usr/bin/env python3
"""Plot a histogram showing frequency of student grades"""
import numpy as np
import matplotlib.pyplot as plt

def frequency():
    """Plots a histogram of student grades with bins of 10"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    bins = range(0, 101, 10)
    plt.hist(student_grades, bins=bins, edgecolor='black')
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.xticks(bins)
    plt.show()
