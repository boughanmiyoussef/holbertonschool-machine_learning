#!/usr/bin/env python3
"""Module to plot a histogram of student grades."""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Plot a histogram of student grades with bins every 10 units."""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    
    # Plot histogram with bins from 0 to 100, every 10 units
    bins = np.arange(0, 101, 10)
    plt.hist(student_grades, bins=bins, edgecolor='black')
    
    # Set x-axis and y-axis labels
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    
    # Set title
    plt.title('Project A')
    
    # Ensure x-axis ranges appropriately
    plt.xlim(0, 100)
    
    # Display the plot
    plt.show()
    