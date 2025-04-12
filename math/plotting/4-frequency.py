#!/usr/bin/env python3
"""Module to plot a histogram of student grades."""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Plot a histogram of student grades with bins every 10 units."""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    
    # Define bins for 0-10, 10-20, ..., 90-100
    bins = np.arange(0, 101, 10)
    
    # Plot histogram with black-outlined bars
    plt.hist(student_grades, bins=bins, edgecolor='black')
    
    # Set axis labels
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    
    # Set title
    plt.title('Project A')
    
    # Set x-axis range and ticks
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 101, 10))
    
    # Adjust layout to match default rendering
    plt.tight_layout()
    
    # Display the plot
    plt.show()