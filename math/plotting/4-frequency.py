#!/usr/bin/env python3
"""Plot a histogram of student grades."""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Plot histogram of student grades with black edges."""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    
    # Create bins from 0 to 100 in steps of 10
    bins = np.arange(0, 101, 10)
    
    # Plot histogram with black edges
    plt.hist(student_grades, bins=bins, edgecolor='black')
    
    # Set labels and title
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    
    # Set x-axis limits to match bins
    plt.xlim(0, 100)
    
    plt.show()
    