# coding=gbk
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


def scatter():
    n = 500
    x = np.random.randn(n)
    y = np.random.randn(n)
    plt.scatter(x, y, marker="x")
    plt.show()
    df = pd.DataFrame({"x": x, "y": y})
    sns.jointplot(x="x", y="y", data=df, kind="scatter")
    plt.show()


def lin_chart():
    x = []
    i = 1900
    while i <= 1910:
        x.append(i)
        i += 1
    y = [265, 323, 136, 220, 305, 350, 419, 450, 560, 720, 830]
    plt.plot(x, y)
    plt.show()
    df = pd.DataFrame({"x": x, "y": y})
    sns.lineplot(x="x", y="y", data=df)
    plt.show()


def bar_chart():
    x = ["c1", "c2", "c3", "c4"]
    y = [15, 18, 5, 26]
    plt.bar(x, y)
    plt.show()
    sns.barplot(x, y)
    plt.show()


def box_plots():
    data = np.random.normal(size=(10, 4))
    # np.random.randn(10, 4)
    labels = ["A", "B", "C", "D"]
    plt.boxplot(data, labels=labels)
    plt.show()
    df = pd.DataFrame(data, columns=labels)
    sns.boxplot(data=df)
    plt.show()


def pie_chart():
    nums = [25, 33, 37]
    labels = ["ADC", "APC", "TK"]
    plt.pie(x=nums, labels=labels)
    plt.show()


def thermodynamic():
    np.random.seed(33)
    data = np.random.rand(3, 3)
    sns.heatmap(data)
    plt.show()
