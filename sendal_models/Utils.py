##
## Utils.py
##

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

def show_graph(real, pred, original):
    plot_figure = plt.figure(figsize=(9, 6))
    plot_rst = plot_figure.add_subplot(111)
    plot_rst.plot(real, label='Real')
    plot_rst.plot(pred, label='Predict')
    plot_rst.plot(original, label='Original', color='lightgray')
    plot_rst.legend()
    plt.show()
    
    
