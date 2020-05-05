#!/usr/bin/python

"""
* @file Monitor.py
* @author Gustavo Diaz H.
* @date 20 Apr 2020
* @brief Plot handler
"""

import matplotlib.pyplot as plt

class Monitor(object):
    def __init__(self, x, y, title, ylabel, xlabel, scale = None, marker = None, sig_name = ["signal"]):
        self.x = x 
        self.y = y
        self.title = title
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.scale = scale
        self.marker = marker
        self.sig_name = sig_name

    def plot(self):
        plt.figure()
        plt.title(self.title)
        plt.ylabel(self.ylabel)
        plt.xlabel(self.xlabel)
        legend_handler = []
        if self.scale != None:
            plt.yscale(self.scale)
            plt.xscale(self.scale)
        for i in range(0, len(self.x)):
            if self.marker != None:
                a, = plt.plot(self.x[i], self.y[i], marker = self.marker[i], label=self.sig_name[i])
                legend_handler.append(a)
                plt.legend(handles=legend_handler)
            else:
                a, = plt.plot(self.x[i], self.y[i], label=self.sig_name[i])
                legend_handler.append(a)
                plt.legend(handles=legend_handler)
        plt.grid(which='both', axis='both')

    def show(self):
        plt.show()

if __name__ == '__main__':
    # TEST
    import numpy as np
    
    x = np.linspace(0, 10, 100)
    f = 5.0/10
    y = np.sin(2*np.pi*f*x) + 10
    monitor = Monitor([x], [y], "test function: sin()", "amplitud", "time", marker = "o")
    monitor.plot()
    monitor.show()