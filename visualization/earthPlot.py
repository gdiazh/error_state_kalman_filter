#!/usr/bin/python

"""
* @file earthPlot.py
* @author Gustavo Diaz H.
* @date 01 Jun 2020
* @brief Plot handler
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class EarthPlot(object):
    def __init__(self):
        self.R = 6371   #[Km]
        self.fig = plt.figure('EarthPlot')
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Make data
        self.u = np.linspace(0, 2 * np.pi, 100)
        self.v = np.linspace(0, np.pi, 100)
        self.x = self.R * np.outer(np.cos(self.u), np.sin(self.v))
        self.y = self.R * np.outer(np.sin(self.u), np.sin(self.v))
        self.z = self.R * np.outer(np.ones(np.size(self.u)), np.cos(self.v))

    def earth(self):
        #Specify some properties of the graph
        plt.grid(which='both', axis='both')
        self.ax.set_xlabel('R [Km] / B [40uT]')
        self.ax.set_ylabel('R [Km] / B [40uT]')
        self.ax.set_zlabel('R [Km] / B [40uT]')

        #Draw and label the X, Y and Z axes
        axis_scale = self.R*1.8
        x_axis_lim = np.array([0 , 1])*axis_scale
        y_axis_lim = np.array([0 , 1])*axis_scale
        z_axis_lim = np.array([0 , 1])*axis_scale
        self.ax.plot(x_axis_lim, [0, 0], [0, 0], 'black')
        self.ax.plot([0, 0], y_axis_lim, [0, 0], 'black')
        self.ax.plot([0, 0], [0, 0], z_axis_lim, 'black')
        self.ax.text(axis_scale, 0, 0, 'X', fontsize=12)
        self.ax.text(0, axis_scale, 0, 'Y', fontsize=12)
        self.ax.text(0, 0, axis_scale, 'Z', fontsize=12)
        self.ax.set_xlim((-axis_scale*1.2,axis_scale*1.2))
        self.ax.set_ylim((-axis_scale*1.2,axis_scale*1.2))
        self.ax.set_zlim((-axis_scale*1.2,axis_scale*1.2))

        # Plot the surface
        self.ax.plot_surface(self.x, self.y, self.z, color = 'green')

    def sun(self, x, y, z):
        r0 = np.linalg.norm([x[0], y[0], z[0]])
        x_u = (x[0]/r0)*self.R*3
        y_u = (y[0]/r0)*self.R*3
        z_u = (z[0]/r0)*self.R*3
        x_ = [0, x_u]
        y_ = [0, y_u]
        z_ = [0, z_u]
        # print("sun unit vector: ", [x_u, y_u, z_u])
        self.ax.plot(x_, y_, z_, 'orange')

    def orbit(self, x, y, z):
        self.earth()
        self.ax.plot(x, y, z, 'black')
        self.ax.text(x[0], y[0], z[0], 'r0', fontsize=12)
        self.ax.text(x[-1], y[-1], z[-1], 'rf', fontsize=12)

    def show(self):
        plt.show()

if __name__ == '__main__':
    # TEST
    monitor = EarthPlot()
    monitor.earth()
    monitor.show()