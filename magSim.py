# -*- coding: utf-8 -*-

"""
* @file magSim.py
* @author Gustavo Diaz H.
* @date 20 Apr 2020
* @brief Magnetometer measuremente error model

* The measurement model considered two types of errors, 
* an additive noise modeled as white gaussian noise
* and a bias constant term.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

class magSim2:
    def __init__(self, nrs_std):
        self.magm = np.zeros(3)
        self.bias = np.zeros(3)

        self.nrs_std = nrs_std
        self.nrs = np.random.normal(0,nrs_std, 3)   #[]

        self.last_time = time.time()
        self.newData = False

    def getMeasure(self, true):
        """
        * Receive the true signal to add noise components.
        *
        * @param true np.array((3,1)) True value of magnetometer measurement
        * @return self.magm np.array((3,1)) Noisy measurement value
        """
        self.nrs = np.random.normal(0,self.nrs_std, 3)      # update normal random sample
        self.magm = true + 30*self.nrs + self.bias
        return self.magm

    def getRNS(self):
        return self.nrs

    def getData(self, true):
        if self.newData:
            data_status = 1
            magm = self.getMeasure(true)  #return new measurement
        else:
            data_status = 0             #return last measurement
            magm = self.magm
        magm = np.append([[data_status]], self.magm)
        self.newData = False
        return magm

    def update(self, rate):
        if(time.time()-self.last_time>=rate):
            self.newData = True
            self.last_time = time.time()

if __name__ == '__main__':
    # TEST

    # Time parameters for simulation
    tf = 10
    dt = 0.01
    N = int(tf/dt)
    t = np.linspace(0, tf, N)
    
    # Signal parameters
    f = 5.0/10
    mag_true_Nx3_b = np.zeros((N,3))
    mag_true_Nx3_b[:,0] = 1*np.sin(2*np.pi*f*t)
    mag_true_Nx3_b[:,1] = 1*np.cos(2*np.pi*f*t)
    mag_true_Nx3_b[:,2] = np.zeros(N)
    
    # Mag Model
    mag = magSim2(nrs_std = 1e-4)
    
    # Data Storage
    mag_m_Nx3_b = np.zeros((N, 3))
    nrs_gen_Nx3_b = np.zeros((N, 3))
    
    # Run simulation
    for i in range(0, N):
        mag_m_Nx3_b[i] = mag.getMeasure(mag_true_Nx3_b[i])
        nrs_gen_Nx3_b[i] = mag.getRNS()
    
    # Data Visualization
    from monitor import Monitor
    mag_mon = Monitor([t, t, t], [mag_m_Nx3_b[:,0], mag_m_Nx3_b[:,1], mag_m_Nx3_b[:,2]], "Mag noisy measurements", "mag[]", "time[s]", sig_name = ["mx", "my", "mz"])
    mag_mon.plot()

    nrs_mon = Monitor([t, t, t], [nrs_gen_Nx3_b[:,0], nrs_gen_Nx3_b[:,1], nrs_gen_Nx3_b[:,2]], "Mag generated random sample", "mag[]", "time[s]", sig_name = ["nx", "ny", "nz"])
    nrs_mon.plot()

    nrs_mon.show()