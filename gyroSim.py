# -*- coding: utf-8 -*-

"""
* @file gyroSim.py
* @author Gustavo Diaz H.
* @date 20 Apr 2020
* @brief Gyroscope measuremente error model

* The measurement model considered three types of errors, 
* an additive noise modeled as white gaussian noise
* a random walk which rate is modeled as white gaussian noise (solve using Runge-Kutta method)
* and a bias constant term.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

class gyroSim:
    def __init__(self, rw_std, rw_step, rw_limit):
        self.wm = np.zeros(3)
        self.bias = np.zeros(3)

        self.rw_std = rw_std                                #[rad/s]
        self.nrs = np.random.normal(0,self.rw_std, 3)       #[rad/s]
        self.rw_limit = rw_limit                            #[rad/s]
        self.rw_step = rw_step                              #[s]
        
        self.rn = np.random.normal(0,self.rw_std, 3)    #[rad/s]
        self.tn = np.zeros(3)                           #[s]

        self.last_time = time.time()
        self.newData = False

    def rwRate(self, tn, rn):
        """
        * The rate of change of the random walk process as funtion of time and the state.
        * It's modeled as white gausian noise in such a way that it stays in a given range
        * Mathematically: d(rw)/dt = rwRate(t, rw)  -->[dy/dt = f(t, y)]
        *
        * @param tn Float Time at which is the process is evaluated/sampled
        * @param rn Float Random walk state at wich evaluate the rate
        * @return rate Float Actual rate at given time and state
        """
        self.nrs = np.random.normal(0,self.rw_std, 3)  # update normal random sample
        rate = np.zeros(3)
        for i in range(0,3):
            if rn[i]>self.rw_limit:
                rate[i] = -abs(self.nrs[i])
            elif rn[i]<-self.rw_limit:
                rate[i] = abs(self.nrs[i])
            else:
                rate[i] = self.nrs[i]
        return rate

    def randomWalkRungeKutta(self):
        """
        * Runge-Kutta method to solve the random walk diferential equation
        *
        * @update self.rn np.array((3,1)) Actual value of the solved random walk process
        * @update self.tn np.array((3,1)) Time at which is solved the actual random walk process
        """
        k1 = self.rwRate(self.tn, self.rn)
        k2 = self.rwRate(self.tn + 0.5*self.rw_step, self.rn + 0.5*self.rw_step*k1)
        k3 = self.rwRate(self.tn + 0.5*self.rw_step, self.rn + 0.5*self.rw_step*k2)
        k4 = self.rwRate(self.tn + self.rw_step, self.rn + self.rw_step*k3)

        self.rn = self.rn + self.rw_step*(k1 + 2*(k2+k3)+k4)/6.0
        self.tn = self.tn + self.rw_step

    def getMeasure(self, true):
        """
        * Receive the true signal to add noise components.
        *
        * @param true np.array((3,1)) True value of gyroscope measurement
        * @return self.wm np.array((3,1)) Noisy measurement value
        """
        self.randomWalkRungeKutta()                         # update random walk
        self.nrs = np.random.normal(0,self.rw_std, 3)   # update normal random sample
        self.wm = true + 30*self.nrs + 1000*self.rn + self.bias
        return self.wm

    def getRW(self):
        return self.rn

    def getRNS(self):
        return self.nrs

    def getData(self, true):
        if self.newData:
            data_status = 1
            wm = self.getMeasure(true)  #return new measurement
        else:
            data_status = 0             #return last measurement
            wm = self.wm
        wm = np.append([[data_status]], self.wm)
        self.newData = False
        return wm

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
    rate_true_Nx3_b = np.zeros((N,3))
    rate_true_Nx3_b[:,0] = 0.1*np.sin(2*np.pi*f*t-2*np.pi/3) + 0.1
    rate_true_Nx3_b[:,1] = 0.1*np.sin(2*np.pi*f*t) + 0.1
    rate_true_Nx3_b[:,2] = 0.1*np.sin(2*np.pi*f*t+2*np.pi/3) + 0.1
    
    # Gyro Model
    gyro = gyroSim(rw_std = 1e-4, rw_step = 0.01, rw_limit = 1e-3)
    
    # Data Storage
    rate_m_Nx3_b = np.zeros((N, 3))
    rw_cal_Nx3_b = np.zeros((N, 3))
    nrs_gen_Nx3_b = np.zeros((N, 3))
    
    # Run simulation
    for i in range(0, N):
        rate_m_Nx3_b[i] = gyro.getMeasure(rate_true_Nx3_b[i])
        rw_cal_Nx3_b[i] = gyro.getRW()
        nrs_gen_Nx3_b[i] = gyro.getRNS()
    
    # Data Visualization
    from monitor import Monitor
    rates_mon = Monitor([t, t, t], [rate_m_Nx3_b[:,0], rate_m_Nx3_b[:,1], rate_m_Nx3_b[:,2]], "Gyro noisy measurements", "rate[rad/s]", "time[s]", sig_name = ["wx", "wy", "wz"])
    rates_mon.plot()

    rw_mon = Monitor([t, t, t], [rw_cal_Nx3_b[:,0], rw_cal_Nx3_b[:,1], rw_cal_Nx3_b[:,2]], "Gyro calculated random walk", "rate[rad/s]", "time[s]", sig_name = ["rx", "ry", "rz"])
    rw_mon.plot()

    nrs_mon = Monitor([t, t, t], [nrs_gen_Nx3_b[:,0], nrs_gen_Nx3_b[:,1], nrs_gen_Nx3_b[:,2]], "Gyro generated random sample", "rate[rad/s]", "time[s]", sig_name = ["nx", "ny", "nz"])
    nrs_mon.plot()

    nrs_mon.show()