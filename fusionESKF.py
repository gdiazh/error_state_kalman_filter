# -*- coding: utf-8 -*-

"""
* @file fusionESKF.py
* @author Gustavo Diaz H.
* @date 20 Apr 2020
* @brief Sensor Fusion Algorithm using Error State Kalman Filter

* This Algorithm fuses gyroscope and magnetic sensors data to estimate
* atitude on the quaternions space, based on Joan Sola's [1] work.
* [1] Joan Solà (2017). Quaternion Kinematics for the error-state Kalman filter
"""

from ESKF import ErrorStateKalmanFilter
from quaternions import *

class FusionEKF:
    def __init__(self, gyro_rate, mag_rate):
        self.eskf = ErrorStateKalmanFilter(dim_x=7, dim_dx=6, dim_u=3, dim_z=3)
        self.gyro_rate = gyro_rate          #[Hz]
        self.mag_rate = mag_rate            #[Hz]

    def hx(self, x):
        """
        * Magnetic Measurement Model
        * Takes a state variable and returns the magnetic
        * measurement that would correspond to that state.
        *
        * @param x np.array((7,1)) Nominal state
        * @return m_33_eci np.array((3,1)) magnetic measurement
        """
        qw = x[0]
        qx = x[1]
        qy = x[2]
        qz = x[3]
        m_33_eci = np.array([
        2*(qx*qy+qw*qz),
        qw**2-qx**2+qy**2-qz**2,
        2*(qy*qz-qw*qx)])

        return m_33_eci

    def Hx(self, x):
        """
        * Jacobian of the Magnetic Measurement Model function
        * Takes a state variable and returns the Jacobian of the
        * measurement model that would correspond to that state.
        *
        * @param x np.array((7,1)) Nominal state
        * @return H np.array((3,6)) Magnetic Jacobian Matrix H
        """

        qw = x[0]
        qx = x[1]
        qy = x[2]
        qz = x[3]

        H = np.array([
        [0, 2*qw*qx-2*qy*qz, qw**2-qx**2+qy**2-qz**2, 0, 0, 0],
        [2*qy*qz-2*qw*qx, 0, -2*qx*qy-2*qw*qz, 0, 0, 0],
        [-qw**2+qx**2-qy**2+qz**2, 2*qx*qy+2*qw*qz, 0, 0, 0, 0]
        ], dtype = float)
        #wolfram code: {{z, y, x, -w}, {w,-x, y, -z}, {x, w, z, y}}*{{-x, -y, -z}, {w,-z, y}, {z, w, -x}, {-y, x, w}}

        return H

    def eskf_process(self, gyroData, magData, dt, i):
        """
        * Iteration of the Fusion algorithm consist of predict of 
        * the nominal state, prediction of the error state, update
        * of the error estate and injection of error into nominal states
        *
        * @param gyroData np.array((3,1)) Gyroscope (x,y,z) noisy measurements
        * @param magData np.array((3,1)) Magnetometer (x,y,z) noisy measurements
        * @param dt Float Time step at which the algorithm runs
        * @param i Int Current iteration of the algorithm
        * @return self.eskf.q np.array((4,1)) Estimated quaternion
        """

        # Pridict state using gyro measurments
        u = np.zeros((3,1))
        u[:,0] = gyroData
        self.eskf.predict_nominal(u, dt)
        self.eskf.predict_error(u, dt)

        # Update state at magnetometer data rate
        if i%(self.gyro_rate/self.mag_rate) == 0:
            # self.eskf.eskfReset()
            z = np.zeros((3,1))
            z[:,0] = magData
            self.eskf.update(z, self.Hx, self.hx)
            # self.eskf.eskfReset()
        return self.eskf.q

if __name__ == '__main__':
    # TEST
    import numpy as np
    import matplotlib.pyplot as plt

    from gyroSim import gyroSim
    from magSim import magSim

    # Time parameters for simulation
    tf = 50
    dt = 0.01
    N = int(tf/dt)
    t = np.linspace(0, tf, N)

    # Gyro Signal parameters
    f_gyro = 5.0/10
    rate_true_Nx3_b = np.zeros((N,3))
    rate_true_Nx3_b[:,0] = np.zeros(N)
    rate_true_Nx3_b[:,1] = np.zeros(N)
    rate_true_Nx3_b[:,2] = np.pi*np.ones(N)
    
    # Gyro Model
    gyro = gyroSim(rw_std = 1e-4, rw_step = 0.01, rw_limit = 1e-3)
    
    # Gyro Data Storage
    rate_m_Nx3_b = np.zeros((N, 3))
    rw_cal_Nx3_b = np.zeros((N, 3))
    nrs_gen_Nx3_b = np.zeros((N, 3))

    # Magnetometer Signal parameters
    f_mag = 3.0/10
    mag_true_Nx3_b = np.zeros((N,3))
    mag_true_Nx3_b[:,0] = 1*np.sin(2*np.pi*f_mag*t+np.pi/18.0)
    mag_true_Nx3_b[:,1] = 1*np.cos(2*np.pi*f_mag*t-np.pi/18.0)
    mag_true_Nx3_b[:,2] = np.zeros(N)

    # Mag Model
    mag = magSim(nrs_std = 1e-4)
    
    # Mag Data Storage
    mag_m_Nx3_b = np.zeros((N, 3))
    nrs_gen_Nx3_b = np.zeros((N, 3))

    # Data Fusion Algoritm
    fusion = FusionEKF(gyro_rate = 100, mag_rate = 10)
    t_sim = 0

    # Fusion Data Storage
    q_estimates_Nx4_i2b = np.zeros((N,4))
    ypr_estimates_Nx3_i2b = np.zeros((N,3))
    hx_est_Nx3_b = np.zeros((N,3))
    dx_est_Nx6_b = np.zeros((N,6))
    P_est_Nx6x6_b = np.zeros((N,6,6))
    detP_est_Nx1_b = np.zeros((N,1))

    print("Running ESKF")
    for i in range(0, N):
        # Get sensor measurement
        rate_m_Nx3_b[i] = gyro.getMeasure(rate_true_Nx3_b[i])
        mag_m_Nx3_b[i] = mag.getMeasure(mag_true_Nx3_b[i])
        
        # Process data
        fusion.eskf_process(rate_m_Nx3_b[i], mag_m_Nx3_b[i], dt, i)
        q_estimates_Nx4_i2b[i] = fusion.eskf.q.T
        
        # Data of interest
        ypr_estimates_Nx3_i2b[i] = toEuler(q_estimates_Nx4_i2b[i]).T
        hx_est_Nx3_b[i] = fusion.hx(fusion.eskf.q).T
        dx_est_Nx6_b[i] = fusion.eskf.dx.T
        P_est_Nx6x6_b[i] = fusion.eskf.P
        detP_est_Nx1_b[i] = np.linalg.det(fusion.eskf.P)
        if 100*i/N%(5)== 0:
            print(100*i/N, "%...")
    print(100, "%")
    print("Test Finished")

    # Data Visualization
    from monitor import Monitor
    
    rates_mon = Monitor([t, t, t], [rate_m_Nx3_b[:,0], rate_m_Nx3_b[:,1], rate_m_Nx3_b[:,2]], "Gyro noisy measurements", "rate[rad/s]", "time[s]", sig_name = ["wx", "wy", "wz"])
    rates_mon.plot()

    mag_mon = Monitor([t, t, t], [mag_m_Nx3_b[:,0], mag_m_Nx3_b[:,1], mag_m_Nx3_b[:,2]], "Mag noisy measurements", "mag[]", "time[s]", sig_name = ["mx", "my", "mz"])
    mag_mon.plot()

    hx_mon = Monitor([t, t, t], [hx_est_Nx3_b[:,0], hx_est_Nx3_b[:,1], hx_est_Nx3_b[:,2]], "Mag vector estimate", "mag[]", "time[s]", sig_name = ["mx", "my", "mz"])
    hx_mon.plot()

    dx_mon = Monitor([t, t, t], [dx_est_Nx6_b[:,0], dx_est_Nx6_b[:,1], dx_est_Nx6_b[:,2]], "Error estimate", "dtheta[rad]", "time[s]", sig_name = ["dx", "dy", "dz"])
    dx_mon.plot()

    P_mon = Monitor([t, t, t], [P_est_Nx6x6_b[:,0, 0], P_est_Nx6x6_b[:,1, 1], P_est_Nx6x6_b[:,2, 2]], "Error covariance estimate", "[sigma2]", "time[s]", sig_name = ["Pxx", "Pyy", "Pzz"])
    P_mon.plot()

    detP_mon = Monitor([t], [detP_est_Nx1_b], "Det Error covariance estimate", "[sigma2]", "time[s]", sig_name = ["det(P)"])
    detP_mon.plot()

    q_mon = Monitor([t, t, t, t], [q_estimates_Nx4_i2b[:,0], q_estimates_Nx4_i2b[:,1], q_estimates_Nx4_i2b[:,2], q_estimates_Nx4_i2b[:,3]], "q estimates", "q", "time[s]", sig_name = ["qw", "qx", "qy", "qz"])
    q_mon.plot()
    q_mon.show()