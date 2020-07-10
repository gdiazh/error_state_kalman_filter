# -*- coding: utf-8 -*-

"""
* @file ESKF.py
* @author Gustavo Diaz H.
* @date 20 Apr 2020
* @brief Error State Kalman Filter Algorithm

* The Algorithm propagates the nominal state and the error estate covariance
* using inertial measurements of angular rates and update the state with
* absolute mesurements of attitude when available. More details on [1].
* [1] Joan Solà (2017). Quaternion Kinematics for the error-state Kalman filter
"""

import numpy as np
from Library.math_sup.Quaternion import Quaternions

class ErrorStateKalmanFilter(object):
    def __init__(self, dim_x, dim_dx, dim_u, dim_z, inertia, invInertia):
        self.dim_x = dim_x
        self.dim_dx = dim_dx
        self.dim_u = dim_u
        self.dim_z = dim_z

        self.x = np.zeros(dim_x)            # state x = [q, wb] (7x1 vector)
        self.dx = np.zeros(dim_dx)          # error state dx = [dtheta, dwb] (6x1 vector)
        self.q = Quaternions([0, 0, 0, 1])	# states 1-4
        self.wb = np.zeros(3)               # states 5-7
        self.dtheta_mag = np.zeros(3)       # tmp magnetic error observation
        self.dtheta_css = np.zeros(3)       # tmp csun sensor error observation
        self.dtheta_fss = np.zeros(3)       # tmp fsun sensor error observation

        self.std_rn_w = 1e-3                # gyro noise standard deviation [rad/s]
        self.std_rw_w = 1e-3                # gyro random walk standard deviation [rad/s*s^0.5]
        self.std_rn_mag = 1e-4              # magnetometer noise standard deviation []

        self.P = np.eye(dim_dx)             # uncertainty covariance (6x6 matrix)
        self.F = np.eye(dim_dx)             # error state transition matrix (6x6 matrix)
        self.R = (self.std_rn_mag**2)*np.eye(dim_z) # measurement uncertainty (3x3 diagonal matrix)
        self.Q = (self.std_rn_w**2)*np.eye(dim_dx)  # process uncertainty (6x6 matrix)
        self.y = np.zeros(dim_z)            # residual (3x1 vector)

        self.K = np.zeros((dim_dx, dim_z))  # kalman gain (6x3 matrix)
        self.S = np.zeros((dim_z, dim_z))   # system uncertainty (3x3 matrix)
        self.SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty (3x3 matrix)

        self.I = np.eye(dim_dx)             # identity matrix (6x6 matrix)
        self.I33 = np.eye(3)                # identity (3x3 matrix)

        self.inertia = inertia              # satellite inertia matrix
        self.invInertia = invInertia        # inverse of satellite inertia matrix

    def predict_nominal(self, u, dt):
        """
        * Propagate the nominal state using inertial angular rate measurements
        * Nominal state is considered deterministic, hence no covariance matrix
        * is considered. The stochastic part is considered in the error state. 
        *
        * @param u np.array((3,1)) Angular rates (x,y,z) measurements
        * @param dt Float Time step at which the algorithm runs
        * @update self.x np.array((7,1)) Nominal state
        * @update self.q np.array((4,1)) Nominal quaternion
        * @update self.wb np.array((3,1)) Nominal gyro bias
        """
        self.integrate(self.q(), u-self.wb, dt, method = 'rk4-d') # quaternion integration

        # self.wb = self.wb
        self.x[0:4] = self.q()
        self.x[4:7] = self.wb

    def predict_error(self, u, dt):
        """
        * Propagate the error state using inertial angular rate measurements.
        * The error state is always zero in this step, because it's observed
        * just in the update step and then reset to zero. Hence just the
        * error state covarience matrix is propagated.
        *
        * @param u np.array((3,1)) Angular rates (x,y,z) measurements
        * @param dt Float Time step at which the algorithm runs
        * @update self.F np.array((6,6)) Error state transition matrix
        * @update self.P np.array((6,6)) Error covariance matrix
        """

        # Error state covariance prediction
        dq_wdt = Quaternions([(u-self.wb)*dt])
        Rwb = dq_wdt.todcm().transpose()
        F1 = np.append(Rwb, -self.I33*dt, axis = 1)
        F2 = np.append(np.zeros((3,3)), self.I33, axis = 1)
        self.F = np.append(F1, F2, axis = 0)
        self.updateQ(dt)
        self.P = self.F * self.P * self.F.T + self.Q

    def update(self, z, HJacobian, h, args, stype):
        """
        * Observe the error state and covariance via filter correction
        * based on absolute attitude measurements and update nominal state.
        *
        * @param z np.array((3,1)) absolute attitude measurements (x,y,z)
        * @param H method funtion to compute Magnetic Jacobian Matrix H
        * @param h method funtion to compute Magnetic measurements
        * @update self.x np.array((7,1)) Nominal state
        * @update self.q np.array((4,1)) Nominal quaternion
        * @update self.wb np.array((3,1)) Nominal gyro bias
        """

        # Kalman Gain
        H = HJacobian(self.x, args)        #(dim_z,dim_dx)
        PHt = self.P.dot(H.T)              #(dim_dx, dim_z)
        S = H.dot(PHt) + self.R            #(dim_z, dim_z)
        SI = np.linalg.inv(S)
        K = PHt.dot(SI)                    #(dim_dx, dim_z)
        
        # Error state update
        self.y = z-h(self.x, args)
        self.dx = K.dot(self.y)
        
        # Error covariance update
        self.P = (self.I-K.dot(H))*self.P

        # Auxiliar error state variables
        dtheta = self.dx[0:3]
        dwb = self.dx[3:6]
        dq = Quaternions([dtheta])

        # Injection of the observed error to the nominal state
        self.q = self.q * dq
        self.wb = self.wb + dwb
        self.x[0:4] = self.q()
        self.x[4:7] = self.wb

        if stype == "mag":
            self.dtheta_mag = dtheta
        elif stype == "css":
            self.dtheta_css = dtheta
        elif stype == "fss":
            self.dtheta_fss = dtheta
        else:
            raise Exception('[ESKF] Not implemented sensor')

    def reset(self):
        self.dx = np.zeros((self.dim_dx, 1)) # dx = 0
        G = np.eye(self.dim_dx)
        self.P = G*self.P*G.T                # P = G*P*G, G=I => P = P, eq285-286, pag.65

    def updateQ(self, dt):
        n = int(self.dim_dx/2)
        for i in range(n):
            self.Q[i,i] = (self.std_rn_w**2) * dt**2
        for i in range(n, self.dim_dx):
            self.Q[i,i] = (self.std_rw_w**2) * dt

    def dynamics(self, q, w):
        Omega = self.omega4kinematics(w)
        return 0.5*Omega.dot(q)

    def dynamics_and_kinematics(self, x):
        w_b_3x3 = x[0:3]
        q_i2b = x[3:]

        Omega_sk = self.skewsymmetricmatrix(w_b_3x3)
        Omega = self.omega4kinematics(w_b_3x3)

        h_total_b = self.inertia.dot(w_b_3x3)

        w_dot = -self.invInertia.dot(Omega_sk.dot(h_total_b))
        q_dot = 0.5*Omega.dot(q_i2b)

        x_dot = np.concatenate((w_dot, q_dot))
        return x_dot

    def rungeKuttaD(self, q, w, dt):
        """
        * Runge-Kutta method to integrate the quaternion kinematics differential equation
        *
        * @param q np.array(4) Actual quaternion state
        * @param w np.array(3) Actual angular rates
        * @update self.q Quaternion() Next quaternion state
        """
        k1 = self.dynamics(q, w)
        k2 = self.dynamics(q + 0.5*dt*k1, w)
        k3 = self.dynamics(q + 0.5*dt*k2, w)
        k4 = self.dynamics(q + dt*k3, w)

        q_next = q + dt*(k1 + 2*(k2+k3)+k4)/6.0

        self.q.setquaternion(q_next)
        self.q.normalize()

    def rungeKuttaDK(self, q, w, dt):
        """
        * Runge-Kutta method to integrate the quaternion kinematics differential equation
        * and the angular rates dynamic differential equation
        *
        * @param q np.array(4) Actual quaternion state
        * @param w np.array(3) Actual angular rates
        * @update self.q Quaternion() Next quaternion state
        """
        x = np.concatenate((w, q))
        k1 = self.dynamics_and_kinematics(x)
        k2 = self.dynamics_and_kinematics(x + 0.5*dt*k1)
        k3 = self.dynamics_and_kinematics(x + 0.5*dt*k2)
        k4 = self.dynamics_and_kinematics(x + dt*k3)

        x_next = x + dt*(k1 + 2*(k2+k3)+k4)/6.0

        self.q.setquaternion(x_next[3:])
        self.q.normalize()

    def integrate(self, q, w, dt, method = 'rk4-d'):
        if method == 'zoi': # zero order integrator
            self.q = self.q * Quaternions([w*dt])
        elif method == 'rk4-d': # 4th order runge-kutta just using kinematics
            self.rungeKuttaD(q, w, dt)
        elif method == 'rk4-dk': # 4th order runge-kutta using dynamics&kinematics
            self.rungeKuttaDK(q, w, dt)
        else:
            raise Exception('[ESKF] Not implemented integration method')

    def skewsymmetricmatrix(self, vector_3x3):
        Omega_sk = np.zeros((3, 3))
        Omega_sk[1, 0] = vector_3x3[2]
        Omega_sk[2, 0] = -vector_3x3[1]

        Omega_sk[0, 1] = -vector_3x3[2]
        Omega_sk[0, 2] = vector_3x3[1]

        Omega_sk[2, 1] = vector_3x3[0]
        Omega_sk[1, 2] = -vector_3x3[0]

        return Omega_sk

    def omega4kinematics(self, omega_3x3):
        Omega = np.zeros((4, 4))
        Omega[1, 0] = -omega_3x3[2]
        Omega[2, 0] = omega_3x3[1]
        Omega[3, 0] = -omega_3x3[0]

        Omega[0, 1] = omega_3x3[2]
        Omega[0, 2] = -omega_3x3[1]
        Omega[0, 3] = omega_3x3[0]

        Omega[1, 2] = omega_3x3[0]
        Omega[1, 3] = omega_3x3[1]

        Omega[2, 1] = -omega_3x3[0]
        Omega[2, 3] = omega_3x3[2]

        Omega[3, 1] = -omega_3x3[1]
        Omega[3, 2] = -omega_3x3[2]
        
        return Omega

    def get_ss_vect(self, I, I_max):
        Ix = max(I[0], I[1])
        sx = -2*np.argmax((I[0], I[1])) + 1
        Iy = max(I[2], I[3])
        sy = -2*np.argmax((I[2], I[3])) + 1
        Iz = max(I[4], 0)
        sz = -1

        noise_thr = 10  #[uA]

        if Ix<noise_thr and Iy<noise_thr and Iz<noise_thr:
            # shadow
            ss_unit_b = np.zeros(3)
        elif Iz < noise_thr:
            # ignore incomplete observations
            ss_unit_b = np.zeros(3)
        else:
            ss_unit_b = np.array([sx*Ix, sy*Iy, sz*Iz])/I_max
        
        return ss_unit_b

    def get_fss_vect(self, Vratio, calib):
        h = calib[0]
        x0 = calib[1]
        y0 = calib[2]
        T = calib[3]
        q_c2b = calib[4]

        if Vratio[0] == 0 and Vratio[1] == 0: #invalid measurement
            return np.array([0, 0, 0])

        xd_ = Vratio[0] + x0
        yd_ = Vratio[1] + y0

        [xd, yd] = T.dot(np.array([xd_, yd_]))

        phi = np.arctan2(xd, yd)
        theta = np.arctan2(np.sqrt(xd**2 + yd**2), h)

        rfss_c = np.array([np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])
        rfss_b = q_c2b.frame_conv(rfss_c, "q")
        return rfss_b

    def get_fss_qdvect(self, Vratio, calib):
        h = calib[0]
        x0 = calib[1]
        y0 = calib[2]
        T = calib[3]

        xd_ = Vratio[0] + x0
        yd_ = Vratio[1] + y0

        [xd, yd] = T.dot(np.array([xd_, yd_]))
        return np.array([xd, yd])
