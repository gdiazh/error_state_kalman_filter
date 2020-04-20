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
from quaternions import *

class ErrorStateKalmanFilter(object):
    def __init__(self, dim_x, dim_dx, dim_u, dim_z):
        self.dim_x = dim_x
        self.dim_dx = dim_dx
        self.dim_u = dim_u
        self.dim_z = dim_z

        self.x = np.zeros((dim_x, 1))       # state x = [q, wb] (7x1 vector)
        self.dx = np.zeros((dim_dx, 1))     # error state dx = [dtheta, dwb] (6x1 vector)
        self.q = np.array([[1],[0],[0],[0]])# states 1-4
        self.wb = (1e-4)*np.ones((3, 1))    # states 5-7

        self.P = np.eye(dim_dx)             # uncertainty covariance (6x6 matrix)
        self.F = np.eye(dim_dx)             # error state transition matrix (6x6 matrix)
        self.R = (1e-4)*np.eye(dim_z)       # measurement uncertainty (3x3 diagonal matrix)
        self.Q = (1e-4)*np.eye(dim_dx)      # process uncertainty (6x6 matrix)
        self.y = np.zeros((dim_z, 1))       # residual (3x1 vector)

        self.K = np.zeros((dim_dx, dim_z))  # kalman gain (6x3 matrix)
        self.S = np.zeros((dim_z, dim_z))   # system uncertainty (3x3 matrix)
        self.SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty (3x3 matrix)

        self.I = np.eye(dim_dx)             # identity matrix (6x6 matrix)
        self.I33 = np.eye(3)                # identity (3x3 matrix)

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

        [v, phi] = eulerRotation((u-self.wb)*dt)    # rotation vector theta = v*phi
        dq_wdt = np.append([[np.cos(phi/2.0)]], v*np.sin(phi/2.0), axis = 0)
        self.q = quaternionComposition(self.q, dq_wdt)
        # self.wb = self.wb
        self.x[0:4] = self.q
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
        [v, phi] = eulerRotation((u-self.wb)*dt)    # rotation vector theta = v*phi
        U_skew = skew(v)
        Rwb = self.I33+np.sin(phi)*U_skew + (1-np.cos(phi))*U_skew*U_skew #Rodrigues rotation formula, eq.77, pag.18
        F1 = np.append(Rwb, -self.I33*dt, axis = 1)
        F2 = np.append(np.zeros((3,3)), self.I33, axis = 1)
        self.F = np.append(F1, F2, axis = 0)
        self.P = self.F * self.P * self.F.T + self.Q

    def update(self, z, HJacobian, h):
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
        H = HJacobian(self.x)              #(dim_z,dim_dx)
        PHt = self.P.dot(H.T)              #(dim_dx, dim_z)
        S = H.dot(PHt) + self.R            #(dim_z, dim_z)
        SI = np.linalg.inv(S)
        K = PHt.dot(SI)                    #(dim_dx, dim_z)
        
        # Error state update
        self.dx = K.dot(z-h(self.x))
        
        # Error covariance update
        self.P = (self.I-K.dot(H))*self.P

        # Auxiliar error state variables
        dtheta = self.dx[0:3]
        dwb = self.dx[3:6]
        [v, phi] = eulerRotation(dtheta)    # rotation vector theta = v*phi
        dq = np.append([[np.cos(phi/2.0)]], v*np.sin(phi/2.0), axis = 0)

        # Injection of the observed error to the nominal state
        self.q = quaternionComposition(self.q, dq)
        self.wb = self.wb + dwb
        self.x[0:4] = self.q
        self.x[4:7] = self.wb

    def eskfReset(self):
        self.dx = np.zeros((self.dim_dx, 1))     # dx = 0
        self.P = np.eye(self.dim_dx)             # P = G*P*G, G=I => P = P, eq285-286, pag.65