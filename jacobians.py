# -*- coding: utf-8 -*-

"""
* @file jacobians.py
* @author Gustavo Diaz H.
* @date 08 May 2020
* @brief Jacobians of sensors for the eskf filter update
"""

import numpy as np
from Library.math_sup.Quaternion import Quaternions
from Library.math_sup.tools_reference_frame import unitVector, skewMatrix

class Jacobians(object):
    def __init__(self):
        self.sensors = "magnetometer"

    def hx_mag(self, x, m_33_i):
        """
        * Magnetic Measurement Model
        * Takes a state variable and the actual reference
        * magnetic vector and returns the magnetic
        * measurement that would correspond to that state.
        *
        * @param x np.array(7) Nominal state
        * @param mr_33_i np.array(3) magnetic reference vector in inercial frame
        * @return m_33_b np.array((3,1)) magnetic measurement in body frame
        """
        mr_33_i = unitVector(m_33_i)
        q = Quaternions([x[0], x[1], x[2], x[3]])
        R = q.todcm()
        mr_33_b = R.dot(mr_33_i)

        return mr_33_b

    def Hx_mag(self, x, m_33_i):
        """
        * Jacobian of the Magnetic Measurement Model function
        * Takes a state variable and returns the Jacobian of the
        * measurement model that would correspond to that state.
        *
        * @param x np.array(7) Nominal state
        * @param m_33_i np.array(3) magnetic reference vector in inercial frame
        * @return H np.array((3,6)) Magnetic Jacobian Matrix H
        """
        mr_33_i = unitVector(m_33_i)
        q = Quaternions([x[0], x[1], x[2], x[3]])
        R = q.todcm()
        mr_33_b = R.dot(mr_33_i)
        H = np.zeros((3,6))
        H[:,0:3] = skewMatrix(mr_33_b)

        return H

    def hx_sun(self, x, s_33_i):
        """
        * Sun Vector Measurement Model
        * Takes a state variable and the actual reference
        * sun vector and returns the sun vector
        * measurement that would correspond to that state.
        *
        * @param x np.array(7) Nominal state
        * @param sr_33_i np.array(3) sun reference vector in inercial frame
        * @return s_33_b np.array((3,1)) sun vector measurement in body frame
        """
        sr_33_i = unitVector(s_33_i)
        q = Quaternions([x[0], x[1], x[2], x[3]])
        R = q.todcm()
        sr_33_b = R.dot(sr_33_i)

        return sr_33_b

    def Hx_sun(self, x, s_33_i):
        """
        * Jacobian of the Sun Measurement Model function
        * Takes a state variable and returns the Jacobian of the
        * measurement model that would correspond to that state.
        *
        * @param x np.array(7) Nominal state
        * @param s_33_i np.array(3) magnetic reference vector in inercial frame
        * @return H np.array((3,6)) Sun Jacobian Matrix H
        """
        sr_33_i = unitVector(s_33_i)
        q = Quaternions([x[0], x[1], x[2], x[3]])
        R = q.todcm()
        sr_33_b = R.dot(sr_33_i)
        H = np.zeros((3,6))
        H[:,0:3] = skewMatrix(sr_33_b)

        return H