# -*- coding: utf-8 -*-

"""
* @file jacobians.py
* @author Gustavo Diaz H.
* @date 08 May 2020
* @brief Jacobians of sensors for the eskf filter update
"""

import numpy as np
from Library.math_sup.Quaternion import Quaternions

class Jacobians(object):
    def __init__(self):
        self.sensors = "magnetometer"

    def hx_mag(self, x, mr_33_i):
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
        q = Quaternions([x[0], x[1], x[2], x[3]])
        R = q.todcm()
        m_33_b = R.dot(mr_33_i)

        return m_33_b

    def Hx_mag(self, x, mr_33_i):
        """
        * Jacobian of the Magnetic Measurement Model function
        * Takes a state variable and returns the Jacobian of the
        * measurement model that would correspond to that state.
        *
        * @param x np.array(7) Nominal state
        * @param mr_33_i np.array(3) magnetic reference vector in inercial frame
        * @return H np.array((3,6)) Magnetic Jacobian Matrix H
        """
        q0 = x[3]
        q1 = x[0]
        q2 = x[1]
        q3 = x[2]
        Hq = np.zeros((3,4))
        Hq[0,0] = np.array([q0, q3, -q2]).dot(mr_33_i)
        Hq[0,1] = np.array([q1, q2, q3]).dot(mr_33_i)
        Hq[0,2] = np.array([-q2, q1, -q0]).dot(mr_33_i)
        Hq[0,3] = np.array([q3, q0, q1]).dot(mr_33_i)
        Hq[1,0] = np.array([-q3, q0, q1]).dot(mr_33_i)
        Hq[1,1] = np.array([q2, -q1, q0]).dot(mr_33_i)
        Hq[1,2] = np.array([q1, q2, q3]).dot(mr_33_i)
        Hq[1,3] = np.array([-q0, -q3, q2]).dot(mr_33_i)
        Hq[2,0] = np.array([q2, -q1, q0]).dot(mr_33_i)
        Hq[2,1] = np.array([q3, -q0, -q1]).dot(mr_33_i)
        Hq[2,2] = np.array([q0, q3, -q2]).dot(mr_33_i)
        Hq[2,3] = np.array([q1, q2, q3]).dot(mr_33_i)

        Qdth = np.array([[-q1, -q2, -q3],
        				 [q0, -q3, q2],
        				 [q3, q0, -q1],
        				 [-q2, q1, q0]])
        H = np.zeros((3,6))
        H[:,0:3] = Hq.dot(Qdth)

        return H