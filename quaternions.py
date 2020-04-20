# -*- coding: utf-8 -*-

"""
* @file quaternions.py
* @author Gustavo Diaz H.
* @date 20 Apr 2020
* @brief Basic quaternion operation
"""

import numpy as np

def eulerRotation(v):
    phi = np.linalg.norm(v)
    if phi!=0:
        u = v/phi
    else:
        u = v
    return [u, phi]

def quaternionComposition(p, q):
    p = unit_quaternion(p)
    q = unit_quaternion(q)
    pw = p[0]
    px = p[1]
    py = p[2]
    pz = p[3]

    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    r0 = pw*qw - px*qx - py*qy - pz*qz
    r1 = pw*qx + px*qw + py*qz - pz*qy
    r2 = pw*qy - px*qz + py*qw + pz*qx
    r3 = pw*qz + px*qy - py*qx + pz*qw

    r = np.array([r0, r1, r2, r3])

    return unit_quaternion(r)

def unit_quaternion(q):
    return q/np.linalg.norm(q)

def skew(a):
    ax = a[0,0]
    ay = a[1,0]
    az = a[2,0]
    a_sk = np.array([
        [0, -az, ay],
        [az, 0, -ax],
        [-ay, ax, 0]])
    return a_sk

def toEuler(q):
    phi = np.arctan2(2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[1]**2+q[2]**2))
    theta = np.arcsin(2*(q[0]*q[2]+q[3]*q[1]))
    psi = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
    return np.array([[phi], [theta], [psi]])
