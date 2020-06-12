#!/usr/bin/python

"""
* @file tools.py
* @author Gustavo Diaz H.
* @date 12 Jun 2020
* @brief Math tools
"""
import numpy as np
rad2deg = 180/np.pi

def Log(q):
    """
    * Logarithmic map for unit quaternions
    *
    * @param q np.array(4) quaternion
    * @return theta np.array(3) angle associated to the given quaternion
    """
    qw = q[0]
    qv = q[1:4]
    qv_norm = np.linalg.norm(qv)
    phi =2*np.arctan2(qv_norm, qw)
    u = qv / qv_norm
    return phi*u

def Log_vect(qx_i, qy_i, qz_i, qw_i):
    n = len(qx_i)
    theta_x_i_n = np.zeros(n)
    theta_y_i_n = np.zeros(n)
    theta_z_i_n = np.zeros(n)
    theta_norm_i_n = np.zeros(n)
    for i in range(0,n):
        qi = np.array([qw_i[i], qx_i[i], qy_i[i], qz_i[i]])
        thetai = Log(qi)
        theta_x_i_n[i] = thetai[0]
        theta_y_i_n[i] = thetai[1]
        theta_z_i_n[i] = thetai[2]
        theta_norm_i_n[i] = np.linalg.norm(thetai)

    return [theta_x_i_n, theta_y_i_n, theta_z_i_n, theta_norm_i_n]

def normalize(qx_i, qy_i, qz_i, qw_i):
    n = len(qx_i)
    qx_i_n = np.zeros(n)
    qy_i_n = np.zeros(n)
    qz_i_n = np.zeros(n)
    qw_i_n = np.zeros(n)
    for i in range(0,n):
        qn = np.sqrt(qx_i[i]**2 + qy_i[i]**2 + qz_i[i]**2 + qw_i[i]**2)
        qx_i_n[i] = qx_i[i] / qn
        qy_i_n[i] = qy_i[i] / qn
        qz_i_n[i] = qz_i[i] / qn
        qw_i_n[i] = qw_i[i] / qn
    return [qx_i_n, qy_i_n, qz_i_n, qw_i_n]

def multiply(px, py, pz, pw, qx, qy, qz, qw):
    n = len(px)
    rx = np.zeros(n)
    ry = np.zeros(n)
    rz = np.zeros(n)
    rw = np.zeros(n)
    for i in range(0,n):
        rx[i] = pw[i] * qx[i] - pz[i] * qy[i] + py[i] * qz[i] + px[i] * qw[i]
        ry[i] = pz[i] * qx[i] + pw[i] * qy[i] - px[i] * qz[i] + py[i] * qw[i]
        rz[i] = - py[i] * qx[i] + px[i] * qy[i] + pw[i] * qz[i] + pz[i] * qw[i]
        rw[i] = - px[i] * qx[i] - py[i] * qy[i] - pz[i] * qz[i] + pw[i] * qw[i]
    return [rx, ry, rz, rw]

if __name__ == '__main__':
    # TEST
    import pandas
    import matplotlib.pyplot as plt

    path = "../../../../Data/logs/"
    file = "yyyy-mm-dd hh-mm-ss.csv" # [test description]

    log_data = pandas.read_csv(path+file)

    time = log_data['time[sec]'].values
    q_t_i2b0 = log_data['q_t_i2b(0)[-]'].values
    q_t_i2b1 = log_data['q_t_i2b(1)[-]'].values
    q_t_i2b2 = log_data['q_t_i2b(2)[-]'].values
    q_t_i2b3 = log_data['q_t_i2b(3)[-]'].values
    q_estEskfTemp_i2b0 = log_data['ADCS_q_estEskfTemp_i2b(0)[-]'].values
    q_estEskfTemp_i2b1 = log_data['ADCS_q_estEskfTemp_i2b(1)[-]'].values
    q_estEskfTemp_i2b2 = log_data['ADCS_q_estEskfTemp_i2b(2)[-]'].values
    q_estEskfTemp_i2b3 = log_data['ADCS_q_estEskfTemp_i2b(3)[-]'].values

    [q_estErr_i2b0, q_estErr_i2b1, q_estErr_i2b2, q_estErr_i2b3] = multiply(q_t_i2b0, q_t_i2b1, q_t_i2b2, q_t_i2b3, -q_estEskfTemp_i2b0, -q_estEskfTemp_i2b1, -q_estEskfTemp_i2b2, q_estEskfTemp_i2b3)
    [theta_estErr_i2b0, theta_estErr_i2b1, theta_estErr_i2b2, theta_estErr_i2bn] = Log_vect(q_estErr_i2b0, q_estErr_i2b1, q_estErr_i2b2, q_estErr_i2b3)

    # Calcuate Statistic
    rmse = np.sqrt(np.mean(theta_estErr_i2bn**2))
    print("RMSE Attitude: ",rmse*rad2deg, "[deg]")

    # Data Visualization
    from monitor import Monitor

    q_estErr_i2b = Monitor([time, time, time, time], [q_estErr_i2b0, q_estErr_i2b1, q_estErr_i2b2, q_estErr_i2b3], "Quaternions Est Err I2B", "q[unit]", "time[s]", sig_name = ["qx","qy", "qz", "qw"])
    q_estErr_i2b.plot()

    theta_estErr_i2b =  Monitor([time, time, time, time], [theta_estErr_i2b0*rad2deg, theta_estErr_i2b1*rad2deg, theta_estErr_i2b2*rad2deg, theta_estErr_i2bn*rad2deg], "Attitude Est Err I2B", "angle[deg]", "time[s]", sig_name = ["tx","ty", "tz", "|t|"])
    theta_estErr_i2b.plot()
    theta_estErr_i2b.show()

