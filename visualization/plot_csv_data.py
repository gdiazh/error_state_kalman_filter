import pandas
import matplotlib.pyplot as plt
import numpy as np

path = "../../../../../Data/logs/"
# file = "yyyy-mm-dd hh-mm-ss.csv" # [test description]

log_data = pandas.read_csv(path+file)

time = log_data['time[sec]'].values
omega_t_bX = log_data["omega_t_b(X)[rad/s]"].values
omega_t_bY = log_data['omega_t_b(Y)[rad/s]'].values
omega_t_bZ = log_data['omega_t_b(Z)[rad/s]'].values
q_t_i2b0 = log_data['q_t_i2b(0)[-]'].values
q_t_i2b1 = log_data['q_t_i2b(1)[-]'].values
q_t_i2b2 = log_data['q_t_i2b(2)[-]'].values
q_t_i2b3 = log_data['q_t_i2b(3)[-]'].values
torque_t_bX = log_data['torque_t_b(X)[Nm]'].values
torque_t_bY = log_data['torque_t_b(Y)[Nm]'].values
torque_t_bZ = log_data['torque_t_b(Z)[Nm]'].values
h_total = log_data['h_total[Nms]'].values
sat_position_iX = log_data['sat_position_i(X)[m]'].values
sat_position_iY = log_data['sat_position_i(Y)[m]'].values
sat_position_iZ = log_data['sat_position_i(Z)[m]'].values
sat_velocity_iX = log_data['sat_velocity_i(X)[m/s]'].values
sat_velocity_iY = log_data['sat_velocity_i(Y)[m/s]'].values
sat_velocity_iZ = log_data['sat_velocity_i(Z)[m/s]'].values
sun_pos_iX = log_data['Sun_pos_i(X) [m]'].values
sun_pos_iY = log_data['Sun_pos_i(Y) [m]'].values
sun_pos_iZ = log_data['Sun_pos_i(Z) [m]'].values
lat = log_data['lat[rad]'].values
lon = log_data['lon[rad]'].values
alt = log_data['alt[m]'].values
RWModel_ADCS_bX = log_data['RWModel_ADCS_b(X)[Nm]'].values
RWModel_ADCS_bY = log_data['RWModel_ADCS_b(Y)[Nm]'].values
RWModel_ADCS_bZ = log_data['RWModel_ADCS_b(Z)[Nm]'].values
gyro_omega_ADCS_cX = log_data['gyro_omega_ADCS_c(X)[rad/s]'].values
gyro_omega_ADCS_cY = log_data['gyro_omega_ADCS_c(Y)[rad/s]'].values
gyro_omega_ADCS_cZ = log_data['gyro_omega_ADCS_c(Z)[rad/s]'].values
gyro_rw_ADCS_cX = log_data['gyro_rw_ADCS_c(X)[rad/s]'].values
gyro_rw_ADCS_cY = log_data['gyro_rw_ADCS_c(Y)[rad/s]'].values
gyro_rw_ADCS_cZ = log_data['gyro_rw_ADCS_c(Z)[rad/s]'].values
gyro_nr_ADCS_cX = log_data['gyro_nr_ADCS_c(X)[rad/s]'].values
gyro_nr_ADCS_cY = log_data['gyro_nr_ADCS_c(Y)[rad/s]'].values
gyro_nr_ADCS_cZ = log_data['gyro_nr_ADCS_c(Z)[rad/s]'].values
magnetometer_vect_ADCS_cX = log_data['magnetometer_vect_ADCS_c(X)[T?]'].values
magnetometer_vect_ADCS_cY = log_data['magnetometer_vect_ADCS_c(Y)[T?]'].values
magnetometer_vect_ADCS_cZ = log_data['magnetometer_vect_ADCS_c(Z)[T?]'].values
magnetometer_vect_ADCS_iX = log_data['magnetometer_vect_ADCS_i(X)[T?]'].values
magnetometer_vect_ADCS_iY = log_data['magnetometer_vect_ADCS_i(Y)[T?]'].values
magnetometer_vect_ADCS_iZ = log_data['magnetometer_vect_ADCS_i(Z)[T?]'].values
ss_I_1 = log_data['SS_(1)_IADCS_[uA]'].values
ss_I_2 = log_data['SS_(2)_IADCS_[uA]'].values
ss_I_3 = log_data['SS_(3)_IADCS_[uA]'].values
ss_I_4 = log_data['SS_(4)_IADCS_[uA]'].values
ss_I_5 = log_data['SS_(5)_IADCS_[uA]'].values
Control_X = log_data['Control_ADCS_b(X)[Nm]'].values
Control_Y = log_data['Control_ADCS_b(Y)[Nm]'].values
Control_Z = log_data['Control_ADCS_b(Z)[Nm]'].values
q_estEskfTemp_i2b0 = log_data['ADCS_q_estEskfTemp_i2b(0)[-]'].values
q_estEskfTemp_i2b1 = log_data['ADCS_q_estEskfTemp_i2b(1)[-]'].values
q_estEskfTemp_i2b2 = log_data['ADCS_q_estEskfTemp_i2b(2)[-]'].values
q_estEskfTemp_i2b3 = log_data['ADCS_q_estEskfTemp_i2b(3)[-]'].values

P_est00 = log_data['ADCS_P_est(0,0)[-]'].values
P_est11 = log_data['ADCS_P_est(1,1)[-]'].values
P_est22 = log_data['ADCS_P_est(2,2)[-]'].values
P_est_det = log_data['ADCS_P_est_det[-]'].values

eskfRes_X = log_data['ADCS_eskfRes(X)[-]'].values
eskfRes_Y = log_data['ADCS_eskfRes(Y)[-]'].values
eskfRes_Z = log_data['ADCS_eskfRes(Z)[-]'].values

GST = log_data['GST [rad]'].values

# Post processing
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

[q_estErr_i2b0, q_estErr_i2b1, q_estErr_i2b2, q_estErr_i2b3] = multiply(q_t_i2b0, q_t_i2b1, q_t_i2b2, q_t_i2b3, -q_estEskfTemp_i2b0, -q_estEskfTemp_i2b1, -q_estEskfTemp_i2b2, q_estEskfTemp_i2b3)

# Data Visualization
from monitor import Monitor
from earthPlot import EarthPlot

omega_gyro = Monitor([time, time, time], [gyro_omega_ADCS_cX, gyro_omega_ADCS_cY, gyro_omega_ADCS_cZ], "Gyro angular rates in BF", "rate[rad/s]", "time[s]", sig_name = ["wx", "wy", "wz"])
omega_gyro.plot()

rw_gyro = Monitor([time, time, time], [gyro_rw_ADCS_cX, gyro_rw_ADCS_cY, gyro_rw_ADCS_cZ], "Gyro random walk in GF", "random walk[rad/s]", "time[s]", sig_name = ["rx", "ry", "rz"])
rw_gyro.plot()

nr_gyro = Monitor([time, time, time], [gyro_nr_ADCS_cX, gyro_nr_ADCS_cY, gyro_nr_ADCS_cZ], "Gyro random noise in GF", "random noise[rad/s]", "time[s]", sig_name = ["nx", "ny", "nz"])
nr_gyro.plot()

mag_vect_c = Monitor([time, time, time], [magnetometer_vect_ADCS_cX, magnetometer_vect_ADCS_cY, magnetometer_vect_ADCS_cZ], "Magnetic vector in BF", "mag[T?]", "time[s]", sig_name = ["mx", "my", "mz"])
mag_vect_c.plot()

mag_vect_i = Monitor([time, time, time], [magnetometer_vect_ADCS_iX, magnetometer_vect_ADCS_iY, magnetometer_vect_ADCS_iZ], "Magnetic vector in IF", "mag[T?]", "time[s]", sig_name = ["mx", "my", "mz"])
mag_vect_i.plot()

ss_I = Monitor([time, time, time, time, time], [ss_I_1, ss_I_2, ss_I_3, ss_I_4, ss_I_5], "Coarse SS currents", "ss[uA]", "time[s]", sig_name = ["ss1", "ss2", "ss3", "ss4", "ss5"])
ss_I.plot()

q_t_i2b = Monitor([time, time, time, time], [q_t_i2b0, q_t_i2b1, q_t_i2b2, q_t_i2b3], "Quaternions I2B", "q[unit]", "time[s]", sig_name = ["qx","qy", "qz", "qw"])
q_t_i2b.plot()

torque_t_b = Monitor([time, time, time], [torque_t_bX, torque_t_bY, torque_t_bZ], "Torque in BF", "torque[Nm]", "time[s]", sig_name = ["tx", "ty", "tz"])
torque_t_b.plot()

h_total_ = Monitor([time], [h_total], "Total Angular Momentum in <>F", "Angular Momentum[Nms]", "time[s]", sig_name = ["h"])
h_total_.plot()

sat_position_i = Monitor([time, time, time], [sat_position_iX/1e3, sat_position_iY/1e3, sat_position_iZ/1e3], "Sat position in IF", "position[Km]", "time[s]", sig_name = ["rx", "ry", "rz"])
sat_position_i.plot()

sat_velocity_i = Monitor([time, time, time], [sat_velocity_iX, sat_velocity_iY, sat_velocity_iZ], "Sat velocity in IF", "position[m]", "time[s]", sig_name = ["vx", "vy", "vz"])
sat_velocity_i.plot()

sun_position_i = Monitor([time, time, time], [sun_pos_iX/1e3, sun_pos_iY/1e3, sun_pos_iZ/1e3], "Sun position in IF", "position[Km]", "time[s]", sig_name = ["sx", "sy", "sz"])
sun_position_i.plot()

RWModel_ADCS_b = Monitor([time, time, time], [RWModel_ADCS_bX, RWModel_ADCS_bY, RWModel_ADCS_bZ], "RW Torque in BF", "torque[Nm]", "time[s]", sig_name = ["tx", "ty", "tz"])
RWModel_ADCS_b.plot()

gyro_omega_ADCS_c = Monitor([time, time, time], [gyro_omega_ADCS_cX, gyro_omega_ADCS_cY, gyro_omega_ADCS_cZ], "Gyro rates in []F", "rates[rad/s]", "time[s]", sig_name = ["gx", "gy", "gz"])
gyro_omega_ADCS_c.plot()

Control_ = Monitor([time, time, time], [Control_X, Control_Y, Control_Z], "Control Torque in []F", "torque[Nm]", "time[s]", sig_name = ["tx", "ty", "tz"])
Control_.plot()

q_est_i2b = Monitor([time, time, time, time], [q_estEskfTemp_i2b0, q_estEskfTemp_i2b1, q_estEskfTemp_i2b2, q_estEskfTemp_i2b3], "Quaternions Est I2B", "q[unit]", "time[s]", sig_name = ["qx","qy", "qz", "qw"])
q_est_i2b.plot()

q_estErr_i2b = Monitor([time, time, time, time], [q_estErr_i2b0, q_estErr_i2b1, q_estErr_i2b2, q_estErr_i2b3], "Quaternions Est Err I2B", "q[unit]", "time[s]", sig_name = ["qx","qy", "qz", "qw"])
q_estErr_i2b.plot()

q_truest_i2b = Monitor([time, time, time, time, time, time, time, time], [q_t_i2b0, q_t_i2b1, q_t_i2b2, q_t_i2b3, q_estEskfTemp_i2b0, q_estEskfTemp_i2b1, q_estEskfTemp_i2b2, q_estEskfTemp_i2b3], "Quaternions True and Est I2B", "q[unit]", "time[s]", marker = "....****", sig_name = ["qx","qy", "qz", "qw", "qEx","qEy", "qEz", "qEw"])
q_truest_i2b.plot()

P_est = Monitor([time, time, time, time], [P_est00, P_est11, P_est22, P_est_det], "P Est", "[-]", "time[s]", sig_name = ["Pxx","Pyy", "Pzz", "P_det"])
P_est.plot()

eskfRes = Monitor([time, time, time], [eskfRes_X, eskfRes_Y, eskfRes_Z], "ESKF residual", "[-]", "time[s]", sig_name = ["Rx","Ry", "Rz"])
eskfRes.plot()

earth_m = EarthPlot()
earth_m.sun(sun_pos_iX, sun_pos_iY, sun_pos_iZ)
earth_m.orbit(sat_position_iX/1e3, sat_position_iY/1e3, sat_position_iZ/1e3)

omega_gyro.show()

