import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import random 
from scipy import stats as st

GAMMA = 0.5 # m/pulses
ALPHA = 1.0 # m/pulses
BETA = 0.315 # rad/pulses
DT = 0.1 # The timestep size

N = 7 # Number of states

Kalman = True
Particle = False


def load_data(filename):
    """Load data from the csv log

    Parameters:
    filename (str)  -- the name of the csv log

    Returns:
    data (dict)     -- the logged data with data categories as keys
                       and values list of floats
    """
    
    f = open(filename + ".csv")

    file_reader = csv.reader(f, delimiter=',')

    # Load data into dictionary with headers as keys
    data = {}
    header = ["Time Stamp", "pUp", "pLeft", "pRight", "X", "Y", "Z", "Theta"]
    for h in header:
        data[h] = []

    row_num = 0
    f_log = open("bad_data_log.txt", "w")
    for row in file_reader:
        for h, element in zip(header, row):
            # If got a bad value just use the previous value
            try:
                data[h].append(float(element))
            except ValueError:
                data[h].append(data[h][-1])
                f_log.write(str(row_num) + "\n")

        row_num += 1
    f.close()
    f_log.close()

    return data

#**********General Helpers**************#

def wrap_to_pi(angle):
    """Wrap angle data in radians to [-pi, pi]

    Parameters:
    angle (float)   -- unwrapped angle

    Returns:
    angle (float)   -- wrapped angle
    """
    wAngle = (angle % np.pi) % (2 * np.pi) - np.pi

    return wAngle

#**********Kalman Filter Helpers**************#

def propogateStateKF(x_t_prev, u_t):
    """Propogate/predict the state based on chosen motion model

    Parameters:
    x_t_prev (np.array)  -- the previous state estimate
    u_t (np.array)       -- the current control input

    Returns:
    x_bar_t (np.array)   -- the predicted state
    """
    xp = x_t_prev[0]
    yp = x_t_prev[1]
    zp = x_t_prev[2]
    thetap = x_t_prev[3]
    vp = x_t_prev[4]
    sp = x_t_prev[5]
    wp = x_t_prev[6]

    uUpT = u_t[0]
    uLeftT = u_t[1]
    uRightT = u_t[2]

    x = xp + vp*DT*math.cos(thetap)
    y = yp + vp*DT*math.sin(thetap)
    z = zp + sp*DT
    theta = thetap + wp*DT
    v = ALPHA*(uLeftT + uRightT)
    s = GAMMA*(uUpT)
    w = BETA*(-uLeftT + uRightT)

    x_bar_t = np.array([x, y, z, theta, v, s, w])

    return x_bar_t

def calcPropJacX(x_t_prev, u_t):
    """Calculate the Jacobian of your motion model with respect to state

    Parameters:
    x_t_prev (np.array) -- the previous state estimate
    u_t (np.array)      -- the current control input

    Returns:
    G_x_t (np.array)    -- Jacobian of motion model wrt to x
    """
    xp = x_t_prev[0]
    yp = x_t_prev[1]
    thetap = x_t_prev[2]
    vp = x_t_prev[3]
    wp = x_t_prev[4]

    uLeftT = u_t[0]
    uRightT = u_t[1]

    G_x_t = np.array([ [1, 0, 0, -vp*DT*np.sin(thetap), DT*np.cos(thetap), 0, 0],
                        [0, 1, 0, vp*DT*np.cos(thetap), DT*np.sin(thetap), 0, 0],
                        [0, 0, 1, 0, 0, DT, 0],
                        [0, 0, 0, 1, 0, 0, DT],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]])

    return G_x_t

def calcPropJacU(x_t_prev, u_t):
    """Calculate the Jacobian of your motion model with respect to input

    Parameters:
    x_t_prev (np.array) -- the previous state estimate
    u_t (np.array)      -- the current control input

    Returns:
    G_x_t (np.array)    -- Jacobian of motion model wrt to x
    """
    xp = x_t_prev[0]
    yp = x_t_prev[1]
    thetap = x_t_prev[2]
    vp = x_t_prev[3]
    wp = x_t_prev[4]

    uLeftT = u_t[0]
    uRightT = u_t[1]

    G_u_t = np.array([  [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, ALPHA, ALPHA],
                        [GAMMA, 0, 0],
                        [0, -BETA, BETA]])

    return G_u_t

def predictionStep(x_t_prev, u_t, sigma_x_t_prev):
    """Compute the prediction of EKF

    Parameters:
    x_t_prev (np.array)         -- the previous state estimate
    u_t (np.array)              -- the control input
    sigma_x_t_prev (np.array)   -- the previous variance estimate

    Returns:
    x_bar_t (np.array)          -- the predicted state estimate of time t
    sigma_x_bar_t (np.array)    -- the predicted variance estimate of time t
    """

    
    sigma_u_t = np.identity(np.shape(u_t)[0])  # Covariance matrix of control input
    x_bar_t = propogateStateKF(x_t_prev, u_t)
    Gx = calcPropJacX(x_t_prev, u_t)
    Gu = calcPropJacU(x_t_prev, u_t)
    stateTerm = np.matmul(Gx, np.matmul(sigma_x_t_prev, np.transpose(Gx)))
    inputTerm = np.matmul(Gu, np.matmul(sigma_u_t, np.transpose(Gu)))
    sigma_x_bar_t = stateTerm + inputTerm

    return [x_bar_t, sigma_x_bar_t]

def calcMeasJac(x_bar_t):
    """Calculate the Jacobian of your measurment model with respect to state

    Parameters:
    x_bar_t (np.array)  -- the predicted state

    Returns:
    H_t (np.array)      -- Jacobian of measurment model
    """
    """STUDENT CODE START"""
    H_t = np.array([[1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0]])
    """STUDENT CODE END"""

    return H_t

def calcKalmanGain(sigma_x_bar_t, H_t, measCovar):
    """Calculate the Kalman Gain

    Parameters:
    sigma_x_bar_t (np.array)  -- the predicted state covariance matrix
    H_t (np.array)            -- the measurement Jacobian
    measCovar (np.array)      -- the measurment covariance matrix

    Returns:
    K_t (np.array)            -- Kalman Gain
    """
    """STUDENT CODE START"""

    sigma_z_t = measCovar
    H_t_T = np.transpose(H_t)
    invTerm = np.linalg.inv(np.matmul(H_t, np.matmul(sigma_x_bar_t, H_t_T)) + sigma_z_t)
    K_t = np.matmul(sigma_x_bar_t, np.matmul(H_t_T, invTerm))

    return K_t

def calcMeasPrediction(x_bar_t):
    """Calculate predicted measurement based on the predicted state

    Parameters:
    x_bar_t (np.array)  -- the predicted state

    Returns:
    z_bar_t (np.array)  -- the predicted measurement
    """

    """STUDENT CODE START"""
    z_bar_t = x_bar_t[0:4]
    """STUDENT CODE END"""

    return z_bar_t

def correctionStep(x_bar_t, z_t, sigma_x_bar_t, measCovar):
    """Compute the correction of EKF

    Parameters:
    x_bar_t       (np.array)    -- the predicted state estimate of time t
    z_t           (np.array)    -- the measured state of time t
    sigma_x_bar_t (np.array)    -- the predicted variance of time t
    measCovar (np.array)        -- the sensor variance

    Returns:
    x_est_t       (np.array)    -- the filtered state estimate of time t
    sigma_x_est_t (np.array)    -- the filtered variance estimate of time t
    """

    """STUDENT CODE START"""
    Ht = calcMeasJac(x_bar_t)
    Kt = calcKalmanGain(sigma_x_bar_t, Ht, measCovar)
    z_bar_t = calcMeasPrediction(x_bar_t)
    x_est_t = x_bar_t + np.matmul(Kt, (z_t - z_bar_t))
    sigma_x_est_t = np.matmul((np.identity(N) - np.matmul(Kt, Ht)), sigma_x_bar_t)
    
    """STUDENT CODE END"""

    return [x_est_t, sigma_x_est_t]

def kalmanFilter(numT, uUp, uLeft, uRight, xGPS, yGPS, zGPS, compass, measCovar):
    """Compute the EKF

    Parameters:
    numT          (int)         -- the number of timesteps
    uUp           (np.array)    -- the up inputs
    uLeft         (np.array)    -- the left inputs
    uRight        (np.array)    -- the right inputs
    xGPS          (np.array)    -- the x GPS measurements
    yGPS          (np.array)    -- the y GPS measurements
    zGPS          (np.array)    -- the z GPS measurements
    compass       (np.array)    -- the compass measurements
    measCovar     (np.array)    -- the sensor variance

    Returns:
    state_estimates     (np.array)    -- the filtered state estimates
    covariance_estimates(np.array)    -- the filtered variance estimates
    """

    #Create the variables to store the previous timestep
    state_est_t_prev = np.zeros(N)
    var_est_t_prev = np.identity(N)

    #Create the large array in which results will be stored
    state_estimates = np.empty((N, numT))
    covariance_estimates = np.zeros((N, N, numT))

    for t in range(numT):
        # Get control input and measurement
        u_t = np.array([uUp[t], uLeft[t] , uRight[t]])
        z_t = np.array([xGPS[t], yGPS[t], zGPS[t], compass[t]])

        #Run the prediction step
        state_pred_t, var_pred_t = predictionStep(state_est_t_prev, u_t, var_est_t_prev)

        #Run the correction step
        state_est_t, var_est_t = correctionStep(state_pred_t, z_t, var_pred_t, measCovar)
        
        #  For clarity sake/teaching purposes, we explicitly update t->(t-1)
        state_est_t_prev = state_est_t
        var_est_t_prev = var_est_t

        # Log Data
        state_estimates[:, t] = state_est_t
        covariance_estimates[:, :, t] = var_est_t

    return state_estimates, covariance_estimates

def measPlotter(tStamps, xGPS, yGPS, zGPS, compass):
    """
    Plots the measurements
    """
    plt.figure(1)
    plt.subplot(3,1,1)
    plt.plot(tStamps, xGPS)
    plt.ylabel("X Position")
    plt.subplot(3,1,2)
    plt.plot(tStamps, yGPS)
    plt.ylabel("Y Position")
    plt.subplot(3,1,3)
    plt.plot(tStamps, compass)
    plt.ylabel("Yaw")
    plt.show()

    plt.figure(2)
    plt.subplot(2,1,1)
    plt.plot(xGPS, yGPS)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.subplot(2,1,2)
    plt.plot(tStamps, compass)
    plt.ylabel("Yaw")
    plt.show()

    plt.figure(3)
    plt.plot(tStamps, zGPS)
    plt.ylabel("Altitude")
    plt.show()

    plt.figure(3)
    plt.plot(tStamps, compass)
    plt.ylabel("Yaw")
    plt.show()

def calc_covar_elipse(covar, pos):
    """
    Calculates the covariance elipse

    Parameters:
    covar      (np.array)    -- the covariance for t
    pos        (np.array)    -- the position for t

    Returns:
    xt     (np.array)    -- the x covarience element
    yt     (np.array)    -- the y covariance element
    
    """
    t = np.linspace(0,(2*math.pi), num=50) #The elipse plotter variable
    lambda1 = (covar[0,0] + covar[1,1]) / 2 + np.sqrt(np.power(((covar[0,0] - covar[1,1]) / 2), 2) + np.power(covar[0,1], 2))
    lambda2 = (covar[0,0] + covar[0,0]) / 2 - np.sqrt(np.power(((covar[0,0] - covar[1,1]) / 2), 2) + np.power(covar[0,1], 2))
    angle = np.arctan(lambda1 - covar[1,1])
    xt = np.sqrt(lambda1)*np.cos(angle)*np.cos(t) - np.sqrt(lambda2)*np.sin(angle)*np.sin(t) + pos[0]
    yt = np.sqrt(lambda1)*np.sin(angle)*np.cos(t) + np.sqrt(lambda2)*np.cos(angle)*np.sin(t) + pos[1]
    
    return xt, yt

def kalmanPlotter(tStamps, numT, stateEsts, varEsts, xGPS, yGPS, zGPS, compass, covar):
    xEst = stateEsts[0]
    yEst = stateEsts[1]
    zEst = stateEsts[2]
    cEst = stateEsts[3]
    vEst = stateEsts[4]
    sEst = stateEsts[5]
    wEst = stateEsts[6]

    xVar = varEsts[0,0, :]
    yVar = varEsts[1,1, :]
    zVar = varEsts[2,2, :]
    tVar = varEsts[3,3, :]

    plt.figure(1)
    plt.subplot(4,1,1)
    plt.plot(tStamps, xEst)
    plt.plot(tStamps, xGPS)
    plt.ylabel("X Position (m)")
    plt.subplot(4,1,2)
    plt.plot(tStamps, yEst)
    plt.plot(tStamps, yGPS)
    plt.ylabel("Y Position (m)")
    plt.subplot(4,1,3)
    plt.plot(tStamps, compass)
    plt.plot(tStamps, compass)
    plt.ylabel("Yaw")
    plt.subplot(4,1,4)
    plt.plot(tStamps, zEst, label="Estimated Altitude")
    plt.plot(tStamps, zGPS, "ro", markersize=1,  label="GPS Altitude")
    plt.show()

    plt.figure(2)
    plt.plot(tStamps, cEst, label="Estimated Yaw")
    plt.plot(tStamps, cEst + np.sqrt(tVar), label="Estimated Yaw + Standard Deviation")
    plt.plot(tStamps, cEst - np.sqrt(tVar), label="Estimated Yaw - Standard Deviation")
    plt.ylabel("Yaw (Radians)")
    plt.xlabel("Time (seconds)")
    plt.legend()
    plt.show()

    plt.figure(3)
    plt.plot(tStamps, zEst, label="Estimated Altitude")
    plt.plot(tStamps, zGPS, "ro", markersize=1,  label="GPS Altitude")
    plt.legend()
    plt.ylabel("Blimp Altitude (m)")
    plt.xlabel("Time (seconds)")

    plt.figure(4)
    plt.plot(xEst, yEst, label="Estimated Position")
    plt.plot(xGPS, yGPS, "bo", markersize=1, label="GPS Position")
    plt.xlim(-5, 5)
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    if covar:
        for t in range(numT):
            if t % 10 == 0:
                xCovarT, yCovarT = calc_covar_elipse(varEsts[:,:,t], stateEsts[:,t])
                plt.plot(xCovarT, yCovarT, 'r')
    
    plt.legend()
    plt.show()

    plt.figure(5)
    plt.subplot(2,2,1)
    plt.plot(tStamps, np.sqrt(xVar))
    plt.ylabel("X Variance")
    plt.subplot(2,2,2)
    plt.plot(tStamps, np.sqrt(yVar))
    plt.ylabel("Y Variance")
    plt.subplot(2,2,3)
    plt.plot(tStamps, np.sqrt(zVar))
    plt.ylabel("Z Variance")
    plt.subplot(2,2,4)
    plt.plot(tStamps, np.sqrt(tVar))
    plt.ylabel("Yaw Variance")
    plt.show()

    return


def main():
    filepath = "./logs/"
    filename = "Data"
    data = load_data(filename)

    #Time and indexing data
    tStamps = np.asarray(data["Time Stamp"])
    numT = np.shape(tStamps)[0]

    #Correction Step Data
    xGPS = np.asarray(data["X"])
    yGPS = np.asarray(data["Y"])
    zGPS = np.asarray(data["Z"])
    compass = np.asarray(data["Theta"])
    varX = np.var(xGPS[0:799])
    varY = np.var(yGPS[0:110])
    varZ = np.var(zGPS[400:799])
    varC = np.var(compass[400:799])
    measCovar = np.array([[varX, 0, 0, 0], [0, varY, 0, 0], [0, 0, varZ, 0], [0, 0, 0, varC]])

    #Input Data
    uUp = np.asarray(data["pUp"])
    uLeft = np.asarray(data["pLeft"])
    uRight = np.asarray(data["pRight"])

    #measPlotter(tStamps, xGPS, yGPS, zGPS, compass)

    kStateEsts, varEsts = kalmanFilter(numT, uUp, uLeft, uRight, xGPS, yGPS, zGPS, compass, measCovar)
    kalmanPlotter(tStamps, numT, kStateEsts, varEsts, xGPS, yGPS, zGPS, compass, covar=False)

    return


main()