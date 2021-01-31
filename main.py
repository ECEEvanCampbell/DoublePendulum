"""
DOUBLE PENDULUM SIMULATION:

1) Massless, rigid rods were used for the connections between masses.
2) Masses at the connection of the rods and end-point are considered as point-masses
3) Angles are measures CCW from downwards facing vector.

"""

import numpy as np
import math
import matplotlib.pyplot as plt

def main():

    # SIMULATION PARAMETERS
    simulation_time = 3 # seconds
    simulation_resolution = 0.1 # seconds
    simulation_steps = int(simulation_time / simulation_resolution) # How many times the simulation will run

    # PLANT PARAMETERS
    m1 = 1 # kg
    m2 = 1 # kg
    l1 = 1 # m
    l2 = 1 # m
    g = 9.81 # m/s^2
    plant_parameters = (m1,m2,l1,l2,g)

    # States are:
    # Theta1 - angle of first pendulum link measured CCW from downward facing vector (radians)
    # Theta2 - angle of second pendulum link measured CCW from downward facing vector (radians)
    # Omega1 - angular velocity of first mass (radians/sec)
    # Omega2 - angular velocity of second mass (radians/sec)
    # Alpha1 - angular acceleration of first mass (radians/sec2)
    # Alpha2 - angular acceleration of second mass (radians/sec2)
    state_history = np.zeros((6,simulation_steps))

    # Initial_conditions
    state_history[:,0] = [math.pi/2,0,0,0,0,0]

    # If controller is enabled, try to reach set point
    set_point = [math.pi, math.pi, 0,0,0,0]

    # CONTROLLER SETUP
    controller_type = 'none'
    if controller_type == 'simultaneous_control':
        print('not coded yet!')
        # Run simultaneous control to get control signal array
    elif controller_type == 'PID':
        print('not coded yet!')
        # Set Point
        # Establish kp,ki,kd
    elif controller_type == 'LQR':
        print('not coded yet!')
        # Set point
        # Get gains
    else: # if none
        control_signal = 0

    for i in range(1,simulation_steps):


        if controller_type == 'simultaneous_control':
            print('not coded yet!')
            # based on controller_resolution, get the input for this simulation step
        elif controller_type == 'PID':
            print('not coded yet!')
            # get control signal from error signal and pre-determined kp,kd,ki values
        elif controller_type == 'LQR':
            print('not coded yet!')
            # get control signal from error signal and pre-determined K value
        else: # if none
            control_signal = 0

        states_dot = plant_update(state_history[:,i-1],control_signal,plant_parameters)
        state_history[:,i] = simulation_advance(state_history[:,i-1], states_dot, simulation_resolution)

    print('Simulation done!')

    
def plant_update(states, control_signal, plant_parameters):
    # All motion is defined by the second-order terms.
    # unpack parameters
    (m1,m2,l1,l2,g) = plant_parameters
    theta1 = states[0]
    theta2 = states[1]
    omega1 = states[2]
    omega2 = states[3]
    alpha1 = states[4]
    alpha2 = states[5]

    dd_theta1 = -1*(m2*l2*alpha2*math.cos(theta1-theta2) + m2*l2*omega2*omega2*math.sin(theta1-theta2) + (m1+m2)*g*math.sin(theta1))/((m1+m2)*l1) + control_signal
    dd_theta2 = (m2*l1*omega1*omega1*math.sin(theta1-theta2)-m2*g*math.sin(theta2)-m2*l1*alpha1*math.cos(theta1-theta2))/(m2*l2)

    return dd_theta1, dd_theta2

def simulation_advance(states, state_change, simulation_resolution):
    # unpack parameters
    theta1 = states[0]
    theta2 = states[1]
    omega1 = states[2]
    omega2 = states[3]
    #alpha1 = states[4]
    #alpha2 = states[5]
    alpha1 = state_change[0]
    alpha2 = state_change[1]

    theta1 = theta1 + omega1 * simulation_resolution + 0.5 * alpha1*simulation_resolution*simulation_resolution
    theta2 = theta2 + omega2 * simulation_resolution + 0.5 * alpha2*simulation_resolution*simulation_resolution
    omega1 = omega1 + alpha1 * simulation_resolution
    omega2 = omega2 + alpha2 * simulation_resolution

    next_states = np.array([theta1,theta2,omega1,omega2,alpha1,alpha2])
    return next_states




if __name__ == "__main__":
    main()