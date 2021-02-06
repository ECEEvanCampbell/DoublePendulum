"""
DOUBLE PENDULUM SIMULATION:

1) Massless, rigid rods were used for the connections between masses.
2) Masses at the connection of the rods and end-point are considered as point-masses
3) Angles are measures CCW from downwards facing vector.

"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import animation
import sympy

def main():

    # SIMULATION PARAMETERS
    simulation_time = 15 # seconds
    simulation_resolution = 0.001 # seconds
    simulation_steps = int(simulation_time / simulation_resolution) # How many times the simulation will run

    # VISUALIZATION PARAMETERS
    visualization_resolution = 0.1

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
    state_history[:,0] = [math.pi*0.75,0,0,0,0,0]

    # If controller is enabled, try to reach set point
    set_point = np.asarray([math.pi, math.pi, 0,0,0,0])

    # CONTROLLER SETUP
    controller_type = 'simultaneous_control'#'none'#
    
    if controller_type == 'simultaneous_control':
        print('in progress!')
        controller_resolution = 0.1 # seconds
        control_end_time = 5 # seconds
        control_sequence = simultaneous_control(plant_parameters, state_history[:,0], set_point, control_end_time, controller_resolution,2)
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
            if i * simulation_resolution < control_end_time - controller_resolution:
                control_signal = control_sequence[int(i*simulation_resolution/controller_resolution)]
            else:
                control_signal = 0
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

    fig = plt.figure()
    ax = plt.axes(xlim=(-1*(l1+l2), l1+l2), ylim=(-1*(l1+l2), l1+l2))
    line, = ax.plot([], [], 'k-o')

    ani = animation.FuncAnimation(fig, animation_update, frames=range(0,simulation_steps,int(visualization_resolution/simulation_resolution)), fargs=(state_history,l1,l2,line), blit=True,save_count=int(simulation_time/visualization_resolution),repeat=False)
    plt.show()
    s = ani.to_jshtml(fps=1/visualization_resolution)
    with open(f'DP_{controller_type}.html', "w") as f:
        f.write(s)

    
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




def simultaneous_control(plant_parameters, initial_conditions, set_point, end_time, controller_resolution, num_iter=3):
    
    
    
    # HOW MANY CONTROL TIME POINTS
    N = int(end_time/controller_resolution)
    # HOW MANY STATES 7(N-1)+4
    # x0,x1,x2,x3,x4,x5,u : 1->(N-1), x0,x1,x2,x3,x4,x5 : N
    num_states = 7*(N-1) + 6
    # HOW MANY CONSTRAINTS?
    # <(N-1) * 6  Dynamics> + <12 BC>
    num_constraints = 6*(N-1) + 12
    # initial size of matrices
    HL = np.zeros((num_states,num_states))
    lambdas = np.ones((1,num_constraints))
    states = np.ones((1,num_states))
    states[0,:6] = initial_conditions

    # for plotting
    x0_history = np.zeros((num_iter, N))
    x1_history = np.zeros((num_iter, N))
    x2_history = np.zeros((num_iter, N))
    x3_history = np.zeros((num_iter, N))
    u_history = np.zeros((num_iter, N-1))

    for i in range(num_iter):
        state_sym, constraints_symbolic, _, symbolic_g, g_eval = get_symbolic_g(plant_parameters, states, lambdas, initial_conditions, set_point, controller_resolution)
        constraints_eval = get_eval_constraints(state_sym, constraints_symbolic, states)
        _, G_eval = get_symbolic_G(state_sym, constraints_symbolic, states)
        _, Hl_eval = get_symbolic_Hl(state_sym, symbolic_g, states)

        g_eval = np.reshape(np.array(g_eval,dtype = 'float'),(-1, 1))
        constraints_eval = np.reshape(np.array(constraints_eval,dtype = 'float'),(-1, 1))
        G_eval = np.array(G_eval,dtype = 'float')
        Hl_eval = np.array(Hl_eval,dtype = 'float')

        KKT1 = np.concatenate((Hl_eval, np.transpose(G_eval)), axis=1)
        KKT2 = np.concatenate((G_eval, np.zeros((G_eval.shape[0], G_eval.shape[0]))), axis=1)
        KKT  = np.concatenate((KKT1,KKT2),axis=0)

        g_c = np.concatenate((g_eval,constraints_eval),axis=0)

        a = -1 * np.linalg.pinv(KKT) * g_c

        dx = a[:g_eval.shape[0],0]
        dlambdas = a[g_eval.shape[0]:,0]
        states = states + dx
        lambdas = lambdas + dlambdas

        # get state vector and lambdas
        (x0_history[i,:], x1_history[i,:],x2_history[i,:],x3_history[i,:],_,_,u_history[i,:]) = unpack_states(states,N)
        
    fig, axs = plt.subplots(5,1)
    for i in range(num_iter):
        axs[0].plot(np.linspace(0,end_time, int(end_time/controller_resolution)), x0_history[i,:])
        axs[0].set_title('Theta 1')

        axs[1].plot(np.linspace(0,end_time, int(end_time/controller_resolution)), x1_history[i,:])
        axs[1].set_title('Theta 2')

        axs[2].plot(np.linspace(0,end_time, int(end_time/controller_resolution)), x2_history[i,:])
        axs[2].set_title('DTheta 1')

        axs[3].plot(np.linspace(0,end_time, int(end_time/controller_resolution)), x3_history[i,:])
        axs[3].set_title('DTheta 2')

        axs[4].plot(np.linspace(0,end_time-controller_resolution, int(end_time/controller_resolution)-1), u_history[i,:])
        axs[4].set_title('U')
    for ax in axs.flat:
        ax.set(xlabel='Time', ylabel='Value')
    plt.show()
    
    control_sequence = u_history[-1,:]
    return control_sequence


def get_symbolic_g(plant_parameters, states, lambdas, initial_conditions, set_point, controller_resolution):
    num_step = int((states.shape[1]-6)/7 + 1)
    (m1,m2,l1,l2,g) = plant_parameters
    (x0,x1,x2,x3,x4,x5,u) = unpack_states(states,num_step)
    # write down all constraints in symbolic form
    state_sym = {}
    for n in range(num_step):
        state_sym[0,n] = sympy.symbols('x0'+str(n))
        state_sym[1,n] = sympy.symbols('x1'+str(n))
        state_sym[2,n] = sympy.symbols('x2'+str(n))
        state_sym[3,n] = sympy.symbols('x3'+str(n))
        state_sym[4,n] = sympy.symbols('x4'+str(n))
        state_sym[5,n] = sympy.symbols('x5'+str(n))
        state_sym[6,n] = sympy.symbols('u1'+str(n))
    
    constraints = []
    # Dynamics based constraints
    for n in range(1,num_step):
        # Each of these get a lambda in front of them.
        # x0n
        constraints.append(lambdas[0,6*(n-1)+0] * (state_sym[0,n]-state_sym[0,n-1]- state_sym[2,n-1]*controller_resolution))
        # x1n
        constraints.append(lambdas[0,6*(n-1)+1] * (state_sym[1,n]-state_sym[1,n-1]- state_sym[3,n-1]*controller_resolution))
        # x2n
        #constraints.append(lambdas[0,6*(n-1)+2] * (state_sym[2,n]-state_sym[2,n-1]  + m2*l2*state_sym[5,n-1]*sympy.cos(state_sym[0,n-1]-state_sym[1,n-1])*controller_resolution/((m1+m2)*l1)  - m2*l2*state_sym[3,n-1]**2*sympy.sin(state_sym[0,n-1]-state_sym[1,n-1]) *controller_resolution/((m1+m2)*l1)    - g/l1*sympy.sin(state_sym[0,n-1])*controller_resolution +state_sym[6,n-1]*controller_resolution  )   )
        constraints.append(lambdas[0,6*(n-1)+2] * (state_sym[2,n]-state_sym[2,n-1]- state_sym[4,n-1]*controller_resolution))
        # x3n
        #constraints.append(lambdas[0,6*(n-1)+3] * (state_sym[3,n]-state_sym[3,n-1]  - l1/l2*state_sym[2,n-1]**2*sympy.sin(state_sym[0,n-1]-state_sym[1,n-1])*controller_resolution             + g/l2*sympy.sin(state_sym[1,n-1])*controller_resolution + l1/l2*state_sym[4,n-1]*sympy.cos(state_sym[0,n-1]-state_sym[1,n-1])*controller_resolution              )      )
        constraints.append(lambdas[0,6*(n-1)+3] * (state_sym[3,n]-state_sym[3,n-1]- state_sym[5,n-1]*controller_resolution))
        # x4n
        constraints.append(lambdas[0,6*(n-1)+4] * (state_sym[4,n] + m2*l2/((m1+m2)*l1)*state_sym[5,n-1] * sympy.cos(state_sym[0,n-1]-state_sym[1,n-1])    + m2*l2/((m1+m2)*l1)*state_sym[3,n-1]**2 *sympy.sin(state_sym[0,n-1]-state_sym[1,n-1])  +g/l1*sympy.sin(state_sym[0,n-1]) -state_sym[6,n-1]  )   )
        # -> Can we enforce a constant angular jerk term? I can't write an expression for a first order term without a second order term.
        # x5n
        constraints.append(lambdas[0,6*(n-1)+5] * (state_sym[5,n] -l1/l2*state_sym[2,n-1]**2 *sympy.sin(state_sym[0,n-1]-state_sym[1,n-1])    +g/l2*sympy.sin(state_sym[1,n-1])    +l1/l2*state_sym[4,n-1]*sympy.cos(state_sym[0,n-1]-state_sym[1,n-1])    )  )

    # Initial conditions boundary
    constraints.append(lambdas[0,6*(n)+0]  *  (state_sym[0,0] - initial_conditions[0])**2 )
    constraints.append(lambdas[0,6*(n)+1]  *  (state_sym[1,0] - initial_conditions[1])**2 )
    constraints.append(lambdas[0,6*(n)+2]  *  (state_sym[2,0] - initial_conditions[2])**2 )
    constraints.append(lambdas[0,6*(n)+3]  *  (state_sym[3,0] - initial_conditions[3])**2 )
    constraints.append(lambdas[0,6*(n)+4]  *  (state_sym[4,0] - initial_conditions[4])**2 )
    constraints.append(lambdas[0,6*(n)+5]  *  (state_sym[5,0] - initial_conditions[5])**2 )

    # End condition boundary
    constraints.append(lambdas[0,6*(n)+6]  *   (state_sym[0,num_step-1] - set_point[0])**2 )
    constraints.append(lambdas[0,6*(n)+7]  *   (state_sym[1,num_step-1] - set_point[1])**2 )
    constraints.append(lambdas[0,6*(n)+8]  *   (state_sym[2,num_step-1] - set_point[2])**2 )
    constraints.append(lambdas[0,6*(n)+9]  *   (state_sym[3,num_step-1] - set_point[3])**2 )
    constraints.append(lambdas[0,6*(n)+10] *   (state_sym[4,num_step-1] - set_point[4])**2 )
    constraints.append(lambdas[0,6*(n)+11] *   (state_sym[5,num_step-1] - set_point[5])**2 )

    # Costs
    costs = []
    for n in range(num_step-1):
        # minimize error at each time step
        costs.append((state_sym[0,n]-set_point[0])**2)
        costs.append((state_sym[1,n]-set_point[1])**2)
        costs.append((state_sym[2,n]-set_point[2])**2)
        costs.append((state_sym[3,n]-set_point[3])**2)
        costs.append((state_sym[4,n]-set_point[4])**2)
        costs.append((state_sym[5,n]-set_point[5])**2)
        # minimize the control signal at each time set
        if n != num_step:
            costs.append(state_sym[6,n]**2)

    # Now differentiate all the constraints to get the gradient
    symbolic_g = [0]*(7*num_step -1)
    for n in range(num_step):
        # Gradient from constraints
        for c in range(len(constraints)):
            # states PDE
            symbolic_g[7*n+0] = symbolic_g[7*n+0] + sympy.diff(constraints[c],state_sym[0,n])
            symbolic_g[7*n+1] = symbolic_g[7*n+1] + sympy.diff(constraints[c],state_sym[1,n])
            symbolic_g[7*n+2] = symbolic_g[7*n+2] + sympy.diff(constraints[c],state_sym[2,n])
            symbolic_g[7*n+3] = symbolic_g[7*n+3] + sympy.diff(constraints[c],state_sym[3,n])
            symbolic_g[7*n+4] = symbolic_g[7*n+4] + sympy.diff(constraints[c],state_sym[4,n])
            symbolic_g[7*n+5] = symbolic_g[7*n+5] + sympy.diff(constraints[c],state_sym[5,n])
            if n != num_step-1:
                # control signal PDE
                symbolic_g[7*n+6] = symbolic_g[7*n+6] + sympy.diff(constraints[c],state_sym[6,n])
        # Gradient from costs
        for c in range(len(costs)):
            # states PDE
            symbolic_g[7*n+0] = symbolic_g[7*n+0] + sympy.diff(costs[c],state_sym[0,n])
            symbolic_g[7*n+1] = symbolic_g[7*n+1] + sympy.diff(costs[c],state_sym[1,n])
            symbolic_g[7*n+2] = symbolic_g[7*n+2] + sympy.diff(costs[c],state_sym[2,n])
            symbolic_g[7*n+3] = symbolic_g[7*n+3] + sympy.diff(costs[c],state_sym[3,n])
            symbolic_g[7*n+4] = symbolic_g[7*n+4] + sympy.diff(costs[c],state_sym[4,n])
            symbolic_g[7*n+5] = symbolic_g[7*n+5] + sympy.diff(costs[c],state_sym[5,n])
            if n != num_step-1:
                # control signal PDE
                symbolic_g[7*n+6] = symbolic_g[7*n+6] + sympy.diff(costs[c],state_sym[6,n])

    # Now eval the symbolic gradient
    g_eval = []
    for n in range(len(symbolic_g)):

        req_symbols = symbolic_g[n].free_symbols
        for s in range(len(req_symbols)):
            # a single required symbol for expression #n
            set_element = req_symbols.pop()
            symbol = str(set_element)
            # find what symbol that actually is
            state_id = symbol[0]
            dof_id = int(symbol[1])
            time_id = int(symbol[2:])
            if state_id == 'x':
                if dof_id == 0:
                    value_for_symbol = x0[0,time_id]
                elif dof_id == 1:
                    value_for_symbol = x1[0,time_id]
                elif dof_id == 2:
                    value_for_symbol = x2[0,time_id]
                elif dof_id == 3:
                    value_for_symbol = x3[0,time_id]
                elif dof_id == 4:
                    value_for_symbol = x4[0,time_id]
                elif dof_id == 5:
                    value_for_symbol = x5[0,time_id]
                else:
                    print('ERROR!! < Unknown DOF ID > ')
                    break
            elif state_id == 'u':
                value_for_symbol = u[0,time_id]
            else:
                print('ERROR!! < Unknown STATE ID > ')
                break

            if s == 0:
                g_eval.append(symbolic_g[n].subs(set_element, value_for_symbol))
            else:
                g_eval[n] = g_eval[n].subs(set_element, value_for_symbol)

    return state_sym, constraints, costs, symbolic_g, g_eval


def get_symbolic_G(state_sym, constraints, states):
    num_step = int((states.shape[1]-6)/7 + 1)
    (x0,x1,x2,x3,x4,x5,u) = unpack_states(states,num_step)
    # Differentiate each constraint by each state
    symbolic_G = [ [0]*states.shape[1] for _ in range(len(constraints)) ]
    for c in range(len(constraints)):
        for n in range(num_step):
            symbolic_G[c][7*(n)+0] = sympy.diff(constraints[c], state_sym[0,n])
            symbolic_G[c][7*(n)+1] = sympy.diff(constraints[c], state_sym[1,n])
            symbolic_G[c][7*(n)+2] = sympy.diff(constraints[c], state_sym[2,n])
            symbolic_G[c][7*(n)+3] = sympy.diff(constraints[c], state_sym[3,n])
            symbolic_G[c][7*(n)+4] = sympy.diff(constraints[c], state_sym[4,n])
            symbolic_G[c][7*(n)+5] = sympy.diff(constraints[c], state_sym[5,n])
            if n != num_step-1:
                symbolic_G[c][7*(n)+6] = sympy.diff(constraints[c], state_sym[0,n])
    # Now eval the symbolic jacobian
    G_eval = [ [0]*states.shape[1] for _ in range(len(constraints)) ]
    for c in range(len(G_eval)):
        for n in range(len(G_eval[0])):
            G_eval[c][n] = symbolic_G[c][n]
            req_symbols = symbolic_G[c][n].free_symbols
            if bool(req_symbols):
                for s in range(len(req_symbols)):
                    # a single required symbol for expression #n
                    set_element = req_symbols.pop()
                    symbol = str(set_element)
                    # find what symbol that actually is
                    state_id = symbol[0]
                    dof_id = int(symbol[1])
                    time_id = int(symbol[2:])
                    if state_id == 'x':
                        if dof_id == 0:
                            value_for_symbol = x0[0,time_id]
                        elif dof_id == 1:
                            value_for_symbol = x1[0,time_id]
                        elif dof_id == 2:
                            value_for_symbol = x2[0,time_id]
                        elif dof_id == 3:
                            value_for_symbol = x3[0,time_id]
                        elif dof_id == 4:
                            value_for_symbol = x4[0,time_id]
                        elif dof_id == 5:
                            value_for_symbol = x5[0,time_id]
                        else:
                            print('ERROR!! < Unknown DOF ID > ')
                            break
                    elif state_id == 'u':
                        value_for_symbol = u[0,time_id]
                    else:
                        print('ERROR!! < Unknown STATE ID > ')
                        break
                    G_eval[c][n] = G_eval[c][n].subs(set_element, value_for_symbol)               
    return symbolic_G, G_eval


def get_symbolic_Hl(state_sym, symbolic_g, states):
    # Get the Hessian of the Legrangian.
    # Previously, we had found the gradient of the Legrangian (symbolic_g <- get_symbolic_g)
    # We only need to differentiate one more time to get the Hessian.
    num_step = int((states.shape[1]-6)/7 + 1)
    (x0,x1,x2,x3,x4,x5,u) = unpack_states(states,num_step)
    # Differentiate each constraint by each state
    symbolic_Hl = [ [0]*states.shape[1] for _ in range(states.shape[1]) ]

    for s1 in range(num_step-1):
        for s2 in range(num_step-1):
            symbolic_Hl[7*s1+0][7*s2+0] = sympy.diff(symbolic_g[s1],state_sym[0,s2])
            symbolic_Hl[7*s1+0][7*s2+1] = sympy.diff(symbolic_g[s1],state_sym[1,s2])
            symbolic_Hl[7*s1+0][7*s2+2] = sympy.diff(symbolic_g[s1],state_sym[2,s2])
            symbolic_Hl[7*s1+0][7*s2+3] = sympy.diff(symbolic_g[s1],state_sym[3,s2])
            symbolic_Hl[7*s1+0][7*s2+4] = sympy.diff(symbolic_g[s1],state_sym[4,s2])
            symbolic_Hl[7*s1+0][7*s2+5] = sympy.diff(symbolic_g[s1],state_sym[5,s2])
            if s2 != num_step-1:
                # control signal PDE
                symbolic_Hl[7*s1+0][7*s2+7] = sympy.diff(symbolic_g[s1],state_sym[6,s2])
    # Now eval the symbolic Hessian
    Hl_eval = [ [0]*states.shape[1] for _ in range(states.shape[1]) ]
    for s1 in range(len(Hl_eval)):
        for s2 in range(len(Hl_eval[0])):
            Hl_eval[s1][s2] = symbolic_Hl[s1][s2]
            if hasattr(symbolic_Hl[s1][s2],'free_symbols'):
                req_symbols = symbolic_Hl[s1][s2].free_symbols
                if bool(req_symbols):
                    for s in range(len(req_symbols)):
                        # a single required symbol for expression #n
                        set_element = req_symbols.pop()
                        symbol = str(set_element)
                        # find what symbol that actually is
                        state_id = symbol[0]
                        dof_id = int(symbol[1])
                        time_id = int(symbol[2:])
                        if state_id == 'x':
                            if dof_id == 0:
                                value_for_symbol = x0[0,time_id]
                            elif dof_id == 1:
                                value_for_symbol = x1[0,time_id]
                            elif dof_id == 2:
                                value_for_symbol = x2[0,time_id]
                            elif dof_id == 3:
                                value_for_symbol = x3[0,time_id]
                            elif dof_id == 4:
                                value_for_symbol = x4[0,time_id]
                            elif dof_id == 5:
                                value_for_symbol = x5[0,time_id]
                            else:
                                print('ERROR!! < Unknown DOF ID > ')
                                break
                        elif state_id == 'u':
                            value_for_symbol = u[0,time_id]
                        else:
                            print('ERROR!! < Unknown STATE ID > ')
                            break
                        Hl_eval[s1][s2] = Hl_eval[s1][s2].subs(set_element, value_for_symbol)               
    return symbolic_Hl, Hl_eval



def get_eval_constraints(state_sym, constraints_symbolic, states):
    num_step = int((states.shape[1]-4)/7 + 1)
    (x0,x1,x2,x3,x4,x5,u) = unpack_states(states,num_step)
    constraints_eval = []
    for n in range(len(constraints_symbolic)):
        req_symbols = constraints_symbolic[n].free_symbols
        for s in range(len(req_symbols)):
            # a single required symbol for expression #n
            set_element = req_symbols.pop()
            symbol = str(set_element)
            # find what symbol that actually is
            state_id = symbol[0]
            dof_id = int(symbol[1])
            time_id = int(symbol[2:])
            if state_id == 'x':
                if dof_id == 0:
                    value_for_symbol = x0[0,time_id]
                elif dof_id == 1:
                    value_for_symbol = x1[0,time_id]
                elif dof_id == 2:
                    value_for_symbol = x2[0,time_id]
                elif dof_id == 3:
                    value_for_symbol = x3[0,time_id]
                elif dof_id == 4:
                    value_for_symbol = x4[0,time_id]
                elif dof_id == 5:
                    value_for_symbol = x5[0,time_id]
                else:
                    print('ERROR!! < Unknown DOF ID > ')
                    break
            elif state_id == 'u':
                value_for_symbol = u[0,time_id]
            else:
                print('ERROR!! < Unknown STATE ID > ')
                break

            if s == 0:
                constraints_eval.append(constraints_symbolic[n].subs(set_element, value_for_symbol))
            else:
                constraints_eval[n] = constraints_eval[n].subs(set_element, value_for_symbol)
    return constraints_eval


def unpack_states(states,num_iter):
    x0 = np.zeros((1,num_iter))
    x1 = np.zeros((1,num_iter))
    x2 = np.zeros((1,num_iter))
    x3 = np.zeros((1,num_iter))
    x4 = np.zeros((1,num_iter))
    x5 = np.zeros((1,num_iter))
    u  = np.zeros((1,num_iter-1))
    for n in range(num_iter):
        x0[0,n] = states[0,7*(n)]
        x1[0,n] = states[0,7*(n)+1]
        x2[0,n] = states[0,7*(n)+2]
        x3[0,n] = states[0,7*(n)+3]
        x4[0,n] = states[0,7*(n)+4]
        x5[0,n] = states[0,7*(n)+5]
        if n != num_iter-1:
            u[0,n]  = states[0,7*(n)+6]
    return x0,x1,x2,x3,x4,x5,u



def animation_update(i,state_history,l1,l2,line):
    x= np.zeros(3)
    x[0] = 0
    x[1] = l1*math.sin(state_history[0,i])
    x[2] = l1*math.sin(state_history[0,i]) + l2*math.sin(state_history[1,i])
    y = np.zeros(3)
    y[0] = 0
    y[1] = -1*l1*math.cos(state_history[0,i])
    y[2] =  -1*l1*math.cos(state_history[0,i]) - l2 *math.cos(state_history[1,i])
    line.set_data(x,y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim((-1*(l1+l2), l1+l2))
    plt.ylim((-1*(l1+l2), l1+l2))
    return line,

if __name__ == "__main__":
    main()