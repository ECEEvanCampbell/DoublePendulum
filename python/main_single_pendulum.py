"""
DOUBLE PENDULUM SIMULATION:

1) Massless, rigid rods were used for the connections between masses.
2) Masses at the connection of the rods and end-point are considered as point-masses
3) Angles are measures CCW from downwards facing vector.

"""
from mpl_toolkits.mplot3d import Axes3D  

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
import sympy
from sympy.tensor.array.sparse_ndim_array import MutableSparseNDimArray

def main():

    # SIMULATION PARAMETERS
    simulation_time = 4 # seconds
    simulation_resolution = 0.001 # seconds
    simulation_steps = int(simulation_time / simulation_resolution) # How many times the simulation will run

    # VISUALIZATION PARAMETERS
    visualization_resolution = 0.1

    # PLANT PARAMETERS
    m1 = 1 # kg
    l1 = 1 # m
    g = 9.81 # m/s^2
    plant_parameters = (m1,l1,g)

    # States are:
    # Theta1 - angle of first pendulum link measured CCW from downward facing vector (radians)
    # Theta2 - angle of second pendulum link measured CCW from downward facing vector (radians)
    # Omega1 - angular velocity of first mass (radians/sec)
    # Omega2 - angular velocity of second mass (radians/sec)
    # Alpha1 - angular acceleration of first mass (radians/sec2)
    # Alpha2 - angular acceleration of second mass (radians/sec2)
    state_history = np.zeros((3,simulation_steps))

    # Initial_conditions
    state_history[:,0] = [0,0,0]

    # If controller is enabled, try to reach set point
    set_point = np.asarray([math.pi,0])

    # CONTROLLER SETUP
    controller_type = 'simultaneous_control'#dynamic_programming'#'none'#dynamic_programming# simultaneous_control
    
    if controller_type == 'simultaneous_control':
        controller_resolution = 0.1 # seconds
        control_end_time = 2 # seconds
        controller = simultaneous_control()
        control_sequence = controller.get_control_law(plant_parameters, state_history[:,0], set_point, control_end_time, controller_resolution,3,True)
    elif controller_type == "shooting_control":
        controller_resolution = 0.1 # seconds
        control_end_time = 2 # seconds
        controller = shooting_control()
        control_sequence = controller.get_control_law(plant_parameters, state_history[:,0], set_point, control_end_time, controller_resolution,3,True)
    elif controller_type == 'dynamic_programming':
        print('in progress!')
        resolution = 51
        controller_resolution = 0.1 # seconds
        controller = dynamic_programming()
        control_policy = controller.get_control_policy(plant_parameters, state_history[:,0], set_point, resolution, controller_resolution, control_options = 10, PLOTTING=True)
    
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
        elif controller_type == 'dynamic_programming':
            t1 = controller.state_value_to_id(state_history[0,i-1],'t1')
            o1 = controller.state_value_to_id(state_history[1,i-1],'o1')
            control_signal = controller.policy_map[t1,o1]
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
    ax = plt.axes(xlim=(-1*(l1), l1), ylim=(-1*(l1), l1))
    line, = ax.plot([], [], 'k-o')

    ani = animation.FuncAnimation(fig, animation_update, frames=range(0,simulation_steps,int(visualization_resolution/simulation_resolution)), fargs=(state_history,l1,line), blit=True,save_count=int(simulation_time/visualization_resolution),repeat=False)
    plt.show()
    s = ani.to_jshtml(fps=1/visualization_resolution)
    with open(f'SP_{controller_type}.html', "w") as f:
        f.write(s)

    
def plant_update(states, control_signal, plant_parameters):
    # All motion is defined by the second-order terms.
    # unpack parameters
    (m1,l1,g) = plant_parameters
    theta1 = states[0]
    
    omega1 = states[1]
    
    alpha1 = states[2]
    
    dd_theta1 = (- g/l1*math.sin(theta1) + control_signal)# TODO: investigate whether there should be a negative sign here
    return dd_theta1

def simulation_advance(states, state_change, simulation_resolution):
    # unpack parameters
    theta1 = states[0]
    omega1 = states[2]
    #alpha1 = states[4]
    alpha1 = state_change

    theta1 = theta1 + omega1 * simulation_resolution # + 0.5 * alpha1*simulation_resolution*simulation_resolution
    omega1 = omega1 + alpha1 * simulation_resolution

    next_states = np.array([theta1,omega1,alpha1])
    return next_states





class simultaneous_control:
    def __init__(self):
        self.control_type = 'simultaneous'

    def get_control_law(self, plant_parameters, initial_conditions, set_point, end_time, controller_resolution, num_iter=1, PLOTTING=False):
        self.time_points = int(end_time/controller_resolution)
        self.num_states = 3*(self.time_points-1)+2
        self.num_constraints = 2*(self.time_points-1)+4

        self.lambdas = np.ones((1, self.num_constraints))
        self.states = np.zeros((1, self.num_states))
        #self.states[0,:4] = initial_conditions
    
        self.set_point = set_point
        self.initial_conditions = initial_conditions
        self.controller_resolution = controller_resolution

        # store the plant parameters given to the controller
        self.plant_parameters = plant_parameters
        # prepare the variables for symbolic math
        self.get_state_sym()
        # define the costs of the problem
        self.get_costs()
        # define symbolic expression for delF
        self.get_delF()
        
        if PLOTTING:
            x0_history = np.zeros((num_iter, self.time_points))
            x1_history = np.zeros((num_iter, self.time_points))
            u_history  = np.zeros((num_iter, self.time_points-1))

        for i in range(num_iter):
            # define the constraints of the problem
            self.states[0,:2] = initial_conditions[:2]
            
            self.get_constraints()
            # define symbolic expression for delL
            self.get_delL()
            # define symbolic expression for G
            self.get_G()
            # define symbolic expression for Hl
            self.get_Hl()

            constraints = eval_expression(self.constraints, self.states, self.time_points)
            del_L       = eval_expression(self.del_L,       self.states, self.time_points)
            del_F       = eval_expression(self.del_F,       self.states, self.time_points)
            G           = eval_expression(self.G,           self.states, self.time_points)
            Hl          = eval_expression(self.Hl,          self.states, self.time_points)
            
            KKT_top = np.concatenate(  (Hl,np.transpose(G)), axis=1)
            KKT_bot = np.concatenate(  (G, np.zeros((G.shape[0], KKT_top.shape[1] - G.shape[1])))  , axis=1   )
            KKT = np.concatenate(  (KKT_top, KKT_bot),  axis=0  )

            perform_expression = np.concatenate(   ( del_F, constraints ), axis=0)

            delta = np.dot( np.linalg.pinv(KKT), perform_expression)
            dx = delta[:self.num_states , 0]
            #dlambdas = delta[self.num_states:,0]

            self.lambdas = -1 * delta[self.num_states:,0].reshape([1,-1])
            self.states = self.states - dx

            # get state vector and lambdas
            (x0_history[i,:], x1_history[i,:],u_history[i,:]) = unpack_states(self.states,self.time_points)

        if PLOTTING:
            _, axs = plt.subplots(3,1)
            for i in range(num_iter):
                axs[0].plot(np.linspace(0,end_time, int(end_time/controller_resolution)), x0_history[i,:])
                axs[0].set_title('Theta 1')

                axs[1].plot(np.linspace(0,end_time, int(end_time/controller_resolution)), x1_history[i,:])
                axs[1].set_title('DTheta 1')

                axs[2].plot(np.linspace(0,end_time-controller_resolution, int(end_time/controller_resolution)-1), u_history[i,:])
                axs[2].set_title('U')
            for ax in axs.flat:
                ax.set(xlabel='Time', ylabel='Value')
            plt.show()
        
        control_sequence = u_history[-1,:]
        return control_sequence

    def get_state_sym(self):
        state_sym = {}
        for n in range(self.time_points):
            state_sym[0,n] = sympy.symbols('x0'+str(n))
            state_sym[1,n] = sympy.symbols('x1'+str(n))
            state_sym[2,n] = sympy.symbols('u1'+str(n))
        self.state_sym = state_sym

    def get_constraints(self):
        (m1,l1,g) = self.plant_parameters
        constraints = []
        # Dynamics based constraints
        for n in range(1,self.time_points):
            # Each of these get a lambda in front of them.
            # x0n
            constraints.append(self.lambdas[0,2*(n-1)+0] * (self.state_sym[0,n]-self.state_sym[0,n-1]- self.state_sym[1,n-1]*self.controller_resolution))
            # x1n
            constraints.append(self.lambdas[0,2*(n-1)+1] * (self.state_sym[1,n]-self.state_sym[1,n-1]- (- g/l1*sympy.sin(self.state_sym[0,n-1]) + self.state_sym[2,n-1])*self.controller_resolution))
            
        # Initial conditions boundary
        constraints.append(self.lambdas[0,2*(n)+0]  *  (self.state_sym[0,0] - self.initial_conditions[0])**2 )
        constraints.append(self.lambdas[0,2*(n)+1]  *  (self.state_sym[1,0] - self.initial_conditions[1])**2 )
        # End condition boundary
        constraints.append(self.lambdas[0,2*(n)+2]  *   (self.state_sym[0,self.time_points-1] - self.set_point[0])**2 )
        constraints.append(self.lambdas[0,2*(n)+3]  *   (self.state_sym[1,self.time_points-1] - self.set_point[1])**2 )

        constraints = MutableSparseNDimArray(constraints, (len(constraints),1))
        self.constraints = constraints

    def get_costs(self):
        costs = []
        for n in range(1,self.time_points-1):
            # minimize error at each time step
            costs.append((self.state_sym[0,n]-self.set_point[0])**2)
            costs.append((self.state_sym[1,n]-self.set_point[1])**2)
            # minimize the control signal at each time set
            if n != self.time_points:
                costs.append(self.state_sym[2,n]**2)            
                 
        costs = MutableSparseNDimArray(costs, (len(costs),1) )
        self.costs = costs

    def get_delL(self):
        del_L = sympy.zeros(self.num_states,1)
        for n in range(self.time_points):
            # Gradient from constraints
            for c in range(self.constraints.shape[0]):
                # states PDE
                del_L[3*n+0,0] = del_L[3*n+0,0] + sympy.diff(self.constraints[c,0],self.state_sym[0,n])
                del_L[3*n+1,0] = del_L[3*n+1,0] + sympy.diff(self.constraints[c,0],self.state_sym[1,n])
                if n != self.time_points-1:
                    # control signal PDE
                    del_L[3*n+2,0] = del_L[3*n+2,0] + sympy.diff(self.constraints[c,0],self.state_sym[2,n])
            # Gradient from costs
            for c in range(self.costs.shape[0]):
                # states PDE
                del_L[3*n+0,0] = del_L[3*n+0,0] + sympy.diff(self.costs[c,0],self.state_sym[0,n])
                del_L[3*n+1,0] = del_L[3*n+1,0] + sympy.diff(self.costs[c,0],self.state_sym[1,n])
                if n != self.time_points-1:
                    # control signal PDE
                    del_L[3*n+2,0] = del_L[3*n+2,0] + sympy.diff(self.costs[c,0],self.state_sym[2,n])
        self.del_L = del_L

    def get_delF(self):
        del_F = sympy.zeros(self.num_states,1)
        for n in range(self.time_points):
            # Gradient from costs
            for c in range(self.costs.shape[0]):
                # states PDE
                del_F[3*n+0,0] = del_F[3*n+0,0] + sympy.diff(self.costs[c,0],self.state_sym[0,n])
                del_F[3*n+1,0] = del_F[3*n+1,0] + sympy.diff(self.costs[c,0],self.state_sym[1,n])
                if n != self.time_points-1:
                    # control signal PDE
                    del_F[3*n+2,0] = del_F[3*n+2,0] + sympy.diff(self.costs[c,0],self.state_sym[2,n])
        self.del_F = del_F

        
    def get_G(self):
        # sometimes called del g, this is the jacobian of the constraints.
        G = sympy.zeros(self.num_constraints, self.num_states)
        for n in range(self.time_points):
            for c in range(self.constraints.shape[0]):
                G[c,3*n]   = sympy.diff(self.constraints[c,0], self.state_sym[0,n])
                G[c,3*n+1] = sympy.diff(self.constraints[c,0], self.state_sym[1,n])
                if n != self.time_points-1:
                    G[c,3*+2] = sympy.diff(self.constraints[c,0], self.state_sym[2,n])
        self.G = G

    def get_Hl(self):
        Hl = sympy.zeros(self.num_states, self.num_states)
        for l in range(self.del_L.shape[0]):
            for n in range(self.time_points):
                Hl[l,3*n]   = sympy.diff(self.del_L[l,0], self.state_sym[0,n])
                Hl[l,3*n+1] = sympy.diff(self.del_L[l,0], self.state_sym[1,n])
                if n!= self.time_points-1:
                    Hl[l,3*n+2] = sympy.diff(self.del_L[l,0], self.state_sym[2,n])
        self.Hl = Hl


class shooting_control:
    def __init__(self):
        self.control_type = 'shooting'

    def get_control_law(self, plant_parameters, initial_conditions, set_point, end_time, controller_resolution, num_iter=1, PLOTTING=False):
        self.time_points = int(end_time/controller_resolution)
        self.num_states = 3*(self.time_points-1)+2
        self.num_constraints = 2*(self.time_points-1)+4

        self.lambdas = np.ones((1, self.num_constraints))
        self.states = np.zeros((1, self.num_states))
        #self.states[0,:4] = initial_conditions
    
        self.set_point = set_point
        self.initial_conditions = initial_conditions
        self.controller_resolution = controller_resolution

        # store the plant parameters given to the controller
        self.plant_parameters = plant_parameters
        # prepare the variables for symbolic math
        self.get_state_sym()
        # define the costs of the problem
        self.get_costs()
        # define symbolic expression for delF
        self.get_delF()
        
        if PLOTTING:
            x0_history = np.zeros((num_iter, self.time_points))
            x1_history = np.zeros((num_iter, self.time_points))
            u_history  = np.zeros((num_iter, self.time_points-1))

        for i in range(num_iter):
            # define the constraints of the problem
            self.states[0,:2] = initial_conditions[:2]
            
            self.get_constraints()
            # define symbolic expression for delL
            self.get_delL()
            # define symbolic expression for G
            self.get_G()
            # define symbolic expression for Hl
            self.get_Hl()

            constraints = eval_expression(self.constraints, self.states, self.time_points)
            del_L       = eval_expression(self.del_L,       self.states, self.time_points)
            del_F       = eval_expression(self.del_F,       self.states, self.time_points)
            G           = eval_expression(self.G,           self.states, self.time_points)
            Hl          = eval_expression(self.Hl,          self.states, self.time_points)
            
            KKT_top = np.concatenate(  (Hl,np.transpose(G)), axis=1)
            KKT_bot = np.concatenate(  (G, np.zeros((G.shape[0], KKT_top.shape[1] - G.shape[1])))  , axis=1   )
            KKT = np.concatenate(  (KKT_top, KKT_bot),  axis=0  )

            perform_expression = np.concatenate(   ( del_F, constraints ), axis=0)

            delta = np.dot( np.linalg.pinv(KKT), perform_expression)
            dx = delta[:self.num_states , 0]
            #dlambdas = delta[self.num_states:,0]

            self.lambdas = -1 * delta[self.num_states:,0].reshape([1,-1])
            self.states = self.states - dx

            # get state vector and lambdas
            (x0_history[i,:], x1_history[i,:],u_history[i,:]) = unpack_states(self.states,self.time_points)

        if PLOTTING:
            _, axs = plt.subplots(3,1)
            for i in range(num_iter):
                axs[0].plot(np.linspace(0,end_time, int(end_time/controller_resolution)), x0_history[i,:])
                axs[0].set_title('Theta 1')

                axs[1].plot(np.linspace(0,end_time, int(end_time/controller_resolution)), x1_history[i,:])
                axs[1].set_title('DTheta 1')

                axs[2].plot(np.linspace(0,end_time-controller_resolution, int(end_time/controller_resolution)-1), u_history[i,:])
                axs[2].set_title('U')
            for ax in axs.flat:
                ax.set(xlabel='Time', ylabel='Value')
            plt.show()
        
        control_sequence = u_history[-1,:]
        return control_sequence

    def get_state_sym(self):
        state_sym = {}
        for n in range(self.time_points):
            state_sym[0,n] = sympy.symbols('x0'+str(n))
            state_sym[1,n] = sympy.symbols('x1'+str(n))
            state_sym[2,n] = sympy.symbols('u1'+str(n))
        self.state_sym = state_sym

    def get_constraints(self):
        (m1,l1,g) = self.plant_parameters
        constraints = []
        # Dynamics based constraints
        for n in range(1,self.time_points):
            # Each of these get a lambda in front of them.
            # x0n
            constraints.append(self.lambdas[0,2*(n-1)+0] * (self.state_sym[0,n]-self.state_sym[0,n-1]- self.state_sym[1,n-1]*self.controller_resolution))
            # x1n
            constraints.append(self.lambdas[0,2*(n-1)+1] * (self.state_sym[1,n]-self.state_sym[1,n-1]- (g/l1*sympy.sin(self.state_sym[0,n-1]) + self.state_sym[2,n-1])*self.controller_resolution))
            
        # Initial conditions boundary
        constraints.append(self.lambdas[0,2*(n)+0]  *  (self.state_sym[0,0] - self.initial_conditions[0])**2 )
        constraints.append(self.lambdas[0,2*(n)+1]  *  (self.state_sym[1,0] - self.initial_conditions[1])**2 )
        # End condition boundary
        constraints.append(self.lambdas[0,2*(n)+2]  *   (self.state_sym[0,self.time_points-1] - self.set_point[0])**2 )
        constraints.append(self.lambdas[0,2*(n)+3]  *   (self.state_sym[1,self.time_points-1] - self.set_point[1])**2 )

        constraints = MutableSparseNDimArray(constraints, (len(constraints),1))
        self.constraints = constraints

    def get_costs(self):
        costs = []
        for n in range(1,self.time_points-1):
            # minimize error at each time step
            costs.append((self.state_sym[0,n]-self.set_point[0])**2)
            costs.append((self.state_sym[1,n]-self.set_point[1])**2)
            # minimize the control signal at each time set
            if n != self.time_points:
                costs.append(self.state_sym[2,n]**2)            
                 
        costs = MutableSparseNDimArray(costs, (len(costs),1) )
        self.costs = costs

    def get_delL(self):
        del_L = sympy.zeros(self.num_states,1)
        for n in range(self.time_points):
            # Gradient from constraints
            for c in range(self.constraints.shape[0]):
                # states PDE
                del_L[3*n+0,0] = del_L[3*n+0,0] + sympy.diff(self.constraints[c,0],self.state_sym[0,n])
                del_L[3*n+1,0] = del_L[3*n+1,0] + sympy.diff(self.constraints[c,0],self.state_sym[1,n])
                if n != self.time_points-1:
                    # control signal PDE
                    del_L[3*n+2,0] = del_L[3*n+2,0] + sympy.diff(self.constraints[c,0],self.state_sym[2,n])
            # Gradient from costs
            for c in range(self.costs.shape[0]):
                # states PDE
                del_L[3*n+0,0] = del_L[3*n+0,0] + sympy.diff(self.costs[c,0],self.state_sym[0,n])
                del_L[3*n+1,0] = del_L[3*n+1,0] + sympy.diff(self.costs[c,0],self.state_sym[1,n])
                if n != self.time_points-1:
                    # control signal PDE
                    del_L[3*n+2,0] = del_L[3*n+2,0] + sympy.diff(self.costs[c,0],self.state_sym[2,n])
        self.del_L = del_L

    def get_delF(self):
        del_F = sympy.zeros(self.num_states,1)
        for n in range(self.time_points):
            # Gradient from costs
            for c in range(self.costs.shape[0]):
                # states PDE
                del_F[3*n+0,0] = del_F[3*n+0,0] + sympy.diff(self.costs[c,0],self.state_sym[0,n])
                del_F[3*n+1,0] = del_F[3*n+1,0] + sympy.diff(self.costs[c,0],self.state_sym[1,n])
                if n != self.time_points-1:
                    # control signal PDE
                    del_F[3*n+2,0] = del_F[3*n+2,0] + sympy.diff(self.costs[c,0],self.state_sym[2,n])
        self.del_F = del_F

        
    def get_G(self):
        # sometimes called del g, this is the jacobian of the constraints.
        G = sympy.zeros(self.num_constraints, self.num_states)
        for n in range(self.time_points):
            for c in range(self.constraints.shape[0]):
                G[c,3*n]   = sympy.diff(self.constraints[c,0], self.state_sym[0,n])
                G[c,3*n+1] = sympy.diff(self.constraints[c,0], self.state_sym[1,n])
                if n != self.time_points-1:
                    G[c,3*n+2] = sympy.diff(self.constraints[c,0], self.state_sym[2,n])
        self.G = G

    def get_Hl(self):
        Hl = sympy.zeros(self.num_states, self.num_states)
        for l in range(self.del_L.shape[0]):
            for n in range(self.time_points):
                Hl[l,3*n]   = sympy.diff(self.del_F[l,0], self.state_sym[0,n])
                Hl[l,3*n+1] = sympy.diff(self.del_F[l,0], self.state_sym[1,n])
                if n!= self.time_points-1:
                    Hl[l,3*n+2] = sympy.diff(self.del_F[l,0], self.state_sym[2,n])
        self.Hl = Hl


class dynamic_programming:
    def __init__(self):
        self.control_type = 'dynamic'
        # policy map is (theta1, omega1)
        # theta1 ranges between  -2pi, 2pi
        self.theta1_lim = (-2*math.pi, 2*math.pi)
        # omega1 ranges between -5, 5
        self.omega1_lim = (-5, 5)

    def get_control_policy(self, plant_parameters, initial_conditions, set_point, resolution, control_resolution, control_options = 5, PLOTTING=False):
        self.plant_parameters = plant_parameters
        self.initial_condition = initial_conditions
        self.set_point = set_point
        self.resolution = resolution
        self.control_resolution = control_resolution
        self.control_options = np.linspace(-50.0, 50.0, num=control_options)
        
        # initialize with a blank policy
        self.policy_map = np.zeros((self.resolution, self.resolution), dtype=int)
        # Define the state cost at the final step to be (t1-setpoint)^2 + (t2-setpoint)^2 + (o1-setpoint)^2 + (o2-setpoint)^2
        self.state_cost_per_step = self.get_state_cost_per_step()
        # otherwise, define state cost as shown in class (uniform except the setpoint)
        #self.state_cost_per_step = self.get_uniform_cost_per_step()
        self.zero_set_point()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        Theta = np.arange(-2*math.pi, 2*math.pi, 4*math.pi/self.resolution)
        DTheta = np.arange(-5,5, 10/self.resolution)
        Theta, DTheta = np.meshgrid(Theta, DTheta)
        
        surf = ax.plot_surface(Theta, DTheta, self.state_cost_per_step, cmap = cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlabel('DTheta')
        ax.set_ylabel('Theta')
        ax.set_zlabel('U')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

        converged = False
        last_state_cost = self.state_cost_per_step
        last_policy_map = np.array(self.policy_map)
        while not converged:
            cost_candidate = np.zeros((self.control_options.shape[0]))
            for t1 in range (self.resolution):
                for o1 in range ( self.resolution):
                    for u in range(self.control_options.shape[0]):
                        # for each discrete state, check where the control action puts the pendulum
                        (t1i,o1i) = self.advance_dynamics(t1,o1,self.control_options[u])
                        # what is the cost for this transition (cost = cost of where you were + cost of state you land in + abs(control signal applied))
                        cost_candidate[u] = last_state_cost[t1,o1] + last_state_cost[t1i,o1i]# + 0.05 * np.abs(self.control_options[u])
                    # find the control option with the smallest 
                    best_control = np.argmin(cost_candidate)
                    # update the policy map w/ the best control option
                    self.policy_map[t1,o1] = self.control_options[best_control]
                    # update the 
                    self.state_cost_per_step[t1,o1] = cost_candidate[best_control]
            policy_update = np.array(self.policy_map - last_policy_map != 0,dtype=int)
            if np.sum(policy_update) > 0.01 * policy_update.size:
                print('not converged')
            else:
                converged = True
            last_policy_map = np.array(self.policy_map)
            last_state_cost = np.array(self.state_cost_per_step)
            # Check if converged
            # converged is defined as less than 1% of the policy changing
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        Theta = np.arange(-2*math.pi, 2*math.pi, 4*math.pi/self.resolution)
        DTheta = np.arange(-5,5, 10/self.resolution)
        Theta, DTheta = np.meshgrid(Theta, DTheta)
        
        surf = ax.plot_surface(Theta, DTheta, self.policy_map, cmap = cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlabel('DTheta')
        ax.set_ylabel('Theta')
        ax.set_zlabel('U')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        return self.policy_map

    def state_id_to_value(self, id, identifier):
        if identifier == 't1':
            return id * (self.theta1_lim[1]- self.theta1_lim[0]) / self.resolution + self.theta1_lim[0]
        elif identifier == 'o1':
            return id * (self.omega1_lim[1]- self.omega1_lim[0]) / self.resolution + self.omega1_lim[0]
        else:
            exit()

    def state_value_to_id(self, val, identifier):
        if identifier == 't1':
            lims = self.theta1_lim
        elif identifier == 'o1':
            lims = self.omega1_lim
        else:
            exit()
        
        if val < lims[0]:
            id = 0
        elif val > lims[1]:
            id = self.resolution - 1
        else:
            id = int(  (val - lims[0]) * self.resolution / (lims[1] - lims[0])  )
        
        return id

    def advance_dynamics(self, t1,o1,control_signal):

        t1_val = self.state_id_to_value(t1,'t1')
        o1_val = self.state_id_to_value(o1,'o1')

        t1_next = t1_val + o1_val * self.control_resolution

        (m1,l1,g) = self.plant_parameters

        alpha1 = (- g/l1*math.sin(t1_val) + control_signal)
        o1_next = o1_val + alpha1 * self.control_resolution

        t1_next = self.state_value_to_id(t1_next,'t1')
        o1_next = self.state_value_to_id(o1_next,'o1')

        return t1_next, o1_next


    def get_state_cost_per_step(self):
        state_cost_per_step = np.zeros((self.resolution, self.resolution), dtype=float)
        for t1 in range(self.resolution):
            t1_val = self.state_id_to_value(t1,'t1')
            cost_t1 = (t1_val - self.set_point[0])**2
            
            for o1 in range(self.resolution):
                o1_val = self.state_id_to_value(o1,'o1')
                cost_o1 = (o1_val - self.set_point[1])**2
                
                state_cost_per_step[t1,o1] = cost_t1 +  cost_o1
        return state_cost_per_step
    
    def get_uniform_cost_per_step(self):
        # As per example in class, assign 5 to all states on the map
        state_cost_per_step = 5*np.ones((self.resolution, self.resolution), dtype=float)
        return state_cost_per_step
    
    def zero_set_point(self):
        theta1_id = int((self.set_point[0] - self.theta1_lim[0]) / (self.theta1_lim[1]-self.theta1_lim[0]) * self.resolution)
        omega1_id = int((self.set_point[1] - self.omega1_lim[0])/ (self.omega1_lim[1]-self.omega1_lim[0]) * self.resolution)
        self.state_cost_per_step[theta1_id, omega1_id] = 0



def sequential_shooting_control(plant_parameters, initial_conditions, set_point, end_time, controller_resolution, num_iter=1):
    # Only optimize our u terms.
    print('not ready')
    # Go through an iteration


def unpack_states(states,num_iter):
    x0 = np.zeros((1,num_iter))
    x1 = np.zeros((1,num_iter))
    u  = np.zeros((1,num_iter-1))
    for n in range(num_iter):
        x0[0,n] = states[0,3*(n)]
        x1[0,n] = states[0,3*(n)+1]
        if n != num_iter-1:
            u[0,n]  = states[0,3*(n)+2]
    return x0,x1,u

def eval_expression(expression, states, num_step):
    (x0,x1,u) = unpack_states(states,num_step)
    evaluated_expression = sympy.MutableSparseNDimArray(expression)
    # Now eval the symbolic gradient
    for n1 in range(evaluated_expression.shape[0]):
        for n2 in range(evaluated_expression.shape[1]):
            req_symbols = evaluated_expression[n1,n2].free_symbols
            if bool(req_symbols):
                for s in range(len(req_symbols)):
                    set_element = req_symbols.pop()
                    symbol = str(set_element)
                    state_id = symbol[0]
                    dof_id = int(symbol[1])
                    time_id = int(symbol[2:])
                    if state_id == 'x':
                        if dof_id == 0:
                            value_for_symbol = x0[0,time_id]
                        elif dof_id == 1:
                            value_for_symbol = x1[0,time_id]
                    elif state_id == 'u':
                        value_for_symbol = u[0,time_id]
                    evaluated_expression[n1,n2] = evaluated_expression[n1,n2].subs(set_element, value_for_symbol)
    return np.array(evaluated_expression.tolist(), dtype=float)


def animation_update(i,state_history,l1,line):
    x= np.zeros(2)
    x[0] = 0
    x[1] = l1*math.sin(state_history[0,i])
    y = np.zeros(2)
    y[0] = 0
    y[1] = -1*l1*math.cos(state_history[0,i])
    line.set_data(x,y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim((-1*(l1), l1))
    plt.ylim((-1*(l1), l1))
    return line,

if __name__ == "__main__":
    main()