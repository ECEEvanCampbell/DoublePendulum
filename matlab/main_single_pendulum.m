clear variables
close all
clc

% Dynamic parameters 

g = 9.81;
m1 = 1;
l1 = 1;
plant_parameters = [g, m1, l1];

% Boundary conditions
start_position = [pi/4; 0];
set_point      = [pi; 0];

% simulation -- wrap theta correctly
theta_lims = [0, 2*pi];

%% controller setup
controller = 3;
% 0 - no control
% 1 - direct simultaneous control
% 2 - direct shooting control
% 3 - dynamic programming
% 4 - iterative linear quadratic gaussian
% 5 - actor critic network
% Control system parameters 


% Equations of motion
syms theta theta_dot control
c_theta_update =  @(theta, theta_dot) theta + theta_dot * control_interval;
c_thetadot_update = @(theta, theta_dot, control) theta_dot + (-g/l1 * sin(theta) -0.7*theta_dot + control) * control_interval;

if controller == 1
    % simultaneous control
    control_duration = 2; % seconds
    control_interval = 0.1; % seconds
    iterations = 1;
    [control_law, TIMINGS] = simultaneous_control(plant_parameters, start_position, set_point, control_duration, control_interval, iterations, 1);
    
elseif controller == 2
    % shooting control
    control_duration = 2; % seconds
    control_interval = 0.1; % seconds
    % TODO
elseif controller == 3
    % dynamic programmming
    state_resolution = 101; % how quantized is the plane
    control_resolution = 10; % how quantized are the control options
    time_resolution = 0.1; % how long are control signals applied (required for shooting)
    [control_policy, theta_bins, omega_bins, theta_lims, omega_lims, TIMINGS] = dynamic_programming(plant_parameters, set_point, state_resolution, control_resolution, time_resolution, 1);

elseif controller == 4
    % iterative linear quadratic gaussian
    
elseif controller == 5
    % actor critic network
    
end
    
    
    
%% simulation of control law / policy

% Simulation setup
simulation_duration = 10; % seconds
simulation_interval = 0.1; % seconds
state_history = zeros(2,simulation_duration / simulation_interval + 1);
state_history(:,1) = start_position;
cost_values = zeros(1, simulation_duration/simulation_interval);
s_theta_update =  @(theta, theta_dot) theta + theta_dot * simulation_interval;
s_thetadot_update = @(theta, theta_dot, control) theta_dot + (-g/l1 * sin(theta) -0.7*theta_dot + control) * simulation_interval;



h = figure();
filename = ['single_pendulum_' num2str(controller) '.gif'];
for i = 1: simulation_duration / simulation_interval
    state_history(1,i) = mod(state_history(1,i), theta_lims(2) - theta_lims(1)) + theta_lims(1);
    if controller == 0
        control_signal = 0;
    elseif controller == 1
        if i*simulation_interval > control_duration - control_interval
            control_signal = 0;
        else
            control_signal = control_law(ceil(i*simulation_interval / control_interval));
        end
    elseif controller == 2
        break
    elseif controller == 3
        [~,discrete_t] = min( abs(state_history(1,i)-theta_bins));
        [~,discrete_o] = min( abs(state_history(2,i)-omega_bins));
        control_signal = control_policy(discrete_t,discrete_o);
    end
    
    % compute cost
    cost_values(i) = (state_history(1,i)-set_point(1))^2 + (state_history(2,i)-set_point(2))^2 + control_signal^2;
    
    % plot pendulum
    subplot(1,3,1)
    pendulum_xlocs = [0 l1*sin(state_history(1,i))];
    pendulum_ylocs = [0 -1*l1*cos(state_history(1,i))];
    plot(pendulum_xlocs, pendulum_ylocs);
    xlim([-l1, l1])
    ylim([-l1, l1])
    xlabel('x')
    ylabel('y')
    title('Pendulum Simulation')
    
    % plot phase portrait
    subplot(1,3,2)
    plot(state_history(1,1:i), state_history(2,1:i))
    xlabel('theta')
    ylabel('theta_dot')
    title('Phase Portrait')
    xlim([-2*pi 2*pi])
    ylim([-10 10])

    
    % plot cost
    subplot(1,3,3)
    plot(cost_values(1:i));
    xlabel('iteration')
    ylabel('quadratic cost')
    title('Cost')
    xlim([0 simulation_duration/simulation_interval])
    ylim([0 50])
    


    frame = getframe(h);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    if i == 1
        imwrite(imind,cm,filename, 'gif', 'Loopcount',inf, 'DelayTime', 0);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append');
    end
    
    % compute update for next iteration
    state_history(1,i+1) = s_theta_update(state_history(1,i), state_history(2,i));
    state_history(2,i+1) = s_thetadot_update(state_history(1,i), state_history(2,i), control_signal);
    
    
end



% 
% 
% df_theta_theta       = diff(f_theta_update, theta);
% df_theta_thetadot    = diff(f_theta_update, theta_dot);
% df_theta_control     = diff(f_theta_update, control);
% 
% df_thetadot_theta    = diff(f_thetadot_update, theta);
% df_thetadot_thetadot = diff(f_thetadot_update, theta_dot);
% df_thetadot_control  = diff(f_thetadot_update, control);
% 
% 
% % subs -- substitute
% % symvar -- gets parameters of symbolic function
% 
% symvar(df_theta_thetadot)
