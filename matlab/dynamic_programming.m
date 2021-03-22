function [control_policy, theta_bins, omega_bins, theta_lims, omega_lims, TIMINGS] = dynamic_programming(plant_parameters, set_point, state_resolution, control_resolution, time_resolution, PLOTTING)
%DYNAMIC_PROGRAMMING Summary of this function goes here
%   Detailed explanation goes here

% unpack parameters
g = plant_parameters(1);
m1 = plant_parameters(2);
l1 = plant_parameters(3);


theta_lims = [0, 2*pi];
omega_lims = [-5, 5];
control_lims = [-7, 7];

% discretize the control inputs
control_bins = linspace(control_lims(1),control_lims(2),control_resolution);
% discretize the states -- use center of state as test point.
theta_bins   = linspace(theta_lims(1), theta_lims(2), state_resolution+1);
theta_bins_width = diff(theta_bins);
theta_bins   = theta_bins(1:end-1) + theta_bins_width(1)/2; 
omega_bins   = linspace(omega_lims(1), omega_lims(2), state_resolution+1);
omega_bins_width = diff(omega_bins);
omega_bins   = omega_bins(1:end-1) + omega_bins_width(1)/2; 

clear theta_bins_width omega_bins_width 

control_policy = zeros(state_resolution);

policy_value = zeros(state_resolution);
% get initial policy_value -- according to cost function
for n1 = 1:state_resolution
    n1_val = theta_bins(n1);
    n1_cost = 5*(n1_val - set_point(1))^2;
    for n2 = 1:state_resolution
        n2_val = omega_bins(n2);
        n2_cost = (n2_val - set_point(2))^2;
        policy_value(n1,n2) = n1_cost + n2_cost;
    end
end

converged = 0;
last_policy_value = policy_value;
last_control_policy = control_policy;
iterations = 1;
figure()

subplot(1,2,1)
colormap('hot');
map_1 = imagesc(policy_value);
axMap_1 = map_1.Parent;
axMap_1.XTickLabels = compose('%g',theta_bins);
axMap_1.XTick = linspace(1, state_resolution, state_resolution);
axMap_1.XTickLabelRotation = 90;
axMap_1.YTickLabels = compose('%g',omega_bins);
axMap_1.YTick = linspace(1, state_resolution, state_resolution);
axMap_1.XLabel.String = 'Angle (rad)';
axMap_1.YLabel.String = 'Angular rate (rad/s)';
axMap_1.Title.String = 'Policy Value';
map_1.CData = policy_value;

subplot(1,2,2)
colormap('hot');
map_2 = imagesc(control_policy);
axMap_2 = map_2.Parent;
axMap_2.XTickLabels = compose('%g',theta_bins);
axMap_2.XTick = linspace(1, state_resolution, state_resolution);
axMap_2.YTickLabels = compose('%g',omega_bins);
axMap_2.XTickLabelRotation = 90;
axMap_2.YTick = linspace(1, state_resolution, state_resolution);
axMap_2.XLabel.String = 'Angle (rad)';
axMap_2.YLabel.String = 'Angular rate (rad/s)';
axMap_2.Title.String = 'Control Policy';
map_1.CData = control_policy;


t1 = tic;
while ~ converged
    
    for t = 1:state_resolution
        t_val = theta_bins(t);
        for o = 1:state_resolution
            cost_candidate = zeros(1, control_resolution);
            o_val = omega_bins(o);
            for u = 1:control_resolution
                control = control_bins(u);
                updated_states = shoot(plant_parameters, [t_val, o_val], control, time_resolution);
                % wrap the theta space
                updated_states(1) = mod(updated_states(1), theta_lims(2) - theta_lims(1)) + theta_lims(1);
                [~,t_update] = min( abs(updated_states(1)-theta_bins));
                [~,o_update] = min( abs(updated_states(2)-omega_bins));
                cost_candidate(u) = last_policy_value(t_update,o_update);
            end
            [min_cost_added, best_control] = min(cost_candidate);
            policy_value(t,o)   = policy_value(t,o) + min_cost_added;
            control_policy(t,o) = control_bins(best_control);
        end
        
    end
    
    TIMINGS(iterations) = toc(t1);
    if sum(sum(~(last_control_policy - control_policy == 0))) < 0.01 * state_resolution^2
        converged = 1;
        disp(['Converged on the ' num2str(iterations) 'th iteration']);
    else
        iterations = iterations + 1;
        last_control_policy = control_policy;
        last_policy_value = policy_value;
    end
    
    set(map_1,'CData',policy_value);
    set(map_2,'CData',control_policy);
    
    pause(0.1)
end
% for plotting -- meshgrid



end

function [updated_states] = shoot(plant_parameters, states, control, time_resolution)

    theta = states(1);
    omega = states(2);

    g  = plant_parameters(1);
    m1 = plant_parameters(2);
    l1 = plant_parameters(3);
    
    updated_states(1) = theta + omega * time_resolution;
    updated_states(2) = omega + (-g/l1*sin(theta) -0.7*omega + control) * time_resolution;
	
end