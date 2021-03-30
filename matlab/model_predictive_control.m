function [control_sequence] = model_predictive_control(lookahead, plant_parameters, set_point, initial_condition, control_resolution, time_resolution)
%MODEL_PREDICTIVE_CONTROL Summary of this function goes here
%   Detailed explanation goes here

control_lims = [-10, 10];

% discretize the control inputs
control_bins = linspace(control_lims(1),control_lims(2),control_resolution);

% get all combinations of possible control signals
k = lookahead;
n = numel(control_bins);
combs = control_bins(dec2base(0:n^k-1,n)-'0'+1);  
combs = combs(all(diff(combs.')>=0, 1),:);  

% evaluate cost of each option using this cost function
compute_cost = @(states) 1e3 * (states(1) - set_point(1))^2 + 1e-3*(states(2) - set_point(2))^2;
cost = zeros(1, size(combs,1));

discount_factor = 0.5;
% check all combinations
for k = 1:size(combs,1)
    states = initial_condition;
    
    for t = 1:lookahead
        % shoot ahead  lookahead times for each brute force control
        % sequence
        
        k1 = rk_dynamics(states, combs(k,t), plant_parameters)';
        k2 = rk_dynamics(states + time_resolution/2 * k1, combs(k,t), plant_parameters)';
        k3 = rk_dynamics(states + time_resolution/2 * k2, combs(k,t), plant_parameters)';
        k4 = rk_dynamics(states + time_resolution * k3, combs(k,t), plant_parameters)';
    
        states = states + time_resolution/6*(k1+2*k2+2*k3+k4);
        
        cost(k) = cost(k) + discount_factor^t * compute_cost(states);
        
    end
    
end
% for plotting -- meshgrid
[~, best_control] = min(cost);
control_sequence = combs(best_control,:);


end

function states_dot = rk_dynamics(states,control, plant_parameters)

g = plant_parameters(1);
m1 = plant_parameters(2);
l1 = plant_parameters(3);

states_dot(1) = states(2);
states_dot(2) = (-g/l1 * sin(states(1)) -0.7*states(2) + control);

end