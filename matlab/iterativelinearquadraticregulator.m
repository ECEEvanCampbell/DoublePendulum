function [k,K,X,control_signal, TIMINGS] = iterativelinearquadraticregulator(plant_parameters, start_position, set_point, control_duration, control_interval, PLOTTING)
%ITERATIVELINEARQUADRATICREGULATOR Summary of this function goes here
%   Detailed explanation goes here
% diehl's lecture ----- https://www.syscop.de/files/2017ss/NOC/recordings/260617.mp4

% unpack parameters
g = plant_parameters(1);
m1 = plant_parameters(2);
l1 = plant_parameters(3);

% levinberg-marquardt regularization
lambda = 1;
lambda_factor = 2;
lambda_max = lambda * 10 ^ 4;

% convergence condition
converge_eps = 0.001; 


% initialize with initial state and initial control sequence
time_points = control_duration/control_interval;
control_signal = ones(time_points-1, 1);
states = ones(time_points, 2);
states(1,:) = start_position;
num_states = 2; %theta, omega

%cost_function = sum((states - set_point).^2,'all')  + sum(control_signal.^2,'all');

% set up iteratation
converged = 0;
sim_new_trajectory = 1;
iterations = 1;
t1 = tic;
while ~ converged
   
   % forward pass
   if sim_new_trajectory
       
       [X, cost] = shoot(states(1,:), control_signal, control_interval, plant_parameters,set_point);
       old_cost = cost;
       
       % use linear dynamics and quadratic cost function
       % x(t+1) = f( x(t),u(t) )
       f_x  = zeros(time_points, num_states, num_states);
       f_u  = zeros(time_points, num_states, 1); % 1 = number of controls
       % cost function
       l    = zeros(time_points,1);
       l_x  = zeros(time_points,num_states);
       l_xx = zeros(time_points, num_states, num_states);
       l_u  = zeros(time_points, 1);
       l_uu = zeros(time_points, 1, 1);
       l_ux = zeros(time_points, 1, num_states);
       
       for t = 1:time_points-1
          % x(t+1) = f(x(t), u(t)) = x(t) + dx(t) * dt
          % linearized dx(t) = A(t) * x(t) + B(t) * u(t)
          % f_x = U /- A(t)
          % f_u = B(t)
          [A, B] = finite_differences(states(t,:), control_signal(t), control_interval, plant_parameters );
          f_x(t,:,:) =  eye(num_states) + A * control_interval;
          f_u(t,:,:) = B * control_interval;
          
          [l(t,1), l_x(t,:), l_xx(t,:,:),...
              l_u(t,:), l_uu(t,:,:), l_ux(t,:,:)] = compute_cost(states(t,:), control_signal(t), set_point );
          
          l(t,1)      = l(t,1) * control_interval;
          l_x(t,:)    = l_x(t,:) * control_interval;
          l_xx(t,:,:) = l_xx(t,:,:) * control_interval;
          l_u(t,:)    = l_u(t,:) * control_interval;
          l_uu(t,:,:) = l_uu(t,:,:) * control_interval;
          l_ux(t,:,:) = l_ux(t,:,:) * control_interval;
           
       end
       
       [l(t+1,:), l_x(t+1,:), l_xx(t+1,:,:)] = compute_final_cost(states(t+1,:), set_point);
       
       sim_new_trajectory = 0;
   end
   
   % initialize V with final state cost and set up k K
   V = l(end,1);
   V_x = l_x(end,:);
   V_xx = squeeze(l_xx(end,:,:));
   
   k = zeros(time_points,1); % feedforward modification % 1 - number of controllable dof
   K = zeros(time_points, 1, num_states); % feedback gain
   
   % backwards iteration to get V, Q, k and K
   for t = time_points-1:-1:1
       
      %Q_x = l_x(t,:) + (squeeze(f_x(t,:,:)) * V_x')';
      Q_x = l_x(t,:) + V_x*squeeze(f_x(t,:,:));
      Q_u = l_u(t,:) + (f_u(t,:) * V_x')';
      
      Q_xx = squeeze(l_xx(t,:,:)) + squeeze(f_x(t,:,:))' * V_xx * squeeze(f_x(t,:,:)); %+ V_x * squeeze(f_xx(t,:,:));
      Q_ux = reshape(l_ux(t,:,:), 1, num_states) + f_u(t,:) * (V_xx * squeeze(f_x(t,:,:)));
      Q_uu = squeeze(l_uu(t,:)) + f_u(t,:) * V_xx * f_u(t,:)';
      
      % calculate Q_uu inv with regularization term set by
      % Levenberg-Marquardt
      
      [Q_uu_evecs, Q_uu_evals] = eig(Q_uu,'vector'); % check if returned order is vals, vecs
      Q_uu_evals(Q_uu_evals < 0) = 0;
      Q_uu_evals = Q_uu_evals + lambda;
      Q_uu_inv = Q_uu_evecs * (diag(1/Q_uu_evals) * Q_uu_evecs');
      %Q_uu_inv = pinv(Q_uu);

      
      k(t,1)   = -1 * Q_uu_inv * Q_u;
      K(t,1,:) = -1 * Q_uu_inv * Q_ux; 
      
      
      V_x  = Q_x - reshape(K(t,:,:),1, num_states) * (Q_uu * k(t));
      V_xx = Q_xx - reshape(K(t,:,:),1,num_states)' * (Q_uu * reshape(K(t,:,:),1,num_states));
       
   end
   
   new_control_signal = zeros(time_points-1,1); % 1 - controllable DOF
   new_states = states(1,:);
   for t = 1:time_points-1
      new_control_signal(t,1) = control_signal(t) + k(t) + reshape(K(t,1,:),1,num_states) * (new_states-X(t,:))';
      [~, new_states] = plant_dynamics(new_states, new_control_signal(t,1), control_interval, plant_parameters);
   end
   % evaluate new trajectory
   [Xnew, costnew] = shoot(states(1,:), new_control_signal, control_interval, plant_parameters, set_point);
   
   
   TIMINGS(iterations) = toc(t1);
   disp(num2str(costnew))
   if costnew < cost
       % decrease lambda
       lambda = lambda / lambda_factor;
       X = Xnew;
       control_signal = new_control_signal;
       oldcost = cost;
       cost = costnew;
   
        sim_new_trajectory = 1; % we did better, so keep going
        
        if and(iterations ~= 0, abs(old_cost - cost)/cost < converge_eps)
            converged = 1;
            disp(['Converged on iteration ' num2str(iterations) '.']);
        end
        
   else
        lambda = lambda * lambda_factor;
        
        if lambda > lambda_max
            disp(['Lambda exceeded maximum value']);
            break
        end
            
            
   end
   iterations = iterations + 1; 
end



% do a forward pass
% simulate the system using x0,U to get the trajectory through state space
% X (full shoot)

% do a backward pass, estimate the value function and dynamics for each
% (x,u) in the state-space and control signal trajectories.

% calculate an updated control signal u and evaluate cost of trajectory
% resulting from (x0,u)
%   if |cost(x0, uhat)  - cost(x0,u)| < threshold -- converged
%   if cost (x0,uhat) < cost(x0,u), then set u = uhat and change the update
%   size to be more aggressive, go back to step 2
%   if cost(x0, uhat) > cost(x0, u) change update size to be more modest,
%   go back to step 3

end




function [X, cost] = shoot(x0, U, dt, plant_parameters, set_point)
    tN = size(U,1);
    num_states = length(x0);
    
    X = zeros(tN, num_states);
    X(1,:) = x0;
    cost = 0;
    for t = 1:tN-1
        
       [~,X(t+1,:)] = plant_dynamics(X(t,:), U(t,:), dt,plant_parameters);
       [l,~,~,~,~,~] = compute_cost(X(t,:), U(t), set_point);
       cost = cost + dt*l;% dt*l
    end
   
    [l_f,~,~] = compute_final_cost(X(end,:), set_point);
    cost = cost + l_f;
end


function [l, l_x, l_xx, l_u, l_uu, l_ux] = compute_cost(x,u,set_point)

    num_states = length(x);
    
    l = sum(u^2);% + sum((x(:,1)-set_point(1)).^2) + sum((x(:,2)-set_point(2)).^2);
    l_x = zeros(1,num_states);%2*sum(x(:,1)- set_point(1)) + 2*sum(x(:,2) - set_point(1));
    l_xx = zeros(num_states,num_states);
    
    l_u = 2*u;
    l_uu = 2 * eye(1); % only 1 control signal
    l_ux = zeros(1, num_states); %only 1 control signal

end

function [l, l_x, l_xx] = compute_final_cost(x, set_point)
    
    num_states = length(x);
    l_x = zeros(num_states,1);
    l_xx = zeros(num_states, num_states);
    
    wp = 1e4;
    wv = 1;
    
    l = wp*(x(1,1)-set_point(1))^2 + wv * (x(1,2)-set_point(2))^2; 
    l_x = [wp*2*(x(1,1) - set_point(1)) ; wv * 2 * (x(1,2)-set_point(2))];
    
    
    l_xx = [ 2*wp,0;0, 2*wv];

end



function [xdot, xnext] = plant_dynamics(x,u,dt, plant_parameters)
    theta = x(1);
    omega = x(2);

    g  = plant_parameters(1);
    m1 = plant_parameters(2);
    l1 = plant_parameters(3);
    
    xnext(1) = theta + omega * dt;
    xnext(2) = omega + (-g/l1*sin(theta) -0.7*omega + u) * dt;

    xdot = ((xnext - x) / dt);
end




function [A, B] = finite_differences(x, u, dt, plant_parameters)

    num_states = length(x);
    dof = length(u);

    A = zeros(num_states, num_states);
    B = zeros(num_states, dof);

    eps = 1e-4;
    for ii = 1:num_states
        % partial diff wrt x
        inc_x = x;
        inc_x(ii) = inc_x(ii) +eps;
        [state_inc, ~] = plant_dynamics(inc_x, u, dt, plant_parameters);
        dec_x = x;
        dec_x(ii) = dec_x(ii) - eps;
        [state_dec, ~] = plant_dynamics(dec_x, u, dt, plant_parameters);
        A(:,ii) = (state_inc - state_dec) / (2*eps);
    end
    
    for ii = 1:dof
        % partial diff wrt u
        inc_u = u;
        inc_u(ii) = inc_u(ii) + eps;
        [state_inc,~] = plant_dynamics(x, inc_u, dt, plant_parameters);
        dec_u = u;
        dec_u(ii) = dec_u(ii) - eps;
        [state_dec,~] = plant_dynamics(x, dec_u, dt, plant_parameters);
        B(:,ii) = (state_inc - state_dec) / (2 * eps);
    end
end