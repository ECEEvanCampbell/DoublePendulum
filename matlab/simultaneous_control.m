function [control_sequence, TIMINGS] = simultaneous_control(plant_parameters, start_position, set_point, control_duration, control_interval, iterations, PLOTTING)
%SIMULTENEOUS_CONTROL Summary of this function goes here
%   Detailed explanation goes here

t1 = tic;
% setup the systems of equations
time_points = control_duration / control_interval;
num_states = 3 * time_points  -1;
num_constraints = 2 * (time_points-1) +4;

lambdas = ones(1, num_constraints);
states  = zeros(1, num_states);

% unpack parameters
g = plant_parameters(1);
m1 = plant_parameters(2);
l1 = plant_parameters(3);

% symbols
S = sym('S', [3 time_points]);
L = sym('L', [num_constraints, 1]);

% cost functions
costs = cell(time_points*3-1,1);
for n = 1:time_points
    costs{3*(n-1) + 1} = 10*(S(1,n) - set_point(1))^2;
    costs{3*(n-1) + 2} = (S(2,n) - set_point(2))^2;
    if n ~= time_points
        costs{3*(n-1) + 3} =  (S(3,n))^2;
    end
end

% constraint functions 
constraints = cell(num_constraints,1);
for n = 2:time_points
    constraints{2*(n-2)+1} = L(2*(n-2)+1) * (S(1,n) - S(1,n-1) - S(2, n-1)*control_interval);
    constraints{2*(n-2)+2} = L(2*(n-2)+2) * (S(2,n) - S(2,n-1) - (-1*g/l1*sin(S(1,n-1)) - 0.7 * S(2,n-1) + S(3,n-1) )*control_interval);
end
constraints{2*(n-1)+1} = L(2*(n-1)+1) * (S(1,1) - start_position(1))^2;
constraints{2*(n-1)+2} = L(2*(n-1)+2) * (S(2,1) - start_position(2))^2;
constraints{2*(n-1)+3} = L(2*(n-1)+3) * (S(1,time_points) - set_point(1))^2;
constraints{2*(n-1)+4} = L(2*(n-1)+4) * (S(2,time_points) - set_point(2))^2;

% del F -- derivative of cost function 
del_F = cell(time_points*3-1,1);
del_F(:,:) = {0};
for n = 1:time_points
    for c = 1:length(costs)
        del_F{3*(n-1)+1} = del_F{3*(n-1)+1} + diff(costs{c}, S(1,n));
        del_F{3*(n-1)+2} = del_F{3*(n-1)+2} + diff(costs{c}, S(2,n));
        if n ~= time_points
            del_F{3*(n-1)+3} = del_F{3*(n-1)+3} + diff(costs{c}, S(3,n));
        end
    end
end

% del L -- derivative of the lagrangian : (costs + lambda*constraints) -- this is also called g
del_L = del_F;
for n = 1:time_points
    for c = 1:length(constraints)
        del_L{3*(n-1)+1} = del_L{3*(n-1)+1} + diff(constraints{c}, S(1,n));
        del_L{3*(n-1)+2} = del_L{3*(n-1)+2} + diff(constraints{c}, S(2,n));
        if n ~= time_points
            del_L{3*(n-1)+3} = del_L{3*(n-1)+3} + diff(constraints{c}, S(3,n));
        end
    end
end

% G - the Jacobian (matrix of the derivative of each constraint in terms of states
G = cell(num_constraints, num_states);
for n = 1:time_points
    for c = 1:num_constraints
        G{c, 3*(n-1) + 1} = diff(constraints{c}, S(1,n));
        G{c, 3*(n-1) + 2} = diff(constraints{c}, S(2,n));
        if n ~= time_points
            G{c, 3*(n-1) + 3} = diff(constraints{c}, S(3,n));
        end
    end
end

% Hl - Hessian of the Legrangian -- or derivative of del_L
Hl = cell(num_states, num_states);
for n1 = 1:num_states
    for n2 = 1:time_points
        Hl{n1,3*(n2-1)+1} = diff(del_L{n1}, S(1,n2));
        Hl{n1,3*(n2-1)+2} = diff(del_L{n1}, S(2,n2));
        if n2 ~= time_points
            Hl{n1,3*(n2-1)+3} = diff(del_L{n1}, S(3,n2));
        end
    end
end



t2 = toc(t1);
disp(['Simultaneous_control setup took: ' num2str(t2) ' seconds.'])


state_history = zeros(iterations+1, num_states);
t1 = tic;
for i = 1:iterations
    % eval what we found symbolically before
    eval_constraints = evaluate_symbolic(constraints, state_history(i,:), lambdas);
    eval_G  = evaluate_symbolic(G,  state_history(i,:), lambdas);
    eval_delL = evaluate_symbolic(del_L, state_history(i,:), lambdas);
    eval_Hl = evaluate_symbolic(Hl, state_history(i,:), lambdas);
    
    
    % Put together KKT matrix
    KKT = [eval_Hl, eval_G.'; eval_G,  zeros(size(eval_G,1), size(eval_Hl,2)+size(eval_G,1) - size(eval_G,2)) ];
    performance_mat = [eval_delL; eval_constraints];
    
    update_amount = -1* pinv(KKT)*performance_mat;    
    state_history(i+1,:) = state_history(i,:) + update_amount(1:num_states)';
    lambdas = lambdas + update_amount(num_states+1:end)';
    TIMINGS(i)= toc(t1);
    disp(['Simultaneous_control iteration ' num2str(i) ' took: ' num2str(TIMINGS(i)) ' seconds.'])
end

if PLOTTING
    figure()
    subplot(3,1,1)
    plot(state_history(:,1:3:num_states)')
    xlabel('Control Step')
    ylabel('Angle')
    subplot(3,1,2)
    plot(state_history(:,2:3:num_states)')
    xlabel('Control Step')
    ylabel('Angular Velocity')
    subplot(3,1,3)
    plot(state_history(:,3:3:num_states)')
    xlabel('Control Step')
    ylabel('Control Signal')
end

control_sequence = state_history(end,3:3:num_states);

end


function [evaluated_expression] = evaluate_symbolic(expression, states, lambdas)
[rows, cols] = size(expression);
evaluated_expression = zeros(rows, cols);
for n1=1:rows
    for n2=1:cols
        eqn = expression{n1,n2};
        symbols = symvar(eqn);
        for i=1:length(symbols)
            symbol = char(symbols(i));
            if symbol(1) == 'L'
                value = lambdas(str2double(symbol(2:end)));
            elseif symbol(1) == 'S'
                dof = str2double(symbol(2));
                time = str2double(symbol(4:end));
                value = states((time-1)*3 + dof);
            end
            
            eqn = subs(eqn, symbol, value); 
        end
        evaluated_expression(n1,n2) = eqn;
    end
end

end
