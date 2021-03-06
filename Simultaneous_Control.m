classdef Simultaneous_Control
    properties
        plant_parameters
        IC
        BC
        control_resolution
        control_duration
        iterations
        control_points
        num_states
        num_constraints
        state_sym
        
    end
    methods
        function sc = Simultaneous_Control(plant_parameters, IC, BC, control_resolution, control_duration, iterations)
            if nargin > 0
                sc.plant_parameters   = plant_parameters;
                sc.IC                 = IC;
                sc.BC                 = BC;
                sc.control_resolution = control_resolution;
                sc.control_duration   = control_duration;
                sc.iterations         = iterations;
                sc.control_points     = control_duration/control_resolution;
                sc.num_states         = 5* (sc.control_points-1)  +4;
                sc.num_constraints    = 4* (sc.control_points-1)  +8;
                
                
                theta1 = sym('t1_', [1 sc.control_points]);
                theta2 = sym('t2_', [1 sc.control_points]);
                omega1 = sym('o1_', [1 sc.control_points]);
                omega2 = sym('o2_', [1 sc.control_points]);
                u1     = sym('u1_', [1 sc.control_points-1]);
                lambdas = sym('l_', [1 sc.num_constraints]);
                sc.state_sym  = {theta1, theta2, omega1, omega2, u1, lambdas};
                
                
            end
        end
        
        function control_sequence = get_control_sequence(obj)
            
            cost_functions        = obj.get_cost();
            constraint_functions  = obj.get_constraints();
            disp('Here!')
            
        end
        
        function cost_functions = get_cost(obj)
            cost_functions = {};
            for i = 1:obj.control_points
                for n = 1:4
                    cost_functions{end+1} = (obj.state_sym{n}(i) - obj.BC(n))^2;
                end
                if i ~= obj.control_points
                    cost_functions{end+1} = (obj.state_sym{5}(i))^2;
                end
            end
        end
        
        function constraint_functions = get_constraints(obj)
            constraints_functions = {};
            for i = 2:obj.control_points
               % t1_i
               constraints{end+1}= obj.state_sym{6}(4*(i-2)+1)*(obj.state_sym{1}(i) - obj.state_sym{1}(i-1)-obj.state_sym{3}(i-1)*obj.control_resolution);
               % t2_i
               constraints{end+1}= obj.state_sym{6}(4*(i-2)+2)*(obj.state_sym{2}(i) - obj.state_sym{2}(i-1)-obj.state_sym{4}(i-1)*obj.control_resolution);
               % o1_i
               
               constraints{end+1} = obj.state_sym{6}(4*(i-1)+3)*(obj.state_sym{3}(i) - obj.state_sym{3}(i-1)-alpha1*obj.control_resolution);
               % o2_i
               
               constraints{end+1} = obj.state_sym{6}(4*(i-1)+4)*(obj.state_sym{4}(i) - obj.state_sym{4}(i-1)-alpha2*obj.control_resolution);
                
                
            end
            
            
            
        end
        
        
    end
end