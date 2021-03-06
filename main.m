function main(controller_type)
%MAIN Summary of this function goes here
%   Detailed explanation goes here

    disp('Setting up plant parameters')
    m1 = 1;
    m2 = 1;
    l1 = 1;
    l2 = 1;
    g = 9.81;
    plant_parameters = [m1, m2, l1, l2, g];
    
    disp('Setting up initial conditions')
    IC = [0,0,0,0];
    
    disp('Setting up final conditions')
    BC = [pi, pi ,0,0];
    
    disp(['Proceeding to control double pendulum using ' controller_type ' control.'])
    control_resolution = 0.1; % seconds
    control_duration   = 5;   % seconds 
    
    switch controller_type 
        case 'none'
            disp('no controller implemented')
            
        case 'simultaneous'
            disp('Beginning simultaneous control')
            iterations = 5;
            simultaneous_controller = Simultaneous_Control(plant_parameters, IC, BC, control_resolution, control_duration, iterations);
            control_sequence = simultaneous_controller.get_control_sequence();

    end
end



    

