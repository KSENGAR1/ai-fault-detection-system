clc; clear; close all;

%%PARAMETERS 
t = 0:0.01:10;
num_samples = 100;

features = [];
labels = [];

for i = 1:num_samples
    

    normal = sin(2*pi*1*t) + 0.1*randn(size(t));
    
    % -------- FAULT 1
    fault1 = sin(2*pi*1*t) + randn(size(t));
    
    % -------- FAULT 2
    fault2 = sin(2*pi*3*t);
    
    % -------- FAULT 3
    fault3 = 2*sin(2*pi*1*t);
    
    % -------- FAULT 4
    fault4 = sin(2*pi*1*t);
    fault4(200:250) = 4;
    
    extract = @(signal) [ ...
        mean(signal), ...
        var(signal), ...
        max(signal), ...
        rms(signal), ...
        skewness(signal), ...
        kurtosis(signal) ...
    ];
    
   
    f_normal = extract(normal);
    f_fault1 = extract(fault1);
    f_fault2 = extract(fault2);
    f_fault3 = extract(fault3);
    f_fault4 = extract(fault4);
    
   
    features = [features;
                f_normal;
                f_normal;  
                f_fault1;
                f_fault2;
                f_fault3;
                f_fault4];
    
    labels = [labels;
              0;
              0;
              1;
              1;
              1;
              1];
end

disp('Dataset Created');
data = [features labels];
writematrix(data, 'dataset.csv');

disp('Dataset saved as dataset.csv');

figure;

subplot(2,1,1);
plot(t, normal, 'LineWidth',1.5);
title('Normal Signal');
grid on;

subplot(2,1,2);
plot(t, fault1, 'LineWidth',1.5);
title('Fault Signal');
grid on;