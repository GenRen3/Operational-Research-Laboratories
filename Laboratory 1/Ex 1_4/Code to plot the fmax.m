clear all; close all; clc;

X=[4,6,8,10,12,14];

Y_split=[40.6172, 40.5248, 39.5032, 39.7115, 40.0511, 38.7714];
Y_no_split=[114.6550, 130.1530, 112.7120, 98.3395, 116.6230, 104.0330];


figure(1);

title('Objective funtions with different seeds');

plot(X,Y_split,'-o',X,Y_no_split,'-o')

grid minor

xlabel('Random Seeds');
ylabel('f_{max}');
legend('Split','No Split')
