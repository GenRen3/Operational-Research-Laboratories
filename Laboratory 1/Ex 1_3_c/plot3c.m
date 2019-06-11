clear all; close all; clc;

set(groot,'defaultLineLineWidth',0.8)

X=[8, 16, 24];


Y_1=[43.65,90.83,96.11];

Y_2=[43.87,91.13,95.58]; 

Y_3=[46.62,92.54,96.00]; 

Y_4=[44.713, 91.5, 95.8966667] %average


figure(1);

title('Objective funtions with different seeds');


semilogy(X,Y_1,'-o',X,Y_2,'-o',X,Y_3,'-o',X,Y_4,'-o')

grid minor


xlabel('Number of Nodes = 8, 16, 24');

ylabel('Gap value}');

legend('Seed = 4','Seed = 6','Seed = 8', 'Average')
