clear all; close all; clc;

set(groot,'defaultLineLineWidth',0.8)

X=[1, 2, 4];


Y_1=[42.98,83.54,59.75];

Y_2=[42.35,82.80,63.21];

Y_3=[40.05,83.62,58.10];

Y_4=[41.7933333, 83.32, 60.3633333]; %average


figure(1);

title('Gap percentages for different \Delta and Seeds');


plot(X,Y_1,'-o',X,Y_2,'-o',X,Y_3,'-o',X,Y_4,'-o')

grid minor


xlabel('Number of \Delta = 1, 2, 4');

ylabel('Gap value (%)');

legend('Location','southeast')
legend('Seed = 4','Seed = 6','Seed = 8', 'Average')
