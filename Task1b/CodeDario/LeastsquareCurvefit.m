clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
imtool close all;	% Close all figure windows created by imtool.

datapath = 'C:\Users\dario\Documents\Intro-to-ML\Task1b\Raw_data\';

Data = csvread(fullfile(datapath, 'train.csv'), 1, 1);

y_train = Data(:, 1);
X = Data(:, 2:6);
[row, col] = size(y_train);

%% Fit gaussian functions for red, green and blue channel:    
my_option = optimset('display','iter','TolFun', 1e-100, 'TolX', 1e-100, 'MaxIter',200, 'MaxFunEvals',10000);
a = 91;
weights = [a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a];
fun = @(weights, X)(weights(1,1).*X(:,1) + weights(1,2).*X(:,2) + weights(1,3).*X(:,3) + weights(1,4).*X(:,4) + weights(1,5).*X(:,5)...
     + weights(1,6).*X(:,1).^2 + weights(1,7).*X(:,2).^2 + weights(1,8).*X(:,3).^2 + weights(1,9).*X(:,4).^2 + weights(1,10).*X(:,5).^2 ...
     + weights(1,11).*exp(X(:,1)) + weights(1,12).*exp(X(:,2)) + weights(1,13).*exp(X(:,3)) + weights(1,14).*exp(X(:,4)) + weights(1,15).*exp(X(:,5)) ...
     + weights(1,16).*cos(X(:,1)) + weights(1,17).*cos(X(:,2)) + weights(1,18).*cos(X(:,3)) + weights(1,19).*cos(X(:,4)) + weights(1,20).*cos(X(:,5)) ...
     + weights(1,21).*1); 
weights =lsqcurvefit(fun,weights,X,y_train,[],[],my_option);
disp(weights);
weights = weights';
csvwrite('result.csv',weights);


