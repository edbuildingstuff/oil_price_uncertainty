clear; clc; 

load('opu_baseline.mat')

dlmwrite('opu_baseline.txt', opu_baseline, 'delimiter','\t','newline','pc')
