% Solves the 1D Poisson's equation with Robin boundary conditions
% same example as elliptic1D that uses addScalarBC1D
%

clc
close all

addpath('../../src/matlab_octave')

west = 0;  % Domain's limits
east = 1;

k = 4;  % Operator's order of accuracy
m = 10;  % Minimum number of cells to attain the desired accuracy
dx = (east-west)/m;  % Step length
grid = [west west+dx/2 : dx : east-dx/2 east]; % grid

% Robin boundary conditions
dc = [0; 1];   % Dirichlet coefficients
nc = [1; 0];   % Neumann coefficients
v  = [2; 1];   % RHS values for the BC


L = lap(k, m, dx);  % 1D Mimetic laplacian operator
U = ones(size(grid))'; % RHS
[L0,U0] = addScalarBC1D(L,U,k,m,dx,dc,nc,v); % add BC to system
U0 = L0\U0;  % Solve a linear system of equations

% Plot result
plot(grid, U0, 'o')
hold on
plot(grid, exp(grid))
legend('Approximated', 'Analytical', 'Location', 'NorthWest')
title('Poisson''s equation with Robin BC')
xlabel('x')
ylabel('u(x)')
