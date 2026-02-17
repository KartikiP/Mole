% Solves the 2D Poisson problem on [0,1]^2
% -∇²u = f(x,y) with mixed boundary conditions
clc
close all
addpath('/MATLAB Drive/Mole/src/matlab_octave')

% Parameters
k = 2;
m = 20; n = 20;
dx = 1/m;
dy = 1/n;

% Exact solution and forcing function
uex = @(X, Y) pi*(X.^2 + Y.^2).*sin(pi*X.*Y);
f = @(X, Y) pi*((pi^2*(X.^2 + Y.^2).^2 - 4).*sin(pi*X.*Y) - 8*pi*X.*Y.*cos(pi*X.*Y));

% Grid
nx = m + 2;
ny = n + 2;
x = linspace(0, 1, nx);
y = linspace(0, 1, ny);
[X, Y] = meshgrid(x, y);

fprintf('Building Laplacian operator...\n')

% Boundary condition types
% Set all to Dirichlet to enforce exact values
dc = [0; 0; 1; 1];  % [left; right; bottom; top] - all Dirichlet
nc = [1; 1; 1; 0];  % [left; right; bottom; top] - no Neumann

% Left (x=0): Neumann ∂u/∂n = -π²y³
y_interior = y(2:end-1);
bcl = (-pi^2 * y_interior.^3)';

% Right (x=1): Neumann ∂u/∂n = π(2sin(πy) + πy(1+y²)cos(πy))
bcr = pi*(2*sin(pi*y_interior) + pi*y_interior.*(1+y_interior.^2).*cos(pi*y_interior))';

% Bottom (y=0): Robin u + ∂u/∂n = -π²x³
bcb = (-pi^2 * x.^3)';

% Top (y=1): Dirichlet u = π(x²+1)sin(πx)
bct = (pi*(x.^2 + 1).*sin(pi*x))';
% Collect boundary conditions
v = {bcl; bcr; bcb; bct};

% Build Laplacian
L = lap2D(k, m, dx, n, dy);

% Right-hand side
RHS = -f(X, Y);
RHS = reshape(RHS.', [], 1);


% Add boundary conditions
[L0, RHS0] = addScalarBC2D(L, RHS, k, m, dx, n, dy, dc, nc, v);
U = L0\RHS0;
U = reshape(U, nx, ny)';

% Exact solution
U_exact = uex(X, Y);


% Plot
figure('Position', [100, 100, 1400, 400])

subplot(1, 3, 1)
surf(X, Y, U_exact, 'EdgeColor', 'none')
title('Exact Solution')
xlabel('x'); ylabel('y'); zlabel('u')
colorbar
set(gcf, 'Color', 'w')
view([-45 30])
zlim([0 max(U_exact(:))])

subplot(1, 3, 2)
surf(X, Y, U, 'EdgeColor', 'none')
title('Numerical Solution')
xlabel('x'); ylabel('y'); zlabel('u')
colorbar
view([-45 30])
zlim([0 max(U(:))])

