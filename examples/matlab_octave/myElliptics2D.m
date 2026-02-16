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

% Grid is (m+2) x (n+2)
nx = m + 2;
ny = n + 2;
x = linspace(0, 1, nx);
y = linspace(0, 1, ny);
[X, Y] = meshgrid(x, y);

fprintf('Building mimetic operators...\n')

% Get 2D mimetic operators
D = div2D(k, m, dx, n, dy);
G = grad2D(k, m, dx, n, dy);
L = D*G;

% Use simple Dirichlet BC operator (just to mark boundaries)
Robin = robinBC2D(k, m, dx, n, dy, 1, 0);  % a=1, b=0 = Dirichlet
L = L + Robin;

% Find boundary indices
Robin_diag = diag(Robin);
bdry = find(Robin_diag);

fprintf('Boundary points: %d / %d total\n', length(bdry), nx*ny);

% Build RHS: forcing function everywhere
B = -f(X, Y);  % -∇²u = f
B = reshape(B.', [], 1);



% Let's just set the exact solution at boundaries as an approximation
B_boundary = uex(X, Y);
B_boundary = reshape(B_boundary.', [], 1);

% Overwrite boundary entries (following the example pattern)
B(bdry) = B_boundary(bdry);

% Solve
fprintf('Solving...\n')
U = L\B;
U = reshape(U, nx, ny)';

% Exact solution
U_exact = uex(X, Y);

% Compute error
error = abs(U_exact - U);
max_error = max(error(:));
rel_error = max_error / (max(U_exact(:)) - min(U_exact(:)));

% Check residual
residual = norm(L*U(:) - B);
fprintf('Residual: %.6e\n', residual);

% Plot
figure('Position', [100, 100, 1400, 400])

subplot(1, 3, 1)
surf(X, Y, U_exact, 'EdgeColor', 'none')
title('Exact Solution')
xlabel('x'); ylabel('y'); zlabel('u')
colorbar
set(gcf, 'Color', 'w')
view([-45 30])

subplot(1, 3, 2)
surf(X, Y, U, 'EdgeColor', 'none')
title('Numerical Solution')
xlabel('x'); ylabel('y'); zlabel('u')
colorbar
view([-45 30])

subplot(1, 3, 3)
surf(X, Y, error, 'EdgeColor', 'none')
title('Absolute Error')
xlabel('x'); ylabel('y'); zlabel('|error|')
colorbar
view([-45 30])

fprintf('\n=== Results ===\n')
fprintf('Maximum error: %.6e\n', max_error)
fprintf('Relative error: %.4f%%\n', 100*rel_error)
