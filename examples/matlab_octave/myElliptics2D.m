% Solves the 2D Poisson problem on [0,1]^2
% -∇²u = f(x,y) with mixed boundary conditions
clc
close all
addpath('../../src/matlab_octave')

% Parameters
k = 2; 
m = 20; n = 20;  % Increased resolution for better accuracy

% Create uniform grid on [0,1]^2
x = linspace(0, 1, m+1);
y = linspace(0, 1, n+1);
[X, Y] = meshgrid(x, y);   % nodal grid (n+1 by m+1)

% Exact solution and forcing function
uex = @(X, Y) pi*(X.^2 + Y.^2).*sin(pi*X.*Y);
f = @(X, Y) pi*((pi^2*(X.^2 + Y.^2).^2 - 4).*sin(pi*X.*Y) - 8*pi*X.*Y.*cos(pi*X.*Y));

% Compute boundary condition functions from exact solution
% Left boundary (x=0): ∂u/∂n(0,y) = -∂u/∂x(0,y)
gl = @(Y) -pi*(2*0.*sin(pi*0.*Y) + pi*0.*(0.^2 + Y.^2).*cos(pi*0.*Y));
gl = @(Y) 0*Y;  % Simplifies to 0

% Right boundary (x=1): ∂u/∂n(1,y) = ∂u/∂x(1,y)
gr = @(Y) pi*(2*1.*sin(pi*1.*Y) + pi*Y.*(1.^2 + Y.^2).*cos(pi*1.*Y));

% Bottom boundary (y=0): u(x,0) + ∂u/∂n(x,0) = u(x,0) - ∂u/∂y(x,0)
gb = @(X) pi*(X.^2 + 0.^2).*sin(pi*X.*0) - pi*(2*0.*sin(pi*X.*0) + pi*X.*(X.^2 + 0.^2).*cos(pi*X.*0));
gb = @(X) 0*X;  % Simplifies to 0

% Top boundary (y=1): u(x,1)
gt = @(X) uex(X, 1);

% Visualize the grid
figure(1)
mesh(X, Y, zeros(n+1, m+1), 'Marker', '.', 'MarkerSize', 10)
view([0 90])
axis tight
set(gcf, 'Color', 'w')
title('Nodal Grid')

[n, m] = size(X);
n = n - 1;
m = m - 1;

% u-component staggered points (vertical edges)
Ux = (X(1:end-1, :) + X(2:end, :))/2;
Uy = (Y(1:end-1, :) + Y(2:end, :))/2;

% v-component staggered points (horizontal edges)
Vx = (X(:, 1:end-1) + X(:, 2:end))/2;
Vy = (Y(:, 1:end-1) + Y(:, 2:end))/2;

% Cell center points
Cx = (Vx(1:end-1, :) + Vx(2:end, :))/2;
Cy = (Uy(:, 1:end-1) + Uy(:, 2:end))/2;

% Extend to include boundary centers
% West-East sides
Cx = [Ux(:, 1) Cx];
Cy = [Uy(:, 1) Cy];
Cx = [Cx Ux(:, end)];
Cy = [Cy Uy(:, end)];

% South-North sides
Cx = [[X(1,1) Vx(1, :) X(1,end)]; Cx];
Cy = [[Y(1,1) Vy(1, :) Y(1,end)]; Cy];
Cx = [Cx; [X(end,1) Vx(end, :) X(end,end)]];
Cy = [Cy; [Y(end,1) Vy(end, :) Y(end,end)]];

% Get curvilinear mimetic operators
fprintf('Building operators...\n')
tic
D = div2DCurv(k, X, Y);
fprintf('Divergence operator: ')
toc

tic
G = grad2DCurv(k, X, Y);
fprintf('Gradient operator: ')
toc

tic
L = D*G;
fprintf('Laplacian operator: ')
toc

% Apply boundary conditions
% robinBC2D(k, m, a_left, n, a_bottom, a_right, a_top)
% For Neumann BC: use a=1, b=0 (∂u/∂n = g means u + 0*∂u/∂n = g at boundary)
% For Robin BC: use a=1, b=1 (u + ∂u/∂n = g)
% For Dirichlet BC: use a=0, b=1 (0*u + 1*∂u/∂n = 0, then impose u=g directly)

% Left: Neumann (∂u/∂n), Right: Neumann, Bottom: Robin (u + ∂u/∂n), Top: Dirichlet
Robin = robinBC2D(k, m, 1, n, 1, 1, 0);  % Adjust parameters as needed
L = L + Robin;

figure(2)
spy(L)
title('Laplacian with Boundary Conditions')

% Set up right-hand side
Robin = diag(Robin);
bdry = find(Robin);

% Interior forcing function
B = -f(Cx, Cy);  % Note: -∇²u = f, so we use -f
B = reshape(B.', [], 1);

% Apply boundary conditions
% The boundary indices need to be identified correctly
% Assuming standard ordering: (i,j) -> i + j*(m+2)

% Bottom boundary (j=1): u + ∂u/∂n = gb
for i = 1:m+2
    idx = i + 0*(m+2);
    B(idx) = gb(Cx(1, i));
end

% Top boundary (j=n+2): u = gt (Dirichlet)
for i = 1:m+2
    idx = i + (n+1)*(m+2);
    B(idx) = gt(Cx(end, i));
end

% Left boundary (i=1): ∂u/∂n = gl (Neumann)
for j = 2:n+1
    idx = 1 + (j-1)*(m+2);
    B(idx) = gl(Cy(j, 1));
end

% Right boundary (i=m+2): ∂u/∂n = gr (Neumann)
for j = 2:n+1
    idx = (m+2) + (j-1)*(m+2);
    B(idx) = gr(Cy(j, end));
end

% Solve the system
fprintf('Solving linear system...\n')
Comp = L\B;
Comp = reshape(Comp, m+2, n+2)';

% Exact solution at centers
Exact = uex(Cx, Cy);

% Plot results
figure(3)
subplot(1, 3, 1)
surf(Cx, Cy, Exact, 'EdgeColor', 'none')
title('Exact Solution')
xlabel('x')
ylabel('y')
zlabel('u')
set(gcf, 'Color', 'w')
colorbar

subplot(1, 3, 2)
surf(Cx, Cy, Comp, 'EdgeColor', 'none')
title('Numerical Solution')
xlabel('x')
ylabel('y')
zlabel('u')
colorbar

subplot(1, 3, 3)
surf(Cx, Cy, Exact - Comp, 'EdgeColor', 'none')
title('Error')
xlabel('x')
ylabel('y')
zlabel('error')
colorbar

% Show error map
figure(4)
surf(Cx, Cy, abs(Exact - Comp), 'EdgeColor', 'none')
title('Absolute Error')
view([0 90])
xlabel('x')
ylabel('y')
set(gcf, 'Color', 'w')
colorbar

% Error metrics
max_error = max(max(abs(Exact - Comp)));
rel_error = max_error / (max(max(Exact)) - min(min(Exact)));

fprintf('\n--- Error Analysis ---\n')
fprintf('Maximum absolute error: %.6e\n', max_error)
fprintf('Relative error: %.6f%%\n', 100*rel_error)
fprintf('L2 norm of error: %.6e\n', norm(Exact(:) - Comp(:), 2)/sqrt(numel(Exact)))
