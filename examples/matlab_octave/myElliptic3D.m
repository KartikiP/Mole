% 3D Staggering example using a 3D Mimetic laplacian
% Modified to have BC = 100 on both front and back faces
%
clc
close all
addpath('../../src/matlab_octave')

k = 2; % Order of accuracy
m = 5; % -> 7 grid points in x
n = 6; % -> 8 grid points in y
o = 7; % -> 9 grid points in z

% Boundary type
% dc = 1 means Dirichlet, nc = 0 means no Neumann
dc = [1;1;1;1;1;1];  % All Dirichlet boundaries
nc = [0;0;0;0;0;0];  % No Neumann boundaries

% Boundary condition values
bcl = zeros(n*o,1);              % Left face (x=0): u = 0
bcr = zeros(n*o,1);              % Right face (x=1): u = 0
bcb = zeros((m+2)*o,1);          % Bottom face (y=0): u = 0
bct = zeros((m+2)*o,1);          % Top face (y=1): u = 0
bcf = 100*ones((n+2)*(m+2),1);   % Front face (z=0): u = 100  [KEPT AS 100]
bcz = 100*ones((n+2)*(m+2),1);   % Back face (z=1): u = 100   [CHANGED FROM 0 TO 100]

% Collect boundary conditions in cell array
v = {bcl; bcr; bcb; bct; bcf; bcz};

% Build 3D Mimetic Laplacian operator
L = lap3D(k, m, 1, n, 1, o, 1, dc, nc);

% Right-hand side (forcing function = 0)
RHS = zeros(m+2, n+2, o+2);
RHS = reshape(RHS, [], 1);

% Add boundary conditions to linear system
[L0, RHS0] = addScalarBC3D(L, RHS, k, m, 1, n, 1, o, 1, dc, nc, v);

% Solve the linear system
SOL = L0\RHS0;
SOL = reshape(SOL, m+2, n+2, o+2);

% Visualize multiple slices
figure('Position', [100, 100, 1200, 400])

% Display three different z-slices
for i = 1:3
    subplot(1, 3, i)
    if i == 1
        p = 2;  % Near front
    elseif i == 2
        p = round((o+2)/2);  % Middle
    else
        p = o+1;  % Near back
    end
    
    page = SOL(:, :, p);
    imagesc(page)
    title(['z-slice at page ' num2str(p)])
    xlabel('x-direction (m+2 points)')
    ylabel('y-direction (n+2 points)')
    set(gca, 'YDir', 'Normal')
    colorbar
    caxis([0 100])  % Fix color scale for comparison
end

set(gcf, 'Color', 'w')

% Display solution statistics
fprintf('\n=== Solution Statistics ===\n')
fprintf('Minimum value: %.4f\n', min(SOL(:)))
fprintf('Maximum value: %.4f\n', max(SOL(:)))
fprintf('Mean value: %.4f\n', mean(SOL(:)))

% Check boundary values
fprintf('\n=== Boundary Values ===\n')
fprintf('Front face (z=1): min=%.2f, max=%.2f, mean=%.2f\n', ...
    min(min(SOL(:,:,1))), max(max(SOL(:,:,1))), mean(mean(SOL(:,:,1))))
fprintf('Back face (z=%d): min=%.2f, max=%.2f, mean=%.2f\n', ...
    o+2, min(min(SOL(:,:,o+2))), max(max(SOL(:,:,o+2))), mean(mean(SOL(:,:,o+2))))
fprintf('Left face (x=1): min=%.2f, max=%.2f\n', ...
    min(min(SOL(1,:,:))), max(max(SOL(1,:,:))))
fprintf('Right face (x=%d): min=%.2f, max=%.2f\n', ...
    m+2, min(min(SOL(m+2,:,:))), max(max(SOL(m+2,:,:))))

% 3D visualization
figure
[X, Y, Z] = meshgrid(1:m+2, 1:n+2, 1:o+2);
slice(X, Y, Z, permute(SOL, [2,1,3]), [1, m+2], [1, n+2], [1, o+2])
xlabel('x')
ylabel('y')
zlabel('z')
title('3D Solution: u = 100 on front and back faces')
colorbar
set(gcf, 'Color', 'w')
