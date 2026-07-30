%% Workspace Clear 
clear; clc; close all;

%% ==== Load the CORA toolbox first to avoid conflicts with same-named methods in zonolab/MPT/YALMIP ====
cora_path = 'D:\PersonalFiles\ResearchWork\CORA-master\CORA-master';  % <- Replace this with your own CORA path
if exist(cora_path, 'dir')
    addpath(genpath(cora_path), '-begin');
    rehash toolboxcache;  % Refresh the toolbox cache
else
    warning('CORA doesn''t exist: %s', cora_path);
end

% Read data
omega_c_01 = readmatrix("error_points_multistep/2026_06_29_00_39_02/zonotopes/Zono_omega_step01_c.csv");
omega_G_01 = readmatrix("error_points_multistep/2026_06_29_00_39_02/zonotopes/Zono_omega_step01_G.csv");
Zono_omega_01 = zonotope(omega_c_01, omega_G_01);

omega_c_08 = readmatrix("error_points_multistep/2026_06_29_00_39_02/zonotopes/Zono_omega_step08_c.csv");
omega_G_08 = readmatrix("error_points_multistep/2026_06_29_00_39_02/zonotopes/Zono_omega_step08_G.csv");
Zono_omega_08 = zonotope(omega_c_08, omega_G_08);

omega_c_15 = readmatrix("error_points_multistep/2026_06_29_00_39_02/zonotopes/Zono_omega_step15_c.csv");
omega_G_15 = readmatrix("error_points_multistep/2026_06_29_00_39_02/zonotopes/Zono_omega_step15_G.csv");
Zono_omega_15 = zonotope(omega_c_15, omega_G_15);

gamma_points = readmatrix("error_points_multistep/2026_06_29_00_39_02/gamma_full_set.csv");
gamma_c = readmatrix("error_points_multistep/2026_06_29_00_39_02/zonotopes/Zono_gamma_c.csv");
gamma_G = readmatrix("error_points_multistep/2026_06_29_00_39_02/zonotopes/Zono_gamma_G.csv");
Zono_gamma = zonotope(gamma_c, gamma_G);

% Start plotting
proj_dim = [1 2 3];

% Create a figure with 1-by-2 subplots
figure(1); clf;

% =========================================================================
% First subplot: gamma case
subplot(1, 2, 1);
mint = [72 209 204]/255;
hPts_gamma = plot3(gamma_points(:, proj_dim(1)), ...
                   gamma_points(:, proj_dim(2)), ...
                   gamma_points(:, proj_dim(3)), ...
                   '.', 'MarkerSize', 5, 'Color', mint);
hold on;
PeachPuff = [255 165 79]/255;
hZ_gamma = plot(Zono_gamma, proj_dim, 'Color', PeachPuff);

% Set the title and grid
grid on; grid minor;
ax = gca;
ax.Box = 'on';

% Set the legend
lgd_gamma = legend([hZ_gamma(1), hPts_gamma], ...
    {'Reconstruction error zonotope $\mathcal{Y}$', 'Reconstruction error points'}, ...
    'Interpreter', 'latex', 'Location', 'north', 'FontSize', 12);


% =========================================================================
% Second subplot: omega case
subplot(1, 2, 2);  % 1 row and 2 columns; the current subplot is the second one
mint = [72 209 204]/255;
hZ_omega_01 = plot(Zono_omega_01, proj_dim, 'Color', mint);
hold on;
Lavender = [184 169 230]/255;
hZ_omega_08 = plot(Zono_omega_08, proj_dim, 'Color', Lavender);
hold on;
PeachPuff = [255 165 79]/255;
hZ_omega_15 = plot(Zono_omega_15, proj_dim, 'Color', PeachPuff);

% Set the title and grid
grid on; grid minor;
ax = gca;
ax.Box = 'on';

% Set the legend
lgd_omega = legend([hZ_omega_01(1), hZ_omega_08(1), hZ_omega_15(1)], ...
    {'Prediction error zonotope $\mathcal{W}^{(1)}$', 'Prediction error zonotope $\mathcal{W}^{(8)}$', ...
    'Prediction error zonotope $\mathcal{W}^{(15)}$'}, 'Interpreter', 'latex', 'Location', 'north', 'FontSize', 12);

% Adjust the layout to make the two subplots more compact
set(gcf, 'Color', 'w');

% Save the figure
outname = 'error_zonotopes_omega_gamma.pdf';
saveas(gcf, outname, 'pdf');
