%% Workspace Clear
clear; clc;close all

%% ==== Load the CORA toolbox first to avoid conflicts with other same-named methods ====
cora_path = 'D:\PersonalFiles\ResearchWork\CORA-master';  % <- Replace this with your own CORA path

if exist(cora_path, 'dir')
    % Add CORA to the beginning of the search path without affecting existing toolboxes
    addpath(genpath(cora_path), '-begin');
    rehash toolboxcache;  % Refresh the toolbox cache
else
    warning('CORA doesn''t exist: %s', cora_path);
end

%% Compute zonotope enclosures for omega / gamma error points in parallel across prediction steps
file_name = "2026_06_29_00_39_02";
data_dir  = "error_points_multistep/" + file_name;
Nfut      = 15;

zono_dir = fullfile(data_dir, "zonotopes");
if ~exist(zono_dir, "dir")
    mkdir(zono_dir);
end

pool = gcp("nocreate");
if isempty(pool)
    parpool("Processes", 4);
end

parfor k = 1:Nfut
    step_name = sprintf("step%02d", k);

    % omega: high-dimensional multi-step prediction error
    omega = readmatrix( ...
        fullfile(data_dir, "omega_full_set_" + step_name + ".csv"))';

    Zono_omega = zonotope.enclosePoints(omega);

    writematrix( ...
        Zono_omega.c, ...
        fullfile(zono_dir, "Zono_omega_" + step_name + "_c.csv"));

    writematrix( ...
        Zono_omega.G, ...
        fullfile(zono_dir, "Zono_omega_" + step_name + "_G.csv"));

    fprintf("[k=%2d] omega/gamma zonotope 已保存 (%s)\n", ...
        k, step_name);
end

% gamma: decoding reconstruction error
gamma = readmatrix( ...
    fullfile(data_dir, "gamma_full_set.csv"))';

Zono_gamma = zonotope.enclosePoints(gamma);

writematrix( ...
    Zono_gamma.c, ...
    fullfile(zono_dir, "Zono_gamma_c.csv"));

writematrix( ...
    Zono_gamma.G, ...
    fullfile(zono_dir, "Zono_gamma_G.csv"));

