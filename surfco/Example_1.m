
function Example_1()
    %%%%% Script to run Example 1: Surface mesh from cardiac contours %%%%%
    %
    %  File is part of SurFCo package: www.github.com/benvillard/surfco. 
    %
    %  Please see the following papers for an explanation of the various 
    %  parameters: 
    %  
    %  [1] B. Villard, V. Grau, and E. Zacur, Surface mesh reconstruction from 
    %  cardiac MRI contours, J. Imaging, vol. 4(1), no. 16, 2018.
    % 
    %   [2]  B. Villard, V. Carapella, R. Ariga,  V. Grau, and E. Zacur, 
    %   Cardiac Mesh Reconstruction from Sparse, Heterogeneous Contours. 
    %   In: Valdés Hernández M., González-Castro V. (Eds.) Medical Image 
    %   Understanding and Analysis. MIUA 2017. 
    %   Communications in Computer and Information Science, 
    %   Vol. 723. Springer, Cham
    
    
    fileName = 'parameters.json'; % filename in JSON extension
    fid = fopen(fileName); % Opening the file
    raw = fread(fid,inf); % Reading the contents
    str = char(raw'); % Transformation
    fclose(fid); % Closing the file
    parameters = jsondecode(str); % Using the jsondecode function to parse JSON from string
    names = split(parameters.name,".");
    name = string(names(1));

    types = ["MYO", "LV"];
    for type = types
        filename = "Output/"+name+"/"+type+"_point_cloud.xyz";
        M = csvread(filename);
        M = M(:, 1:3);

        % [M, f, n, c, stltitle] = stlread2('model11.stl');

        num_of_pts = sum(M(:,3)==M(1,3));
        group_of_pts = length(M)/num_of_pts;

        cloud = {};
        for i = 1:group_of_pts
            cloud = [cloud; M(1*num_of_pts*(i-1)+1:num_of_pts*i,:)];
        end

        % Run Mesh Reconstruction Algorithm
        M = SurFCo(cloud, 'getLids', 'plot' ); 
        % Write to stl
        P = M.xyz;
        V = M.tri;

        if type == "MYO"
            myo_x_max = max(P(:, 1));
            myo_x_min = min(P(:, 1));
            myo_y_max = max(P(:, 2));
            myo_y_min = min(P(:, 2));
            myo_z_max = max(P(:, 3));
            myo_z_min = min(P(:, 3));
            
            myo_mid_x = (myo_x_max - myo_x_min)/2;
            myo_mid_y = (myo_y_max - myo_y_min)/2;
            myo_mid_z = (myo_z_max - myo_z_min)/2;
        elseif type == "LV"
            lv_x_max = max(P(:, 1));
            lv_x_min = min(P(:, 1));
            lv_y_max = max(P(:, 2));
            lv_y_min = min(P(:, 2));
            lv_z_max = max(P(:, 3));
            lv_z_min = min(P(:, 3));
            
            lv_mid_x = (lv_x_max - lv_x_min)/2;
            lv_mid_y = (lv_y_max - lv_y_min)/2;
            lv_mid_z = (lv_z_max - lv_z_min)/2;
            
            P(:, 1) = P(:, 1) + abs(myo_mid_x-lv_mid_x);
            P(:, 2) = P(:, 2) + abs(myo_mid_y-lv_mid_y);
%             P(:, 3) = P(:, 3) + abs(myo_mid_z-lv_mid_z);

            lv_z_max = max(P(:, 3));
            lv_z_min = min(P(:, 3));
%             
%             if myo_x_max - lv_x_max <0 || myo_x_min-lv_x_min>0 || myo_y_max - lv_y_max <0 || myo_y_min-lv_y_min>0 || myo_z_max - lv_z_max <0 || myo_z_min-lv_z_min>0
%                 wad
%                 P(:, 1) = P(:, 1) + abs(myo_mid_x-lv_mid_x);
%                 P(:, 2) = P(:, 2) + abs(myo_mid_y-lv_mid_y);
%             end
        end

        stlwrite("Output/"+name+"/"+type+"_surface.stl", V, P);
        close all
    end