clear all;
addpath('./KittiRaw/utils');
mkdir('testing');
Eigen_Test_File = './KittiRaw/utils/split/eigen_test_files.txt';

folder = './KittiRaw/';

fileID = fopen(Eigen_Test_File);
C = textscan(fileID,'%s %s');

left_all = C{1};right_all = C{2};
len = length(left_all);
if(len~=697)
    error('Size not match for Eigen Depth Split...\n');
end

cam = 2;

for i = 1:len
    fprintf('Processing image %d/%d...\n',i,len);
    left = imread([folder left_all{i}(1:end-4) '.png']);
    right = imread([folder right_all{i}(1:end-4) '.png']);
    
    calib_dir = [folder left_all{i}(1:10)];
    base_dir = [folder left_all{i}(1:37)];
    frame = ( left_all{i}(end-13:end-4) );
    
    % load calibration
    calib = loadCalibrationCamToCam(fullfile(calib_dir,'calib_cam_to_cam.txt'));
    Tr_velo_to_cam = loadCalibrationRigid(fullfile(calib_dir,'calib_velo_to_cam.txt'));

    %get focal length and baseline for restoring disparity to depth.
    b2 = calib.P_rect{cam+1}(1,4) / -calib.P_rect{cam+1}(1,1);
    b3 = calib.P_rect{cam+2}(1,4) / -calib.P_rect{cam+2}(1,1);
    baseline = b3-b2;

    focal_length = calib.P_rect{cam+1}(1,1);

    % compute projection matrix velodyne->image plane
    R_cam_to_rect = eye(4);
    R_cam_to_rect(1:3,1:3) = calib.R_rect{1};
    P_velo_to_img = calib.P_rect{cam+1}*R_cam_to_rect*Tr_velo_to_cam;
    
    fid = fopen(sprintf('%s/velodyne_points/data/%s.bin',base_dir,frame),'rb');
    velo = fread(fid,[4 inf],'single')';
    fclose(fid);
    
    %remove some first
    velo(velo(:,1)<0,:) = [];
    velo_img = project(velo(:,1:3),P_velo_to_img);
    velo_img(:,3) = velo(:,1); %put the depth to 2D projection points
    
    %check all in bound
    velo_img(:,1) = round(velo_img(:,1));velo_img(:,2) = round(velo_img(:,2)); 
    val_ind = (velo_img(:,1) >0) & (velo_img(:,2) >0) & (velo_img(:,1) <=size(left,2)) & (velo_img(:,2) <=size(left,1));
    velo_img = velo_img(val_ind,:);
    
    depth = zeros([size(left,1) size(left,2)],'double');
    for l = 1:size(velo_img,1)
        if( depth(velo_img(l,2),velo_img(l,1)) ~= 0)
            if( depth(velo_img(l,2),velo_img(l,1)) > velo_img(l,3))
                depth(velo_img(l,2),velo_img(l,1)) = velo_img(l,3);
            else    
                fprintf('Find one not replace...\n');
            end
        else
            depth(velo_img(l,2),velo_img(l,1)) = velo_img(l,3);
        end    
    end    
  
    %% plot points
    % img = left;
    % fig = figure('Position',[20 100 size(img,2) size(img,1)]); axes('Position',[0 0 1 1]);
    % imshow(img); hold on;
    % cols = jet;
    % for j=1:size(velo_img,1)
    %   col_idx = round(64/(velo(j,1)+1));
    %   plot(velo_img(j,1),velo_img(j,2),'o','LineWidth',4,'MarkerSize',1,'Color',cols(col_idx,:));
    % end
    test.name = 'Eigen';
    test.left{i}=left;
    test.leftAdd{i} = [folder left_all{i}(1:end-4) '.png'];
    test.right{i}=right;
    test.rightAdd{i} = [folder right_all{i}(1:end-4) '.png'];
    test.depth{i}=depth;
    test.baseline{i} = baseline;
    test.focal{i} = focal_length;
    test.length = 697;
end   

fprintf('    Writing testing set...');
filename = sprintf('./testing/testKittiEigen.mat');
save(filename,'test','-v7.3');
clear test;
    