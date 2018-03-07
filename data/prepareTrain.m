clear all;
mkdir('training');
%%Prepare Kitti Stereo 2015 training set(200 high-quality labels)
leftsAdd = './KittiStereo2015/training/image_2/';
rightsAdd = './KittiStereo2015/training/image_3/';
dispsAdd = './KittiStereo2015/training/disp_noc_0/';

lefts  = dir([leftsAdd  '*10.png']);
rights = dir([rightsAdd '*10.png']);
disps = dir([dispsAdd '*10.png']);

%compare num of image and ground truth
if(length(lefts)~=length(rights))
    error('Numbers are not matched... Pleae check numbers...\n');
end  

fprintf('Processing Kitti Stereo 2015 training set...\n');

for j = 1:length(lefts)
   fprintf('    Processing image %d/%d ...\n',j,length(lefts));
   leftName = [ leftsAdd lefts(j).name];
   rightName= [ rightsAdd rights(j).name];
   left = imread(leftName);
   right = imread(rightName);
   dispName = [ dispsAdd disps(j).name];
   disp = imread(dispName);
   
   gt.size = size(lefts);
   gt.train = 1; 
   gt.left{j,1} = left;
   gt.right{j,1} = right;
   gt.disp{j,1} = disp;
end

filename = sprintf('./training/trainKitti15.mat');
save(filename,'gt','-v7.3');
clear gt;



%%Prepare Kitti Eigen Training set
addpath('./KittiRaw/utils');

Eigen_Train_File = './KittiRaw/utils/split/eigen_train_files.txt';

folder = './KittiRaw/';

fileID = fopen(Eigen_Train_File);
C = textscan(fileID,'%s %s');

left_all = C{1};right_all = C{2};
len = length(left_all);
if(len~=22600)
    error('Size not match for Eigen Depth Split(Training 26000 images)...\n');
end

cam = 2;
maxLen = 1024; %num of samples in one mat file.

setCount = 1;
fileCount = 1;

fprintf('Processing Kitti Eigen training set...\n');
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
                depth(velo_img(l,2),velo_img(l,1)) = velo_img(l,3); %keep the nearest point.
            else    
                fprintf('Find one not replace...\n');
            end
        else
            depth(velo_img(l,2),velo_img(l,1)) = velo_img(l,3);
        end    
    end    
      
    % plot points. For visualization purpose.
%     img = left;
%     fig = figure('Position',[20 100 size(img,2) size(img,1)]); axes('Position',[0 0 1 1]);
%     imshow(img); hold on;
%     cols = jet;
%     for j=1:size(velo_img,1)
%       col_idx = round(64/(velo(j,1)+1));
%       plot(velo_img(j,1),velo_img(j,2),'o','LineWidth',4,'MarkerSize',1,'Color',cols(col_idx,:));
%     end
    train.name = 'Eigen_Train';
    train.left{fileCount}=left;
    train.leftAdd{fileCount} = [folder left_all{i}(1:end-4) '.png'];
    train.right{fileCount}=right;
    train.rightAdd{fileCount} = [folder right_all{i}(1:end-4) '.png'];
    train.depth{fileCount}=depth;
    train.baseline{fileCount} = baseline;
    train.focal{fileCount} = focal_length;
    fileCount = fileCount + 1;
    
    if(fileCount > maxLen)  %specify max_num of pair in one mat. Write 2GB each file.
          train.train = 1;
          train.length = fileCount-1;
          filename = sprintf(['./training/trainKittiEigen' num2str(setCount) '.mat']);
          fprintf('    Writing Set %d...',setCount);
          fprintf('    It contains %d pair of images...\n',fileCount-1);
          save(filename,'train','-v7.3');
          clear train;

          fileCount = 1;
          setCount = setCount + 1;
          clear train;
     end
end   
      train.train = 1;
      train.length = fileCount-1;
      filename = sprintf(['./training/trainKittiEigen' num2str(setCount) '.mat']);
      fprintf('    Writing Set %d...',setCount);
      fprintf('    It contains %d pair of images...\n',fileCount-1);
      save(filename,'train','-v7.3');
      clear train;