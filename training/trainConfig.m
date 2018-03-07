function param = trainConfig()

% Path of caffe. You can change to your own caffe just for testing
caffepath = '../caffe/matlab';

fprintf('You set your caffe in caffePath.cfg at: %s\n', caffepath);
addpath(caffepath);
caffe.reset_all();

fprintf('Setting to GPU mode\n');
caffe.set_mode_gpu();
caffe.set_device(0);

%% Parameter setting
param.model(1).preModel = './prototxt/viewSynthesis/preModel/preViewSyn.caffemodel';%VGG_ILSVRC_16_layers.caffemodel';
param.model(1).trainFile = './prototxt/viewSynthesis/ViewSyn_train.prototxt';
param.model(1).solverFile = './prototxt/viewSynthesis/ViewSyn_solver.prototxt';
param.model(1).description = 'View Synthesis model trained on Kitti. Developed based on deep3D model';
param.model(1).description_short = 'ViewSyn_network';
param.model(1).saveDir = './prototxt/viewSynthesis';
param.model(1).width = 640; 
param.model(1).height = 192;
param.model(1).batchSize = 4;
param.model(1).channel = 65;
param.model(1).stride = 1;   %this control the disparity scale step. Max disparity will be 32*stride. So only 1/2 will be needed.

param.model(2).preModel1 = './prototxt/viewSynthesis/caffemodel/viewSyn.caffemodel';  %first stage network
param.model(2).preModel2 = './prototxt/stereo/caffemodel/disp_flyKitti.caffemodel';  %second stage network
param.model(2).trainFile1 = './prototxt/svs/svs_train.prototxt';
param.model(2).trainFile2_tmp = './prototxt/svs/stereo_train_tmp.prototxt';
param.model(2).solverFile = './prototxt/svs/svs_solver.prototxt';
param.model(2).description = 'Single View Stereo Matching model end2end training on Kitti, two stages are pretrained';
param.model(2).description_short = 'svs_end2end';
param.model(2).saveDir = './prototxt/svs';
param.model(2).width = 640;
param.model(2).height = 192;
param.model(2).W = 1280;
param.model(2).H = 384;
param.model(2).batchSize = 2;
param.model(2).channel = 65;
param.model(2).stride = 1; %this control the disparity scale variance. Max disparity will be 64*stride. So only 1/2 will be needed.


