function param = testConfig()

param.use_gpu = 1;
% GPU device number
GPUdeviceNumber = 0;

% Path of caffe. You can change to your own caffe just for testing
caffepath = '../caffe/matlab';

fprintf('You set your caffe in caffePath.cfg at: %s\n', caffepath);
addpath(caffepath);
caffe.reset_all();

if(param.use_gpu)
    fprintf('Setting to GPU mode, using device ID %d\n', GPUdeviceNumber);
    caffe.set_mode_gpu();
    caffe.set_device(GPUdeviceNumber);
else
    fprintf('Setting to CPU mode.\n');
    caffe.set_mode_cpu();
end

%% Parameter setting
param.model(1).preModel1 = '../training/prototxt/viewSynthesis/caffemodel/viewSyn.caffemodel';
param.model(1).preModel2 = '../training/prototxt/stereo/caffemodel/disp_flyKitti.caffemodel'; %disp_flyThings.caffemodel';
param.model(1).deployFile1 = '../training/prototxt/viewSynthesis/ViewSyn_deploy.prototxt';
param.model(1).deployFile2 = '../training/prototxt/stereo/stereo_deploy.prototxt';
param.model(1).description = 'Monocular depth estimation using SVS before end2end finetune';
param.model(1).description_short = 'svs_beforeFinetune_kittiEigen';
param.model(1).width = 640;
param.model(1).height = 192;
param.model(1).W = 1280;
param.model(1).H = 384;
param.model(1).batchSize = 1;
param.model(1).channel = 65;
param.model(1).stride = 1;   %this control the disparity scale variance. Max disparity will be 64*stride. So only 1/2 will be needed.

param.model(2).preModel = '../training/prototxt/svs/caffemodel/svs_end2end.caffemodel';
param.model(2).deployFile = '../training/prototxt/svs/svs_deploy.prototxt';
param.model(2).description = 'Monocular depth estimation using SVS after end2end finetune';
param.model(2).description_short = 'svs_afterFinetune_kittiEigen';
param.model(2).width = 640;
param.model(2).height = 192;
param.model(2).W = 1280;
param.model(2).H = 384;
param.model(2).batchSize = 1;
param.model(2).channel = 65;
param.model(2).stride = 1;   %this control the disparity scale variance. Max disparity will be 64*stride. So only 1/2 will be needed.

