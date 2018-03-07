%add paths
clear all;
addpath('./src');

%train data address
trainAdd = '../data/training/';
trainSets = dir([trainAdd '*.mat']);
      
param = trainConfig();
        
model = param.model(3);

fprintf('Training model %s ...\n',model.description);

solver = caffe.Solver(model.solverFile);
saveDir = model.saveDir;

preModel1 = model.preModel1;
preModel2 = model.preModel2;
solver.net.copy_from(preModel1); 

net_stage2 = model.trainFile2_tmp;
net2 = caffe.Net(net_stage2,preModel2, 'train');

%This is due to some naming problem. Some of the layers in pretrain Stereo
%Matching network share the same names with some layers in ViewSyn network.
%In the end2end version, we make them distinct.
for l = 3 :115
    L_name = net2.layer_names(l);
    par = net2.layers(L_name{1}).params;
    len = length(par);
    if(len ~= 2 && len~= 0)
        error('Layer name %s contains different num of blobs, please check...\n',L_name{1});
    else    
        if(size(par,1) == 0 && size(par,2) == 0)
        else
            R_name = solver.net.layer_names(l+207);
            if(~strcmp(L_name{1},R_name{1}) && ~strcmp('conv3_1_',R_name{1}) && ~strcmp('conv4_1_',R_name{1}) && ~strcmp('conv5_1_',R_name{1}) )
                error('Layer name not match for ind %d...\n',l);
            end    
            fprintf('Copying weights of Layer name %s ...\n',L_name{1});
            for o = 1:2
               solver.net.layer_vec(l+207).params(o).set_data(par(o).get_data() );
            end    
        end
    end    
end

%model & transformation params
batch_size = model.batchSize;
dim = [model.height model.width];
dim_large = [model.H model.W];
stride = model.stride;
channel = model.channel;
       
epoch = 200;
for ep = 1:epoch
    order_set = randperm(length(trainSets));
    for i = 1:length(trainSets)      
        clear train;
        fprintf('Loading trainSet %s for Training....\n',trainSets(i).name);
        trainData  = load([trainAdd '/' trainSets(i).name]);
        if(strcmp(trainSets(i).name,'trainKitti15.mat'))
            train = trainData.gt;
            seqLength = train.size(1);
            eigen = 0;
        else
            train = trainData.train;
            seqLength = train.length;
            eigen = 1;
        end    
        clear trainData;
        
        %check whether using training or not
        if train.train~=1
            fprintf('Error in train Sequence. Not Belong to Train Set.\n');
        end
        %random permutation to get the data, increase randomness
        order = randperm(seqLength);
        
        for batch = 1:(seqLength/batch_size) 

            data_ = zeros([dim(2) dim(1) 3 batch_size]);
            label_ = zeros([dim(2) dim(1) 3 batch_size]);
            shift = zeros([dim(2) dim(1) 3 batch_size channel]);
            left_large_ = zeros([dim_large(2) dim_large(1) 3 batch_size]);
            dis_ = zeros([dim_large(2) dim_large(1) 1 batch_size]);
            for n = 1:batch_size
                trainInd = order( (batch-1)*batch_size + n);

                %seperately conduct transformation. Keep transformation in a
                %sequence consistent, but random in diff sequence.   
                [input,shiftInput,label,left_large,dis] = transformation_svs_end2end(train,trainInd,dim,dim_large,channel,stride,eigen);  

                data_(:,:,:,n) = (input);
                label_(:,:,:,n) = (label);
                shift(:,:,:,n,:) =(shiftInput);
                left_large_(:,:,:,n) =(left_large);
                dis_(:,:,:,n) =(dis);

            end   
            solver.net.blobs('data').set_data(single(data_));
            solver.net.blobs('label').set_data(single(label_));
            for k =1:channel
               ss = strcat('dis',num2str(k-1));
               solver.net.blobs(ss).set_data(single(shift(:,:,:,:,k)));
            end 

            solver.net.blobs('left_ori').set_data(single(left_large_));
            solver.net.blobs('disp_gt_aug').set_data(single(dis_));

            clear input label shiftInput left_large dis data_ label_ shift left_large_ dis_;

            solver.step(1);

            iter = solver.iter(); 

            %save the model
            if(rem(iter,10000)==0)
                fprintf('Saving model for iter %d...\n',iter);
                solver.net.save([saveDir '/caffemodel/svs_end2end_iter_' num2str(iter) '.caffemodel']);
            end  
            
            %Save visual result
            if(rem(iter,5000)==0 || iter == 1 )
                data = solver.net.blobs('data').get_data();
                label = solver.net.blobs('label').get_data();
                pred = solver.net.blobs('pred_right').get_data();

                idx = 1;

                h=figure('Visible', 'off');hold on;subplot(1,3,1);imshow(uint8(recover(data(:,:,:,idx))),[]);subplot(1,3,2);imshow(uint8(recover(label(:,:,:,idx))),[]);subplot(1,3,3);imshow(uint8(recover(pred(:,:,:,idx))),[]);

                saveas(h,strcat(saveDir,'/fig/figure_',num2str(iter),'.png'));
                clf; clear data label pred;
                
                dd = -solver.net.blobs('predict_flow0').get_data();
                h=figure('Visible', 'off');hold on;imshow(dd(:,:,:,idx),[]);

                saveas(h,strcat(saveDir,'/fig/figure_',num2str(iter),'_disp.png'));
            end            

        end   
    end    
end        
        

       
