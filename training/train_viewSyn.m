%add paths
clear all;
addpath('./src');

%train data address
trainAdd = '../data/training/';
trainSets = dir([trainAdd '*.mat']);
      
param = trainConfig();
        
model = param.model(1);

%Choose to finetune or continue
solver = caffe.Solver(model.solverFile);
preModel = model.preModel;
solver.net.copy_from(preModel); 

%model & transformation params
batch_size = model.batchSize;
dim = [model.height model.width];
stride = model.stride;
channel = model.channel;
saveDir = model.saveDir;

fprintf('This is for model %s ...\n',model.description);
   
epoch = 200;
for ep = 1:epoch
    order_set = randperm(length(trainSets));
    for i = 1:length(trainSets)
        clear train;
        fprintf('Loading trainSet %s for Training....\n',trainSets(order_set(i)).name);
        trainData  = load([trainAdd '/' trainSets(order_set(i)).name]);
        if(strcmp(trainSets(order_set(i)).name,'trainKitti15.mat'))
            train = trainData.gt;
            seqLength = train.size(1);
        else
            train = trainData.train;
            seqLength = train.length;
        end    
        clear trainData;
        
        %check whether using training or not
        if train.train~=1
            fprintf('Error in train Sequence. Not Belong to Train Set.\n');
        end
 
        %random permutation to get the data, increase randomness
        order = randperm(seqLength);
        for batch = 1:floor(seqLength/batch_size) 

            data_ = zeros([dim(2) dim(1) 3 batch_size]);
            label_ = zeros([dim(2) dim(1) 3 batch_size]);
            shift = zeros([dim(2) dim(1) 3 batch_size channel]);
            for n = 1:batch_size
                trainInd = order( (batch-1)*batch_size + n);
 
                [input,shiftInput,label] = transformation_viewSyn(train,trainInd,dim,channel,stride);   
                data_(:,:,:,n) = (input);
                label_(:,:,:,n) = (label);
                shift(:,:,:,n,:) =(shiftInput);
            end

            solver.net.blobs('data').set_data(single(data_));
            solver.net.blobs('label').set_data(single(label_));
            for k =1:channel
               ss = strcat('dis',num2str(k-1));
               solver.net.blobs(ss).set_data(single(shift(:,:,:,:,k)));
            end    
            clear input label shiftInput data_ label_ shift input;
            solver.step(1);
            iter = solver.iter();   

            %save the model
            if(rem(iter,10000)==0)
                fprintf('Saving model for iter %d...\n',iter);
                solver.net.save([saveDir '/caffemodel/viewSyn_iter_' num2str(iter) '.caffemodel']);
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
            end            
        end
    end     
end        
        

       
