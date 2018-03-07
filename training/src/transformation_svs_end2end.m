function [ image, shiftInput, label, left_large,disp ] = transformation_svs_end2end( train, ind ,dim,dim_large,channel, stride,eigen)
%Transformation Parameters (scale/resize, rotate, crop, flip, addLabels)
%Scale
scale_control = 1;        %determine how random is the resize. Default as 1.(Always do). 0: not do.
min_scale = 1.05;
max_scale = 1.3;
scale_rate = min_scale + (max_scale-min_scale)*rand(1);         %random scale between min~max. All frames should be the same.

%Flip
flip_control = 0;       %determine how random is the rotation. Default as 0.5.(Do 50% times).

% Whether to do the transformation or not
% Put it here to make the transformation for whole sequence consistent
if rand(1) > scale_control      %Do the resize operation at frequency (100*scale_control)%
   scale_rate=1;
end
if rand(1) < flip_control       %Do the flip operation at frequency (100*flip_control)%
    do_flip=1;
else
    do_flip=0;
end

color_ = 0.8 + 0.4*rand(1);

image = train.left{ind};
label = train.right{ind};
if(eigen)
    disp =  train.focal{ind}* train.baseline{ind} ./ double(train.depth{ind});
    disp(disp ==Inf) = 0;
else    
    disp = double(train.disp{ind})/256; 
end    

    %Resize the image
    [image, label, disp] = imgScale(image,label,disp,scale_rate);   
    
    image = single(image);
    label = single(label);
    disp = single(disp);
                                                                        
    %flip the image, actually unused here. If we need to use, we need the
    %right disparity as well. Need to revised training dataset.
    if (do_flip)
        [image, label, disp] = imgFlip(image, label, disp);
    end
    
    %crop to default size
    [image, label, disp] = imgCrop(image, label, disp, dim_large);
    
    %Image preprocessing(Mean, width<->height,RBG->BRG,doLog to depth if needed)
    [image, shiftInput,left_large,label, disp] = preprocess(image,label,disp, dim, color_,channel,stride);

end

function plotImage(image,label,disp)
    figure(1);hold on;
    subplot(3,1,1);imshow(uint8((image)));
    subplot(3,1,2);imshow(uint8((label)));
    subplot(3,1,3);imshow(uint8((disp)),[]);
end

function [image_scale, label_scale, disp_scale] = imgScale(image, label, disp, scale_rate)
        %need to rescale to 1/2 of origin first, them 1~1.5 larger
        image_scale = imresize(image , scale_rate ,'bilinear');
        label_scale = imresize(label , scale_rate ,'bilinear');%./scale_rate; this may need to be revised
        disp_scale = imresize(disp , scale_rate ,'nearest');
        disp_scale(disp_scale<0) = 0;
        disp_scale = disp_scale * scale_rate; %need to scale up the disp as well if we resize the image and disp
end

function [image_crop,label_crop, disp_crop] = imgCrop(image, label, disp, dim)

    center_x = round( size(image,2) / 2 );
    center_y = round( size(image,1) / 2 );

    left = center_x - dim(2)/2;
    up   = center_y - dim(1)/2;

    try
        image_crop = image(up:up+dim(1)-1, left:left+dim(2)-1, :);
        label_crop = label(up:up+dim(1)-1, left:left+dim(2)-1, :);
        disp_crop = disp(up:up+dim(1)-1, left:left+dim(2)-1, :);
    catch
        error('      something wrong happens in cropping....\n');
    end
end

function [image_flip, label_flip] = imgFlip(image_crop,label_crop) % need to be revised if we want to use it.

    image_flip = flip(image_crop,2);
    label_flip = flip(label_crop,2);
    
    temp = image_flip;
    image_flip = label_flip;
    label_flip = temp;
end

function [img_out,shiftInput,left_large_out, label_out, disp_out] = preprocess(img,label,disp, dim,color_,channel,stride)
    img = img .* color_;
    label = label .* color_;
    
    %Imresize here to make the input to viewSyn network small(as dim)
    img_ = imresize((img) , [dim(1) dim(2)] ,'bilinear');
    label_ = imresize((label) , [dim(1) dim(2)] ,'bilinear');
     
    img_out = zeros(size(img_),'single');
    label_out = zeros(size(label_),'single');
    
    mean_vgg = single([103.939 116.779 123.68]);
    for c = 1:3
        img_out(:,:,c) = (img_(:,:,c)) -mean_vgg(c) ;
        label_out(:,:,c) = (label_(:,:,c)) -mean_vgg(c);
    end
    
    %create shift inputs.
    shiftInput = zeros([size(img_out,1) size(img_out,2) size(img_out,3) channel],'single');
    for i=1:channel
        dis = stride * (i-1);
        shiftInput(:,:,:,i) = shiftDis(img_out,dis);
    end    

    img_out = permute(img_out, [2 1 3]);
    img_out = img_out(:,:,[3 2 1]); 
    
    disp_out = permute(disp, [2 1 3]) * -1;  %follow the general setting
    left_large_out = permute(img, [2 1 3]);
    left_large_out = left_large_out(:,:,[3 2 1]); 
    
    shiftInput = permute(shiftInput, [2 1 3 4]);
    shiftInput = shiftInput(:,:,[3 2 1],:); 
    
    label_out = permute(label_out, [2 1 3]);
    label_out = label_out(:,:,[3 2 1]); 
end

function shiftInput_ = shiftDis(img,dis) %shift the whole image to the left by #dis pixels
    shiftInput_ = zeros(size(img),'single');
    shiftInput_(:,1:size(img,2)-dis,:) = img(:,1+dis:size(img,2),:);
end    
    
