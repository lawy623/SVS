function [ image, shiftInput, label ] = transformation_viewSyn( train, ind ,dim, channel, stride)
%Transformation Parameters (scale/resize, crop, flip)
%Scale
scale_control = 1;        %determine how random is the resize. Default as 1.(Always do). 0: not do.
min_scale = 1;
max_scale = 1.1;
scale_rate = min_scale + (max_scale-min_scale)*rand(1);         %random scale between min~max. 

%Flip
flip_control = 0.5;       %determine how random is the flipping. Default as 0.5.(Do 50% times).

% Whether to do the transformation or not
if rand(1) > scale_control      %Do the resize operation at frequency (100*scale_control)%
   scale_rate=1;
end
if rand(1) < flip_control       %Do the flip operation at frequency (100*flip_control)%
    do_flip=1;
else
    do_flip=0;
end

color_ = 0.8 + 0.4*rand(1);  %color intensity augmentation.

image = train.left{ind};
label = train.right{ind};

    %Resize the image
    [image, label] = imgScale(image,label,scale_rate);
    
    image = (single(image));
    label = (single(label));
                                                                                                                                                           
    %flip the image
    if (do_flip)
        [image, label] = imgFlip(image, label);
    end
    
    %crop to default size
    [image, label] = imgCrop(image, label, dim);
    
    %Image preprocessing(Mean, width<->height,RBG->BRG,doLog to depth if needed)
    [image, shiftInput, label] = preprocess(image,label,color_,channel,stride);

end

function plotImage(image,label)
    figure(1);hold on;
    subplot(2,1,1);imshow(image);
    subplot(2,1,2);imshow(label);
end

function [image_scale, label_scale] = imgScale(image, label, scale_rate)
        %Need to first resize to 1/2 * 1/2 of original image.
        image_scale = imresize(image , scale_rate/1.9 ,'bilinear');
        label_scale = imresize(label , scale_rate/1.9 ,'bilinear');
end

function [image_crop, label_crop] = imgCrop(image_rotate, label_rotate, dim)

    center_x = round( size(image_rotate,2) / 2 );
    center_y = round( size(image_rotate,1) / 2 );

    left = center_x - dim(2)/2;
    up   = center_y - dim(1)/2;

    try
        image_crop = image_rotate(up:up+dim(1)-1, left:left+dim(2)-1, :);
        label_crop = label_rotate(up:up+dim(1)-1, left:left+dim(2)-1, :);
    catch
        error('      something wrong happens in cropping....\n');
    end
end

function [image_flip, label_flip] = imgFlip(image_crop,label_crop)
    image_flip = flip(image_crop,2);
    label_flip = flip(label_crop,2);
    
    %After flipping, need to interchange two images.
    temp = image_flip;
    image_flip = label_flip;
    label_flip = temp;
end

function [img_out,shiftInput, label_out] = preprocess(img,label,color_,channel,stride)
    img = img .* color_;
    label = label .* color_;
    
    img_out = (single(zeros(size(img))));
    label_out = (single(zeros(size(label))));
    
    mean_vgg = [103.939 116.779 123.68];  %inital from VGG. Do the mean.
    for c = 1:3
        img_out(:,:,c) = (img(:,:,c)) -mean_vgg(c) ;
        label_out(:,:,c) = (label(:,:,c)) -mean_vgg(c);
    end
    
    %create shift inputs.
    shiftInput = (single(zeros([size(img,1) size(img,2) 3 channel])));
    for i=1:channel
        dis = stride * (i-1);
        shiftInput(:,:,:,i)= shiftDis(img_out,dis);      
    end    

    img_out = permute(img_out, [2 1 3]);
    img_out = img_out(:,:,[3 2 1]); 
    
    shiftInput = permute(shiftInput, [2 1 3 4]);
    shiftInput = shiftInput(:,:,[3 2 1],:); 
    
    label_out = permute(label_out, [2 1 3]);
    label_out = label_out(:,:,[3 2 1]); 
end

function shiftInput_ = shiftDis(img,dis) %shift the whole image to the left by #dis pixels
    shiftInput_ = (single(zeros(size(img))));
    shiftInput_(:,1:size(img,2)-dis,:) = img(:,1+dis:size(img,2),:);
end    
    
