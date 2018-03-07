function [ image,left_large, shiftInput ] = transformation_svs( image ,dim,dim_large,channel, stride)
%Transformation Parameters (scale/resize, rotate, crop, flip, addLabels)

    %Resize the image
    [image,left_large] = imgScale(image,dim,dim_large);  
    
    image = (single(image)); 
    left_large = (single(left_large)); 
    
    %Image preprocessing(Mean, width<->height,RBG->BRG,doLog to depth if needed)
    [image, shiftInput, left_large] = preprocess(image,left_large,channel,stride);

end

function [image_scale, left_large] = imgScale(image, dim,dim_large)
        image_scale = imresize(image , [dim(1) dim(2)] ,'bilinear');
        left_large = imresize(image , [dim_large(1) dim_large(2)] ,'bilinear');
end

function [img_out,shiftInput,left_large_out] = preprocess(img,left_large,channel,stride)
    
    img_out = (single(zeros(size(img)))); 
    
    mean_vgg = [103.939 116.779 123.68];
    for c = 1:3
        img_out(:,:,c) = (img(:,:,c)) -mean_vgg(c) ;
    end
    
    %create shift inputs.
    shiftInput = (single(zeros([size(img_out,1) size(img_out,2) size(img_out,3) channel]))); 
    for i=1:channel
        dis = stride * (i-1);
        shiftInput(:,:,:,i) = shiftDis(img_out,dis);
    end    

    img_out = permute(img_out, [2 1 3]);
    img_out = img_out(:,:,[3 2 1]); 
    
    left_large_out = permute(left_large, [2 1 3]);
    left_large_out = left_large_out(:,:,[3 2 1]); 
    
    shiftInput = permute(shiftInput, [2 1 3 4]);
    shiftInput = shiftInput(:,:,[3 2 1],:); 
    
end

function shiftInput = shiftDis(img,dis) %shift the whole image to the left by #dis pixels
    shiftInput = (single(zeros(size(img)))); 
    shiftInput(:,1:size(img,2)-dis,:) = img(:,1+dis:size(img,2),:);
end    
    
