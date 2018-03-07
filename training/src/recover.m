function [ img_recover ] = recover( img )
    img_recover = img(:,:,[3 2 1]);
    mean_vgg = [103.939 116.779 123.68];
    for c = 1:3
        img_recover(:,:,c) = (img_recover(:,:,c)) +mean_vgg(c) ;
    end
    %img_recover = permute(img_recover,[2 1 3]);
end

