function [ error ] = calError( dep , gt, min_,max_, crop )
       %%Error calculation for Kitti(Eigen Split) Benchmark
       if(size(dep,1)~=size(gt,1) ||size(dep,2)~=size(gt,2))
           error('Size not match...Please check...\n');
       end   
       
       gt_h = size(gt,1);
       gt_w = size(gt,2);
       
       %We follow Garg's(ECCV 16) method to crop the image
       if(strcmp(crop,'garg'))
           crop = [0.40810811 * gt_h 0.99189189 * gt_h 0.03594771*gt_w 0.96405229 * gt_w];
       elseif(strcmp(crop,'eigen'))   
           crop = [0.3324324 * gt_h 0.91351351 * gt_h 0.0359477*gt_w 0.96405229 * gt_w];
       else
           fprintf('No crop in the evaluation...\n');
           crop = [1 gt_h 1 gt_w];
       end
       
       crop = uint16(crop);
       
       mask = zeros([gt_h gt_w]);
       mask(crop(1):crop(2),crop(3):crop(4)) = 1;
       
       %limit the prediction
       dep(dep>max_) = max_;
       dep(dep<min_)=  min_;
       
       mask_ = mask & gt>min_ & gt<max_;

       dep_ = dep(mask_);
       gt_ = gt(mask_);
       
       error = zeros([7 1]);
       
       % Seven Error using in benchmark
        abs_rel = mean(abs(gt_(:) - dep_(:)) ./ gt_(:));  
        sql_rel = mean((gt_(:) - dep_(:)).^2 ./ gt_(:));  
        rmse = sqrt(  mean((gt_(:) - dep_(:)).^2)  );
        rmse_log = sqrt(mean(  (log(gt_(:)) - log(dep_(:))  ).^2));

        thresh = max((gt_ ./ dep_), (dep_ ./ gt_));
        a1 = mean(mean(thresh < 1.25));
        a2 = mean(mean(thresh < 1.25^2));
        a3 = mean(mean(thresh < 1.25^3));

        error(1) = abs_rel;
        error(2) = sql_rel;
        error(3) = rmse;
        error(4) = rmse_log;
        error(5) = a1;
        error(6) = a2;
        error(7) = a3;
end

