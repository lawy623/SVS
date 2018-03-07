function [] = print_Result( error_mean_1,error_mean_2 )
    %print result Matrics
    metricName = {'ARD         ', 'SRD         ', 'RMSE      ', 'RMSE(log)  ', 'delta<1.25 ', 'delta<1.25^2 ', 'delta<1.25^3'};
    fprintf('----------------------------------------Results-------------------------------------------\n');
    fprintf('        ');
    for k=1:length(metricName)
        fprintf(metricName{k});
    end 
    fprintf('\n0-80m  ');
    for k=1:length(error_mean_1)
        fprintf('%.03f',(error_mean_1(k)));
        fprintf('       ');
    end  
    fprintf('\n1-50m  ');
    for k=1:length(error_mean_1)
        fprintf('%.03f',(error_mean_2(k)));
        fprintf('       ');
    end  
    fprintf('\n');
end

