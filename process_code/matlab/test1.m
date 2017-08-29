addpath('/home/liuwen/PycharmProjects/battleNet0/process_code/matlab/');  % current folder
W=dir('/home/liuwen/ssd/kaggle/original/test/*.jpeg');
parpool(12);
parfor i=1:length(W)
    rawImg=imread(['/home/liuwen/ssd/kaggle/original/test/' W(i).name]);
    cropimg=imAutoResize(rawImg);
    %　resizeimg=imresize(new,[900 900]);
    %　[IV,III,I,B]=qiyizhi_normg(cropimg,BP);
    %　IV=imresize(cropimg,[800 800]);
    imwrite(cropimg,strcat('/home/liuwen/ssd/kaggle/original/test_crop/',W(i).name))
end;
