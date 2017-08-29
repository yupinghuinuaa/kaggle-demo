%function resizeOperate1()
imgfiles=findAllFilesInDir('/home/imed269506/norm/precessimg/train/train0/','jpeg');
len_imgfs=length(imgfiles);
goal_size=[224,224];%(r,c)
for i=1:len_imgfs
    img=imread(imgfiles{i});
    s_img=size(img);%(r,c)
    ratioW=s_img(2)/goal_size(2);
    ratioH=s_img(1)/goal_size(1);
    resiImg=[];
    cropImg=[]';
    if ratioW>ratioH                   %����������
        resiImg=imresize(img,[s_img(1)/ratioH,s_img(2)/ratioH]);
        s_resiImg=size(resiImg);
        xoffset=(s_resiImg(2)-goal_size(2))/2;
        cropImg=imcrop(resiImg,[xoffset,0,goal_size(2),goal_size(1)]);
    else
        resiImg=imresize(img,[s_img(1)/ratioW,s_img(2)/ratioW]);
        s_resiImg=size(resiImg);
        yoffset=(s_resiImg(1)-goal_size(1))/2;
        cropImg=imcrop(resiImg,[0,yoffset,goal_size(2),goal_size(1)]);
    end
    cropImg=imresize(cropImg,goal_size);
  %  mkdir('D:\caffe-windows\project\650\seps4\AGLAIA_GT_649\1\11\');%�½�һ��Ŀ���ļ���
    %newpath=createPathByCopiedPath(imgfiles{i},'/home/imed269506/norm/precessimg/train/train_800/train0/');
    imwrite(cropImg,strcat('/home/imed269506/norm/precessimg/train/train_800/train0/',imgfiles{i}.name));
end;