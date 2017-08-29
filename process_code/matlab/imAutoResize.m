function  new=imAutoResize(rawImg)
%rawImg=imread('/home/imed269506/norm/precessimg/train/train0/679_right.jpeg');
 grayImg=rawImg;
 s_rawImg=size(rawImg);
 if length(s_rawImg)==3&&s_rawImg(3)==3
     grayImg=rgb2gray(rawImg);
 end
 %BW=imbinarize(grayImg,'adaptive');
 BW=im2bw(grayImg,0.04);
% figure,imshow(BW);
 [B,L]=bwboundaries(BW);
MB=findMainContour(B);
[bbw bbh bbox pbbox]=findBoundingBox(MB);
bbw=bbw+1;
bbh=bbh+1;
cropimg=rawImg(pbbox(1,2):pbbox(3,2),pbbox(1,1):pbbox(3,1),:);%%crop imge
%
%create a square matrix of crop image 
if bbw==bbh
    %imwrite(cropimg,)
    new=cropimg;
elseif bbw>bbh
    cha=floor((bbw-bbh)/2);
    h=2*cha+bbh;
    new=uint8(zeros(h,bbw,3));
    new(cha+1:h-cha,:,:)=cropimg;
   % imwrite(new,);

elseif bbw<bbh
    cha=floor((bbh-bbw)/2);
    w=2*cha+bbw;
    new=uint8(zeros(bbh,w,3));
    new(:,cha+1:w-cha,:)=cropimg;
  %  imwrite(new,);
end
% %%
%   %find crop image mask
%   cropgrayimg=rgb2gray(cropimg);
%   BP=im2bw(cropgrayimg,0.04);
%   BP=double(BP);
% %imwrite(new,)
% %figure,imshow(new);
% 
% %% test contours output %%
% %figure, imshow(label2rgb(L, @jet, [.5 .5 .5]))
%  %hold on
% % for k = 1:length(B)
% %    boundary = B{k};
% %    plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2)
% % end
% 
% %% test bounding box %%
% % figure,imshow(rawImg)
% % hold on
% %     plot(pbbox(:,1), pbbox(:,2), 'w', 'LineWidth', 2)
% %  hold off

%%

 
end