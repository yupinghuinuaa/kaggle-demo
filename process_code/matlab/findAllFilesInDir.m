function imgfiles=findAllFilesInDir(imgdir,varargin)
imgfiles={};
imgfDir=dir([imgdir,'\*']);



indx=1;
for i=1:length(imgfDir)
    if length(imgfDir(i).name)<4
        continue;
    end
    arginLen=length(varargin);
    if arginLen==0
        sel=true;
    else
        sel=false;
    for j=1:length(varargin)
        if(length(findstr(imgfDir(i).name,char(varargin(j))))>0)
            sel=true;
        end
    end
    end
    
    if ~sel
        continue;
    end
    
    imgfile=[imgdir,imgfDir(i).name];
    %imgfiles{imfidx}=imgfile;
    imgfiles{indx}=imgfile;
    indx=indx+1;
end
end