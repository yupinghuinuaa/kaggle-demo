function newpath=createPathByCopiedPath(oldpath,newdir,newtail,newname)
nameInds1=findstr(oldpath,'\');
nameInds2=findstr(oldpath,'/');
nameInd=max([nameInds1,nameInds2]);
olddir=oldpath(1:nameInd);
oldname=oldpath(nameInd+1:length(oldpath));
tailInd=max(findstr(oldname,'.'));
tail=[];
hasName=0;
if tailInd>0
    tail=oldname(tailInd:length(oldname));
    oldname=oldname(1:tailInd-1);
    hasName=1;
end

switch nargin
    case 1%��ȡdir
        newpath=olddir;
%         if newpath(length(newpath))~='/'||newpath(length(newpath))~='\'
%             newpath=[newpath,'\'];
%         end
    case 2%ͬ��ͬ�����ļ��洢����Ŀ¼
        newpath=newdir;
%         if newpath(length(newpath))~='/'||newpath(length(newpath))~='\'
%             newpath=[newpath,'\'];
%         end
        newpath=[newpath,oldname,tail];
    case 3%ͬ����ͬ�����ļ��洢����Ŀ¼
        newpath=newdir;
%         if newpath(length(newpath))~='/'||newpath(length(newpath))~='\'
%             newpath=[newpath,'\'];
%         end
%         if newtail(1)~='.'
%             newtail=['.',newtail];
%         end
        newpath=[newpath,oldname,newtail];
    case 4%��ͬ����ͬ�����ļ��洢����Ŀ¼
        newpath=newdir;
%         if newpath(length(newpath))~='/'||newpath(length(newpath))~='\'
%             newpath=[newpath,'\'];
%         end
%         if newtail(1)~='.'
%             newtail=['.',newtail];
%         end
        newpath=[newpath,newname,newtail];
end


end