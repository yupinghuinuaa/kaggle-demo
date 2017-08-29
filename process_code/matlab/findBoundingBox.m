function [bbw bbh bbox pbbox]=findBoundingBox(contour)
%contour=MB;
bbx=0;
bby=0;
bbw=0;
bbh=0;

lp=[];
tp=[];
rp=[];
bp=[];

lt=[];
rt=[];
lb=[];
rb=[];


bbx=min(contour(:,2));
bby=min(contour(:,1));
bbw=max(contour(:,2))-bbx;
bbh=max(contour(:,1))-bby;

bbox=[bbx,bby,bbw,bbh];

lt=[bbx,bby];
rt=[bbx+bbw,bby];
lb=[bbx,bby+bbh];
rb=[bbx+bbw,bby+bbh];

pbbox=lt;
pbbox=[pbbox;rt];
pbbox=[pbbox;rb];
pbbox=[pbbox;lb];

end