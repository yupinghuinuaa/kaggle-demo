function mc=findMainContour(B)
mc={};
mc_area=0;
for i=1:length(B)
    cur_c=B{i};
    c_area=polyarea(cur_c(:,2),cur_c(:,1));
    if c_area>=mc_area
        mc=cur_c;
        mc_area=c_area;
    end
end
end