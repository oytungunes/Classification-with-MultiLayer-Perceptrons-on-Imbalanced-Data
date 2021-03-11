function [ label, numberofclasses,ispure ] = find_classes( data )

changes=diff(data(:,22))~=0;
ind_of_chg=find(changes==1);  

if isempty(ind_of_chg)
    ispure=1;
else
    ispure=0;
end;

numberofclasses(1)=sum(data(:,22)==1);
numberofclasses(2)=sum(data(:,22)==2);
numberofclasses(3)=sum(data(:,22)==3);
    
[min,label]=max(numberofclasses);


end


