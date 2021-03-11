

close all; clear all; clc;

% You	 will	 conduct	 your	 experiments	 on	 the	 “Thyroid	 data	 set”,	 which	 is	 taken	 from	 the	 UCI	
% repository	and	available	on	the	course	web	page.	The	details	of	this	data	set	are	given	as	follows:
% ? It	contains	separate	training	(“ann-train.data”)	and	test	(“ann-test.data”)	sets.
% ? The	training	set	contains	3772	instances	and	the	test	set	contains	3428	instances.
% ? There	are	a	total	of	3	classes.	Note	that	this	dataset	has	unbalanced	class	distributions.	
% Thus,	you	may	want	to	consider	this	in	training	your	decision	tree	classifier.	
% ? In	the	data	files,	each	line	corresponds	to	an	instance	that	has	21	features	(15	binary	and	
% 6	continuous	features)	and	1	class	label.	

load ann-train.data


data=ann_train;
[sorteddata,I]= sortrows(data(:,22),1); % sorta gerek yok

%make it balanced
[ label, numberofclasses,~ ] = find_classes( data);
sum(data(:,22)==3)
c1_data= data(I(1:numberofclasses(1)),:);
c2_data= data(I(numberofclasses(1)+1:numberofclasses(1)+1+numberofclasses(2)),:); 
% make c1 and c2 3000 by oversampling
balanced_training_data=[data;repmat(c1_data,30,1);repmat(c2_data,15,1)];
sum(balanced_training_data(:,22)==3)

%shuffle the data
balanced_training_data=balanced_training_data(randperm(size(balanced_training_data,1)),:);



% chekck the number of classes
[ label, numberofclasses2,~ ] = find_classes( balanced_training_data);

save balanced_training_data.mat balanced_training_data









