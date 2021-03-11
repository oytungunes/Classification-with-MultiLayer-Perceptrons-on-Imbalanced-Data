
clear all; close all; clc;

load balanced_training_data
load ann-test.data

data=balanced_training_data;
data_test=ann_test;
%check the normalized data
% mean(normalized_data)
% var(normalized_data)

% features and the target
input= normalize_data( data(:,1:21) );
target=data(:,22);

% test data
input_test= normalize_data( data(:,1:21) );
target_test=data(:,22);




%% convert to an appropriate format for neural networks
input=input';
target_new=zeros(3,size(target,1));
target_new(1,:)=target==1;
target_new(2,:)=target==2;
target_new(3,:)=target==3;

%% test convert to an appropriate format for neural networks
input_test=input_test';
target_new_test=zeros(3,size(target_test,1));
target_new_test(1,:)=target_test==1;
target_new_test(2,:)=target_test==2;
target_new_test(3,:)=target_test==3;





% create a neural network
net = feedforwardnet([4]);
% train net
net.divideParam.trainRatio = 1; % training set [%]
% net.divideParam.testRatio = 0.15; % test set [%]


%% optimizing parameters


% net.trainParam.mu=0.1;

% train a neural network
[net2,tr] = train(net,input,target_new);
Y=net2(input_test);

% valTargets = target_new .* tr.valMask{1};
[c,cm,ind,per]  = confusion(target_new_test,Y);


% over_all_accuracy=1-c;
% val_accuracy= 1-perform(net2,valTargets,Y);










