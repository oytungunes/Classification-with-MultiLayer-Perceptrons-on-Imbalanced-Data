
clear all; close all; clc;
format shortG;

load balanced_training_data
load ann-test.data


data=balanced_training_data;
data_test=ann_test;

%% normalized case
% features and the target
input= ( data(:,1:21) );
target=data(:,22);

% test data
input_test=  data(:,1:21) ;
% input_test(:,15)=zeros(size(input_test,1),1);
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


%% optimizing parameters


 % create a neural network
net = feedforwardnet([3]);
% train net
net.divideParam.trainRatio = 1; % training set [%]
net.trainParam.max_fail=2;
net.trainParam.mu=0.066;

% train a neural network with normalized
[net2,tr] = train(net,input,target_new);
Y=net2(input_test);
[c,cm] = confusion(target_new_test,Y);
accuracies=[cm(1,1)/sum(cm(1,:))  cm(2,2)/sum(cm(2,:)) cm(3,3)/sum(cm(3,:))  1-c   ]; 

save unnormalized_data_accuracies.mat accuracies


