
clear all; close all; clc;

load balanced_training_data


data=balanced_training_data;
%check the normalized data
% mean(normalized_data)
% var(normalized_data)

% features and the target
input= normalize_data( data(:,1:21) );
% input=data(:,1:21);


target=data(:,22);


%% convert to an appropriate format for neural networks
input=input';
target_new=zeros(3,size(target,1));
target_new(1,:)=target==1;
target_new(2,:)=target==2;
target_new(3,:)=target==3;



% create a neural network
net = feedforwardnet([4]);
% train net
net.divideParam.trainRatio = 0.80; % training set [%]
net.divideParam.valRatio = 0.20; % validation set [%]
% net.divideParam.testRatio = 0.15; % test set [%]


%% optimizing parameters


mu_vector=linspace(0.01,0.1,20);
for i= 1:length(mu_vector)

net.trainParam.mu=mu_vector(i);
% net.trainParam.mu=0.1;

% train a neural network
[net2,tr] = train(net,input,target_new);
Y=net2(input);

valTargets = target_new .* tr.valMask{1};
[c,cm] = confusion(target_new,valTargets);
% over_all_accuracy=1-c;
val_accuracy(i) = 1-perform(net2,valTargets,Y);

end;

plot(mu_vector,val_accuracy);
xlabel('Learning Rate');
ylabel('Over_all_training_accuracy');
title('Accuracy vs Learning Rate');





%printing
% fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
% fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);

% nntraintool








