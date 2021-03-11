
clear all; close all; clc;
format shortG;
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



%% optimizing parameters


h_unit_vector=4:10;
for i= 1:length(h_unit_vector)
 % create a neural network
net = feedforwardnet([h_unit_vector(i)]);
% train net
net.divideParam.trainRatio = 1; % training set [%]
net.trainParam.max_fail=2;
net.trainParam.mu=0.066;

% train a neural network
[net2,tr] = train(net,input,target_new);
Y=net2(input);

% valTargets = target_new .* tr.valMask{1};

[c,cm] = confusion(target_new,Y);
% over_all_accuracy=1-c;

accuracies(i,:)=[h_unit_vector(i)  cm(1,1)/sum(cm(1,:))  cm(2,2)/sum(cm(2,:)) cm(3,3)/sum(cm(3,:))  1-c   ]; 


end;

save training_accuracies.mat accuracies

% % plot(h_unit_vector,val_accuracy);
% xlabel('Number of Hidden Units');
% ylabel('Overall validation accuracy');
% title('Accuracy vs Hidden Units');








