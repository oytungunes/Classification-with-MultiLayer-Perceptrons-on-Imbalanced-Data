
clear all; close all; clc;
format shortG;

load balanced_training_data
load ann-test.data


data=balanced_training_data;
data_test=ann_test;


% features and the target
input= normalize_data( data(:,1:21) );
target=data(:,22);


% test data
input_test= normalize_data(data_test(:,1:21));
target_test=data_test(:,22);


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


h_unit_vector=4:6;
for i= 1:length(h_unit_vector)
 % create a neural network
net = feedforwardnet([h_unit_vector(i)]);
% train net
net.divideParam.trainRatio = 0.80; % training set [%]
net.divideParam.valRatio = 0.20; % training set [%]
% net.trainParam.max_fail=7;
% net.trainParam.mu=0.066;

% train a neural network
[net2,tr] = train(net,input_test,target_new_test);
Y=net2(input_test);


[c,cm] = confusion(target_new_test,Y);

accuracies(i,:)=[h_unit_vector(i)  cm(1,1)/sum(cm(1,:))  cm(2,2)/sum(cm(2,:)) cm(3,3)/sum(cm(3,:))  1-c   ]; 


end;

% save test_accuracies.mat accuracies

% % plot(h_unit_vector,val_accuracy);
% xlabel('Number of Hidden Units');
% ylabel('Overall validation accuracy');
% title('Accuracy vs Hidden Units');


% latex_table = latex(sym(accuracies))





