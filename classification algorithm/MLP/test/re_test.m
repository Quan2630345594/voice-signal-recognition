% 清空环境
clear; close all; clc;

final_matrix1=load('final_matrix.mat');
final_matrix2=struct2cell(final_matrix1);
final_matrix=cell2mat(final_matrix2);
GroupTrain1=load('GroupTrain.mat');%训练标签
GroupTrain2=struct2cell(GroupTrain1);
GroupTrain=cell2mat(GroupTrain2);
load('net.mat')
Test_num=7500;
accur_num=0;
for i=1:Test_num
    test_data = final_matrix(i,:)';
    test_output = round(trained_net(test_data));
    if(test_output==GroupTrain(i,1))
        accur_num=accur_num+1;
    end
end
accuracy=accur_num/Test_num;
disp('正确预测个数为：');
disp(accur_num);
disp('正确率为')
disp(accuracy);