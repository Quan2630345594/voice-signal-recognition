% 清空环境
clear; close all; clc;

final_matrix1=load('final_matrix.mat');
final_matrix2=struct2cell(final_matrix1);
final_matrix=cell2mat(final_matrix2);
GroupTrain1=load('GroupTrain.mat');%训练标签
GroupTrain2=struct2cell(GroupTrain1);
GroupTrain=cell2mat(GroupTrain2);

% 创建输入和目标数据
% 输入数据: 4个样本，每个样本有2个特征
input_data = final_matrix';
% 目标数据: 4个样本，每个样本有1个输出
target_data = GroupTrain';

% 创建神经网络
% 创建一个具有2个输入神经元、2个隐层神经元和1个输出神经元的神经网络
net = feedforwardnet(2);

% 设置训练参数
net.trainParam.epochs = 100; % 最大训练轮次
net.trainParam.goal = 1e-5; % 训练目标误差
net.trainParam.lr = 0.1; % 学习率

% 划分数据集
% 将所有数据用于训练（不推荐，仅用于此示例）
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 20/100;

% 训练神经网络
[trained_net, tr] = train(net, input_data, target_data);

save('net.mat','trained_net');       % 将网络net保存为.mat文件，后面可直接调用
%load('net.mat');     % 导入之前保存的网络

% 测试神经网络
Test_num=7500;
accur_num=0;
for i=1:7500
    test_data = final_matrix(i,:)';
    test_output = round(trained_net(test_data));
    if(test_output==GroupTrain(i,1))
        accur_num=accur_num+1;
    end
end
accuracy=accur_num/Test_num;
disp(accuracy);


% 显示结果
%disp("测试输出：");
%disp();

% 可视化神经网络结构
view(trained_net);

% 可视化训练过程中的性能
figure;
plotperform(tr);
title('训练过程中的性能');

% 可视化训练过程中的梯度
figure;  
plot(tr.gradient);
xlabel('迭代次数');
ylabel('梯度');
title('训练过程中的梯度');