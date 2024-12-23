samples=[];
dir_info=dir('symbol/*.mat');
for i=1:length(dir_info)
    file_name=dir_info(i).name;
    file_path=fullfile('symbol',file_name);
    data=load(file_path);
    data_matrix=struct2cell(data);
    temp_matrix1=cell2mat(data_matrix);
    samples(:,:,i)=temp_matrix1;
end
labels=[];%训练标签
for i=1:75
    for j=0:9
        for k=1:10
         labels=[labels;j];
        end
    end
end

%% random sample
r=randperm(size(samples,3));
samples=samples(:,:,r);
labels=labels(r);
%% dataset
ch=[5 6 9];
ptt=0.8;
trainsize=ceil(size(samples,3)*ptt);
x_train=samples(ch,:,1:trainsize);
x_train=reshape(x_train,[size(x_train,[1 2]),1,size(x_train,3)]);
y_train=categorical(labels(1:trainsize));
x_test=samples(ch,:,trainsize+1:size(samples,3));
x_test=reshape(x_test,[size(x_test,[1 2]),1,size(x_test,3)]);
y_test=categorical(labels(trainsize+1:size(samples,3)));

%%搭建网络并训练预测
%% set net
layers = [ ...
    %input
    imageInputLayer([size(x_train,[1 2 3])])
    %c 1
    convolution2dLayer(2,4)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    %output
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

% analyzeNetwork(layers);
%% set options
options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'Shuffle','every-epoch',...
    'MaxEpochs',10, ...
    'MiniBatchSize',100, ...
    'InitialLearnRate',0.001,...
    'Plots','training-progress');
    
%% train
net=trainNetwork(x_train,y_train,layers,options);

% save net;
% load net.mat

%% test
preds=classify(net,x_train);
acc=size(find((double(preds)-double(y_train))==0),1)/size(y_train,1);
fprintf(strcat('train acc:',num2str(acc*100),'%%\n'));

preds=classify(net,x_test);
acc=size(find((double(preds)-double(y_test))==0),1)/size(y_test,1);
fprintf(strcat('test acc:',num2str(acc*100),'%%\n'));
plotconfusion(transpose(round(double(preds))-1),transpose(round(double(y_test))-1));
