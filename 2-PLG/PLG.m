clc;clear;close all;warning off;
prompt2 = 'Please type the name of dataset \n (e.g., UP, SA, KSC) : ';
name = input(prompt2,'s');
if strcmpi('UP', name)
    load ../log/RCN/embeddings/UP_embeddings
    load ../log/RCN/embeddings/UP_test_set_idx
    load ../log/RCN/embeddings/UP_train_set_idx
    a = embeddingsUP;
    o = test_set_idx_UP;
    l1 = train_set_idx_UP;
    d = 20 ; % window size
    
elseif strcmpi('SA', name)
    load ../log/RCN/embeddings/SA_embeddings
    load ../log/RCN/embeddings/SA_test_set_idx
    load ../log/RCN/embeddings/SA_train_set_idx
    a = embeddingsSA;
    o = test_set_idx_SA;
    l1 = train_set_idx_SA;
    d = 10 ; % window size
    
elseif strcmpi('KSC', name)
    load ../log/RCN/embeddings/KSC_embeddings
    load ../log/RCN/embeddings/KSC_test_set_idx
    load ../log/RCN/embeddings/KSC_train_set_idx
    a = embeddingsKSC;
    o = test_set_idx_KSC;
    l1 = train_set_idx_KSC;
    d = 5 ; % window size
end
%%
tic
s = 10; % The number of training samples in each class
samp_R = 0.5; % The level of confidence
sample_num = s; % number of samples selected in each class
window_size = d;
%%
height = size(a,1);
width = size(a,2);
channels = size(a,3);
a = reshape(a , [] , channels);
o = reshape(o , [], 1);
l = reshape(l1 , [] , 1);

iter = max(l);
index_vec =[];
for i = 1 : iter
    index_vec (i , : ) = find(l == i);
end
index_vec_r =[];
index_vec_c =[];
 for i = 1 : iter
     [x  , y] = find(l1 == i);
     index_vec_r (i,:) = x;
     index_vec_c (i,:) = y;
 end

index_window_r_right = index_vec_r +d;
index_window_r_left = index_vec_r -d;

index_window_c_right = index_vec_c +d;
index_window_c_left = index_vec_c -d;

for j= 1 : length(index_window_r_right(1,:));
    for i = 1 : length(index_window_r_right(:,1));
        if index_window_r_right(i,j) > height;
            index_window_r_right(i,j) = height;
        end
        if index_window_r_left (i,j) <=0;
            index_window_r_left(i,j)=1;
        end
    end
end

for j= 1 : length(index_window_c_right(1,:));
    for i = 1 : length(index_window_c_right(:,1));
        if index_window_c_right(i,j) > width;
            index_window_c_right(i,j) = width;
        end
        if index_window_c_left (i,j) <=0;
            index_window_c_left(i,j)=1;
        end
    end
end


F_matrix = zeros(length(index_window_r_right(:,1) ) , height * width );

for i = 1:length(index_window_r_right(:,1));  % or for i = 1 : iter
    
        f = zeros(height, width);
        for j =  1:length(index_window_r_right(1,:) );
            f(index_window_r_left(i,j) : index_window_r_right(i,j) , ...
            index_window_c_left(i,j) : index_window_c_right(i,j) ) = i;
        end
        F_matrix(i,:)= reshape(f, [] , 1);
        index_v{i} = find(F_matrix(i,:) == i);
        
        
end

for i = 1:iter;
 index_u{i} = a ((F_matrix(i,:) == i) , :);
 index_a{i} = a (index_vec(i,:) , :);
 index_a_mean {i} =  mean(index_a{i});
end




for i = 1:iter; 
    for j = 1 : sample_num
         q_vec{i}(j,1) =  corr(( index_a_mean {i})' , (index_a{i}(j , :))');
         E_vec{i} (j,1) =  wentropy(index_a{i}(j , :), 'log energy');

    end
end

% find the info of the distribution
for  i = 1:iter;
    pd_inf{i} = fitdist(q_vec{i},'Normal');
    pe_inf{i} = fitdist(E_vec{i},'Normal');

end

%%% -------------- comment section
for  i = 1:iter;
    for j = 1: size ( index_u{i}, 1);
        p_vec{i}(j,1) = corr(( index_a_mean {i})' , (index_u{i}(j , :))');
        pp_vec{i}(j,1) = finddelay( ( index_a_mean {i}) , index_u{i}(j , :) ); 
        EE{i}(j,1) = wentropy(index_u{i}(j , :), 'log energy');

    end

end
%%---------------------------------

for i = 1 : iter;
    mu_v{i} = paramci( pd_inf{i} , 'Alpha',samp_R);
    index_mu{i} = find(p_vec{i} >= mu_v{i}(1,1)  & p_vec{i}<= mu_v{i}(2,1)  );

end   

%% train
train_v = zeros(height * width, 1);

 for i = 1 : iter;
     train_v ( index_v{i}(index_mu{i}) , 1) = i;
     train_v (index_vec(i,:)) = i;
 end

 toc 
 %% Illustration
 train_v = reshape(train_v , height , []);
train_L = label2rgb(train_v); train_L_O = label2rgb(reshape(o , height , []));
figure;subplot(1,2,1);imshow(train_L);
title('sampling pixel using sample_num original pixel');subplot(1,2,2);
imshow(train_L_O);title('original Ground Truth');

L = find(l1);
o(L) = 0;o = reshape(o , height , []);
figure;imshow(label2rgb(o));paviaU_gt1 = o;
% save('paviaU_gt1.mat' , 'paviaU_gt1')

%% Excel report of PLG

new_train = train_v; a = find(new_train);
b = find(o); inter = intersect(a,b);
yhat = new_train(inter);
ytrue = o(inter);
gtv = double(ytrue);
clmap = yhat;

accuracy = classperf(gtv, clmap);
accuracy_err = accuracy.ErrorRate;
num_sample_g = size(yhat);

PseudoMatchWithGt=[]; SizeOfWindow=[]; NumberOfPseudo=[]; 
PseudoMatchWithGtByPercent=[]; ErrorOfPseudoByPercent=[];
portion = (size(inter,1)/size(a,1)) * 100;
PseudoMatchWithGt = [PseudoMatchWithGt num_sample_g(1)];
SizeOfWindow = [SizeOfWindow window_size];
NumberOfPseudo = [NumberOfPseudo size(a,1)];
PseudoMatchWithGtByPercent = [PseudoMatchWithGtByPercent portion];
ErrorOfPseudoByPercent = [ErrorOfPseudoByPercent accuracy_err];
%end
T = table(SizeOfWindow, NumberOfPseudo, PseudoMatchWithGt, ...
    PseudoMatchWithGtByPercent, ErrorOfPseudoByPercent);
% T = [ transpose(nv1) transpose(nv3) transpose(nv)  ...
%     transpose(nv4) transpose(nv2) ...
%  ];

if strcmpi('UP', name)
    filename = 'Report/UP_PC_info.csv';
    save('../Datasets/UP/new_train.mat','new_train')
    writetable(T, filename)
    load ../log/RCN/embeddings/UP_test_set_idx
    save('../Datasets/UP/UP_test_set_idx.mat','test_set_idx_UP')
elseif strcmpi('SA', name)
    filename = 'Report/SA_PC_info.csv';
    new_train = train_v;save('../Datasets/SA/new_train.mat','new_train')
    writetable(T, filename)
    load ../log/RCN/embeddings/SA_test_set_idx
    save('../Datasets/SA/SA_test_set_idx.mat','test_set_idx_SA')
elseif strcmpi('KSC', name)
    filename = 'Report/KSC_PC_info.csv';
    new_train = train_v;save('../Datasets/KSC/new_train.mat','new_train')
    writetable(T, filename)
    load ../log/RCN/embeddings/KSC_test_set_idx
    save('../Datasets/KSC/KSC_test_set_idx.mat','test_set_idx_KSC')
end

