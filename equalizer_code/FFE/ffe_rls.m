function [y,e,w] = ffe_rls(sym_pam,ref_sym_pam,train_len,test_len,taps_num,lamda,delay)
% FFE 使用rls更新抽头系数
% sym_pam 滤波器输入信号,行向量
% ref_sym_pam  参考信号，行向量
% train_len 训练长度，int
% test_len 测试长度，int
% taps_num 抽头数，最好是奇数
% lamda 遗忘因子，double 
% delay 延迟，int

sym_pam = sym_pam(:).';
ref_sym_pam = ref_sym_pam(:).';
%初始化
w = zeros(taps_num,1);
SD = eye(taps_num);

%% train 训练
for i_train = 1:train_len 
    x = sym_pam(i_train : i_train+taps_num-1).';
    e(i_train) = ref_sym_pam(i_train+delay) - x.' * w;
    
    %使用rls更新抽头
    SD = ( SD - ((SD * x) * (SD * x).') / (lamda + (SD * x).' * x ) ) / lamda;
    w = w +  e(i_train) * SD * x;
end

% figure;plot(abs(e)) % 看误差曲线
% figure;plot(w) % 看抽头分布

%% test测试
for i_test = train_len+1:train_len+test_len 
    y(i_test-train_len) =  sym_pam(i_test : i_test+taps_num-1) * w;  
end
end

