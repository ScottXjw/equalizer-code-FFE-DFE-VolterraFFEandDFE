function [y,e,w] = ffe_lms(sym_pam,ref_sym_pam,train_len,test_len,taps_num,step_len,delay)
% FFE 使用lms更新抽头系数
% sym_pam 滤波器输入信号,行向量
% ref_sym_pam  参考信号，行向量
% train_len 训练长度，int
% test_len 测试长度，int
% taps_num 抽头数，最好是奇数
% step_len 步长，double 
% delay 延迟，int
sym_pam = sym_pam(:).';
ref_sym_pam = ref_sym_pam(:).';
%初始化
w = zeros(taps_num,1);

%% train 训练
for i_train = 1:train_len 
    e(i_train) = ref_sym_pam(i_train+delay) - sym_pam(i_train : i_train+taps_num-1) * w;
    
    %使用lms更新抽头
    w = w + step_len * e(i_train) * sym_pam(i_train : i_train+taps_num-1).';
end

% figure;plot(abs(e)) % 看误差曲线
% figure;plot(w) % 看抽头分布

%% test测试
for i_test = train_len+1:train_len+test_len 
    y(i_test-train_len) =  sym_pam(i_test : i_test+taps_num-1) * w;  
end


end





