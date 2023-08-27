function [y,e,w] = volterra_ffe_dfe_rls(sym_pam,ref_sym_pam,train_len,test_len,taps_list,lamda,delay)
% % 三阶Volterra级数LMS算法
% % sym_pam 滤波器输入信号,行向量
% % ref_sym_pam 期望信号，行向量
% % taps_list中包含的分别是一阶、二阶、三阶的记忆长度
sym_pam = sym_pam(:).';
ref_sym_pam = ref_sym_pam(:).';

tapslen_1 = taps_list(1);
tapslen_2 = taps_list(2);
tapslen_3 = taps_list(3);

fblen_1 = taps_list(4);
fblen_2 = taps_list(5);
fblen_3 = taps_list(6);

%初始化
w = zeros(tapslen_1+tapslen_2*(tapslen_2+1)/2 + tapslen_3*(tapslen_3+1)*(tapslen_3+2)/6 +...
    fblen_1+fblen_2*(fblen_2+1)/2 + fblen_3*(fblen_3+1)*(fblen_3+2)/6 ,1);
fb = zeros(1,fblen_1);
SD = eye(length(w));

%% train 训练
for i_train = 1:train_len
    %构建volterra输入
    %一阶前馈输入
    x1 = sym_pam(i_train : i_train+tapslen_1-1);
    %二阶前馈输入
    x2 = x1(round((tapslen_1-tapslen_2)/2)+1 : end - fix((tapslen_1-tapslen_2)/2));
    x2_vol = BuildVolterraInput(x2,2);
    %三阶前馈输入
    x3 = x1(round((tapslen_1-tapslen_3)/2)+1 : end - fix((tapslen_1-tapslen_3)/2));
    x3_vol = BuildVolterraInput(x3,3);
    %一阶反馈输入
    fb1_vol = fb(1:fblen_1);
    %二阶反馈输入
    fb2_vol = BuildVolterraInput(fb(1:fblen_2),2);
    %三阶反馈输入
    fb3_vol = BuildVolterraInput(fb(1:fblen_3),3);
    %组合所有输入
    x_all = [x1 x2_vol x3_vol fb1_vol fb2_vol fb3_vol].';
    
    e(i_train) = ref_sym_pam(i_train+delay) - x_all.' * w;
    
    %使用rls更新抽头
    SD = ( SD - ((SD * x_all) * (SD * x_all).') / (lamda + (SD * x_all).' * x_all ) ) / lamda;
    w = w +  e(i_train) * SD * x_all;
    
    %反馈更新
    fb = [ref_sym_pam(i_train+delay) fb(1:end-1)];
    
end
% figure;plot(abs(e)) % 看误差曲线
% figure;plot(w) % 看抽头分布

fb = zeros(1,fblen_1);
%% test测试
for i_test = train_len+1:train_len+test_len
    %构建volterra输入
    x1 = sym_pam(i_test : i_test+tapslen_1-1);
    x2 = x1(round((tapslen_1-tapslen_2)/2)+1 : end - fix((tapslen_1-tapslen_2)/2));
    x3 = x1(round((tapslen_1-tapslen_3)/2)+1 : end - fix((tapslen_1-tapslen_3)/2));
    %二阶输入
    x2_vol = BuildVolterraInput(x2,2);
    %三阶输入
    x3_vol = BuildVolterraInput(x3,3);
    %一阶反馈输入
    fb1_vol = fb(1:fblen_1);
    %二阶反馈输入
    fb2_vol = BuildVolterraInput(fb(1:fblen_2),2);
    %三阶反馈输入
    fb3_vol = BuildVolterraInput(fb(1:fblen_3),3);
    %组合所有输入
    x_all = [x1 x2_vol x3_vol fb1_vol fb2_vol fb3_vol];
    
    y(i_test-train_len) =  x_all * w;
    
    %反馈更新
    if  fblen_1 ~= 0
        fb = [pammod(pamdemod(y(i_test-train_len),4),4) fb(1:end-1)];
    end
end

end
