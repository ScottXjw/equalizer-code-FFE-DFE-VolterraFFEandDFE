clc,clear,close all;

%% 一些参数
mod_order = 4;    %调制阶数
sym_num = 100000;   %传输符号数 ，一般是PRBS码调制而成的，本文使用随机信号调制
sps = 4;          % 上采样倍数
fir_len = 100;     % 滤波器参数
cutoff_factor = 0.0001;   % 滤波器参数


%% 生成PAM信号
sym = fix(mod_order*rand([1 sym_num]));
sym_pam = pammod(sym,mod_order);

%% 上采样
sym_up_pam = kron(sym_pam,[1 ones(1,sps-1)]);

%% 滤波    相当于人为加入符号间干扰/码间串扰（ISI)
w = rcosdesign(cutoff_factor,fir_len,sps,'sqrt');
% 也可以rcosine
sym_filter_up_pam = conv(sym_up_pam,w);

%对齐
sym_filter_up_pam = sym_filter_up_pam(round(length(w)/2):end-fix(length(w)/2));

for i = 1:10
    snr(i) = i+9;
    %% 加噪声
    sym_noise_filter_up_pam = awgn(sym_filter_up_pam,snr(i),'measured');
    
    %% 下采样
    % 是否需要下采样要根据均衡的结构
    sym_noise_filter_down_pam = sym_noise_filter_up_pam(round(sps/2):sps:end);
    
    %% ffe_lms均衡
    train_len = 3000;
    test_len = 90000;
    taps_num = 31;
    step_len = 0.0001;
    delay = fix(taps_num/2);
    [equalizer_pam_lms,e_lms,w_lms] = ffe_lms(sym_noise_filter_down_pam,sym_pam,train_len,test_len,taps_num,step_len,delay);
    
    %% ffe_rls均衡
    lamda = 0.9999;
    [equalizer_pam_rls,e_rls,w_rls] = ffe_rls(sym_noise_filter_down_pam,sym_pam,train_len,test_len,taps_num,lamda,delay);
    
    %% 判决
    sym_noise_filter_up_lms = pamdemod(equalizer_pam_lms,mod_order);
    sym_noise_filter_up_rls = pamdemod(equalizer_pam_rls,mod_order);
    
    %% 计算误码率
    [~,BER_lms(i)] = biterr(sym_noise_filter_up_lms.',sym(train_len+delay+1:train_len+delay+test_len).',log2(mod_order));
    [~,BER_rls(i)] = biterr(sym_noise_filter_up_rls.',sym(train_len+delay+1:train_len+delay+test_len).',log2(mod_order));
    %% 画图
    %抽头对比图
    figure
    plot(w_lms)
    hold on
    plot(w_rls)
    legend("FFE-LMS","FFE-RLS")
    title("均衡器抽头")
    
    %训练的对比图
    figure
    plot(abs(e_lms))
    hold on
    plot(abs(e_rls))
    legend("FFE-LMS","FFE-RLS")
    xlabel("迭代次数")
    ylabel("误差")
end

figure
semilogy(snr,BER_lms,'-o')
hold on
semilogy(snr,BER_rls,'-*')
grid on
legend("FFE-LMS","FFE-RLS")
xlabel("信噪比")
ylabel("误码率")
