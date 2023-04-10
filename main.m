clc,clear,close all;
addpath("PostEqualizer")
%% 一些参数
mod_order = 4;
sym_num = 100000;
sps = 4;
fir_len = 100;
cutoff_factor = 0.0001;
snr = 15;

%% 生成PAM信号
sym = fix(mod_order*rand([1 sym_num]));
sym_pam = pammod(sym,mod_order);
% figure 
% pwelch(sym_pam)

%% 上采样
sym_up_pam = kron(sym_pam,[1 ones(1,sps-1)]);

%% 滤波    相当于人为加入符号间干扰/码间串扰（ISI)
w = rcosdesign(cutoff_factor,fir_len,sps,'sqrt'); 
% 也可以rcosine
% figure 
% pwelch(w)
sym_filter_up_pam = conv(sym_up_pam,w);
% figure 
% pwelch(sym_filter_up_pam(1:sps:end))

%对齐
sym_filter_up_pam = sym_filter_up_pam(round(length(w)/2):end-fix(length(w)/2));

%% 加噪声
sym_noise_filter_up_pam = awgn(sym_filter_up_pam,snr,'measured');

% figure 
% pwelch(sym_noise_filter_up_pam(round(sps/2):sps:end))

%% 下采样 
% 是否需要下采样要根据均衡的结构
sym_noise_filter_down_pam = sym_noise_filter_up_pam(round(sps/2):sps:end);


%% ffe_lms均衡
% train_len = 2000;
% test_len = 50000;
% taps_num = 9;
% step_len = 0.001;
% delay = fix(taps_num/2);
% equalizer_pam = ffe_lms(sym_noise_filter_down_pam,sym_pam,train_len,test_len,taps_num,step_len,delay);

%% ffe_rls均衡
train_len = 4000;
test_len = 50000;
taps_num = 21;
lamda = 0.9999;
delay = fix(taps_num/2);
equalizer_pam = ffe_rls(sym_noise_filter_down_pam,sym_pam,train_len,test_len,taps_num,lamda,delay);

%% volterra_ffe_dfe_lms均衡
% train_len = 4000;
% test_len = 50000;
% taps = [21 11 5 9 7 5];
% step_len = 0.001;
% delay = fix(taps(1)/2);
% equalizer_pam = volterra_ffe_dfe_lms(sym_noise_filter_down_pam,sym_pam,train_len,test_len,taps,step_len,delay);

%% volterra_ffe_dfe_rls均衡
% train_len = 4000;
% test_len = 50000;
% taps = [21 11 5 20 7 5];
% lamda = 0.9999;
% delay = fix(taps(1)/2);
% equalizer_pam = volterra_ffe_dfe_rls(sym_noise_filter_down_pam,sym_pam,train_len,test_len,taps,lamda,delay);
% 


%% 判决
sym_noise_filter_up = pamdemod(equalizer_pam,mod_order);

%% 计算误码率
[error_bit,BER] = biterr(sym_noise_filter_up.',sym(train_len+delay+1:train_len+delay+test_len).',log2(mod_order));
