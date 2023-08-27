clc,clear,close all;

%% 一些参数
mod_order = 4;    %调制阶数
sym_num = 100000;   %传输符号数 ，一般是PRBS码调制而成的，本文使用随机信号调制
sps = 4;          % 上采样倍数
fir_len = 100;     % 滤波器参数
cutoff_factor = 0.0001;   % 滤波器参数
snr = 20;     % 设置的信噪比

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

%% 加噪声
sym_noise_filter_up_pam = awgn(sym_filter_up_pam,snr,'measured');

%% 下采样 
% 是否需要下采样要根据均衡的结构
sym_noise_filter_down_pam = sym_noise_filter_up_pam(round(sps/2):sps:end);

%% LUT预均衡 --求表
[LUT_e] = LUTtableIndex(sym_pam,sym_noise_filter_down_pam,5,4);


%%
%以上部分是求出查找表的表
%% 

%% 生成PAM信号
sym = fix(mod_order*rand([1 sym_num]));
sym_pam = pammod(sym,mod_order);

%% LUT预均衡
sym_pam = LUT(sym_pam,LUT_e,5,4).';

%% 上采样
sym_up_pam = kron(sym_pam,[1 ones(1,sps-1)]);

%% 滤波    相当于人为加入符号间干扰/码间串扰（ISI)
sym_filter_up_pam = conv(sym_up_pam,w);

%对齐
sym_filter_up_pam = sym_filter_up_pam(round(length(w)/2):end-fix(length(w)/2));

%% 加噪声
sym_noise_filter_up_pam = awgn(sym_filter_up_pam,snr,'measured');

%% 下采样 
sym_noise_filter_down_pam = sym_noise_filter_up_pam(round(sps/2):sps:end);


%% 判决
sym_noise_filter_up_lut = pamdemod(sym_noise_filter_down_pam,mod_order);

%% 计算误码率
[~,BER_lut] = biterr(sym_noise_filter_up_lut.',sym.',log2(mod_order));
