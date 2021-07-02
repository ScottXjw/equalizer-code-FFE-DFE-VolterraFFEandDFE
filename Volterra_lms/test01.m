clc;clear;

%随机生成
pam4 = mapminmax(round(3*rand(1,20000)),-3,3);

% actual_isi=[0.05 -0.063 0.088 -0.126 -0.25 0.9047 0.25 0 0.126 0.038 0.088];
actual_isi = [0.05 -0.063 0.088 -0.126 -0.245 0.9047 0.25 0 0.126 0.038 0.002 0.001 0.088];
y = conv(pam4,actual_isi);

y = y(7:end-6);

%归一化
pam4 = mapminmax(pam4,0,3);
y = mapminmax(y,0,3);

% figure
% plot(xcorr(pam4,pam4 ))
% find(xcorr(pam4,pam4 ) == max(xcorr(pam4,pam4 )))
% figure
% plot(xcorr(y,pam4 ))
% find(xcorr(y,pam4 ) == max(xcorr(y,pam4 )) )


training_length = round(length(pam4)/3 * 2);

training = pam4(1:training_length);
training_y =  y(1:training_length);


M = 21;

[W_3jievoterralms,e_3jievolterralms] = Volterra3jie_LMS( training_y,training,M,0,0);
plot(e_3jievolterralms)

%测试长度
test_length = length(y);
fix_d = fix(M/2);
y = [zeros(1,fix_d) y zeros(1,fix_d)]; 

%-------------------------三阶Volterra_LMS算法的预测结果 ------%
pam4_sym_est_3jieVolterra = Volterra3jie_LMStest(y,W_3jievoterralms(:,end),test_length,M,0,0);


ber_3jieVolterra = BER03(pam4_sym_est_3jieVolterra,pam4);









