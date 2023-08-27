function [y,e,w] = volterra_ffe_dfe_rls(sym_pam,ref_sym_pam,train_len,test_len,taps_list,lamda,delay)
% % ����Volterra����LMS�㷨
% % sym_pam �˲��������ź�,������
% % ref_sym_pam �����źţ�������
% % taps_list�а����ķֱ���һ�ס����ס����׵ļ��䳤��
sym_pam = sym_pam(:).';
ref_sym_pam = ref_sym_pam(:).';

tapslen_1 = taps_list(1);
tapslen_2 = taps_list(2);
tapslen_3 = taps_list(3);

fblen_1 = taps_list(4);
fblen_2 = taps_list(5);
fblen_3 = taps_list(6);

%��ʼ��
w = zeros(tapslen_1+tapslen_2*(tapslen_2+1)/2 + tapslen_3*(tapslen_3+1)*(tapslen_3+2)/6 +...
    fblen_1+fblen_2*(fblen_2+1)/2 + fblen_3*(fblen_3+1)*(fblen_3+2)/6 ,1);
fb = zeros(1,fblen_1);
SD = eye(length(w));

%% train ѵ��
for i_train = 1:train_len
    %����volterra����
    %һ��ǰ������
    x1 = sym_pam(i_train : i_train+tapslen_1-1);
    %����ǰ������
    x2 = x1(round((tapslen_1-tapslen_2)/2)+1 : end - fix((tapslen_1-tapslen_2)/2));
    x2_vol = BuildVolterraInput(x2,2);
    %����ǰ������
    x3 = x1(round((tapslen_1-tapslen_3)/2)+1 : end - fix((tapslen_1-tapslen_3)/2));
    x3_vol = BuildVolterraInput(x3,3);
    %һ�׷�������
    fb1_vol = fb(1:fblen_1);
    %���׷�������
    fb2_vol = BuildVolterraInput(fb(1:fblen_2),2);
    %���׷�������
    fb3_vol = BuildVolterraInput(fb(1:fblen_3),3);
    %�����������
    x_all = [x1 x2_vol x3_vol fb1_vol fb2_vol fb3_vol].';
    
    e(i_train) = ref_sym_pam(i_train+delay) - x_all.' * w;
    
    %ʹ��rls���³�ͷ
    SD = ( SD - ((SD * x_all) * (SD * x_all).') / (lamda + (SD * x_all).' * x_all ) ) / lamda;
    w = w +  e(i_train) * SD * x_all;
    
    %��������
    fb = [ref_sym_pam(i_train+delay) fb(1:end-1)];
    
end
% figure;plot(abs(e)) % ���������
% figure;plot(w) % ����ͷ�ֲ�

fb = zeros(1,fblen_1);
%% test����
for i_test = train_len+1:train_len+test_len
    %����volterra����
    x1 = sym_pam(i_test : i_test+tapslen_1-1);
    x2 = x1(round((tapslen_1-tapslen_2)/2)+1 : end - fix((tapslen_1-tapslen_2)/2));
    x3 = x1(round((tapslen_1-tapslen_3)/2)+1 : end - fix((tapslen_1-tapslen_3)/2));
    %��������
    x2_vol = BuildVolterraInput(x2,2);
    %��������
    x3_vol = BuildVolterraInput(x3,3);
    %һ�׷�������
    fb1_vol = fb(1:fblen_1);
    %���׷�������
    fb2_vol = BuildVolterraInput(fb(1:fblen_2),2);
    %���׷�������
    fb3_vol = BuildVolterraInput(fb(1:fblen_3),3);
    %�����������
    x_all = [x1 x2_vol x3_vol fb1_vol fb2_vol fb3_vol];
    
    y(i_test-train_len) =  x_all * w;
    
    %��������
    if  fblen_1 ~= 0
        fb = [pammod(pamdemod(y(i_test-train_len),4),4) fb(1:end-1)];
    end
end

end
