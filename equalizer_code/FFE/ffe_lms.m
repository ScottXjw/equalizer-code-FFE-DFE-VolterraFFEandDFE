function [y,e,w] = ffe_lms(sym_pam,ref_sym_pam,train_len,test_len,taps_num,step_len,delay)
% FFE ʹ��lms���³�ͷϵ��
% sym_pam �˲��������ź�,������
% ref_sym_pam  �ο��źţ�������
% train_len ѵ�����ȣ�int
% test_len ���Գ��ȣ�int
% taps_num ��ͷ�������������
% step_len ������double 
% delay �ӳ٣�int
sym_pam = sym_pam(:).';
ref_sym_pam = ref_sym_pam(:).';
%��ʼ��
w = zeros(taps_num,1);

%% train ѵ��
for i_train = 1:train_len 
    e(i_train) = ref_sym_pam(i_train+delay) - sym_pam(i_train : i_train+taps_num-1) * w;
    
    %ʹ��lms���³�ͷ
    w = w + step_len * e(i_train) * sym_pam(i_train : i_train+taps_num-1).';
end

% figure;plot(abs(e)) % ���������
% figure;plot(w) % ����ͷ�ֲ�

%% test����
for i_test = train_len+1:train_len+test_len 
    y(i_test-train_len) =  sym_pam(i_test : i_test+taps_num-1) * w;  
end


end





