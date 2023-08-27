function [y,e,w] = ffe_rls(sym_pam,ref_sym_pam,train_len,test_len,taps_num,lamda,delay)
% FFE ʹ��rls���³�ͷϵ��
% sym_pam �˲��������ź�,������
% ref_sym_pam  �ο��źţ�������
% train_len ѵ�����ȣ�int
% test_len ���Գ��ȣ�int
% taps_num ��ͷ�������������
% lamda �������ӣ�double 
% delay �ӳ٣�int

sym_pam = sym_pam(:).';
ref_sym_pam = ref_sym_pam(:).';
%��ʼ��
w = zeros(taps_num,1);
SD = eye(taps_num);

%% train ѵ��
for i_train = 1:train_len 
    x = sym_pam(i_train : i_train+taps_num-1).';
    e(i_train) = ref_sym_pam(i_train+delay) - x.' * w;
    
    %ʹ��rls���³�ͷ
    SD = ( SD - ((SD * x) * (SD * x).') / (lamda + (SD * x).' * x ) ) / lamda;
    w = w +  e(i_train) * SD * x;
end

% figure;plot(abs(e)) % ���������
% figure;plot(w) % ����ͷ�ֲ�

%% test����
for i_test = train_len+1:train_len+test_len 
    y(i_test-train_len) =  sym_pam(i_test : i_test+taps_num-1) * w;  
end
end

