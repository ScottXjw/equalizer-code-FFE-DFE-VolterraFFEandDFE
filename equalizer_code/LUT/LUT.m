function [y] = LUT(x,LUT_e,len,M)
%  LUT ���ݲ��ұ�����Ԥ����
%  x �ܵ����˵��ź�
%   LUT_e  ��
%  len ���䳤��  ��Ҫ������
%  M ���ƽ���  
x=x(:);
y = x;
x_inedx = mapminmax(x.',0,3).';

N = fix(len/2);
for i = N+1:length(x)-N
    %���ݲο��ź�
    %���ÿ��������������Ӧ���±�
    ind = num2str(x_inedx(i-N:i+N).','%d');
    index = base2dec(ind,M)+1;
    y(i) = x(i)-LUT_e(index);
end

end

