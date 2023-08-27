function [LUT_e] = LUTtableIndex(x,y,len,M)
%LUTTABLEINDEX �˴���ʾ�йش˺�����ժҪ PAM4��
%  ��Ҫ�ǵó����ұ�
%  x �ο��ź�
%  y �ܵ����˵��ź�
%  len ���䳤��  ��Ҫ������
%  M ���ƽ���  

x = x(:);
y = y(:);
%���Ƚ��ò�ͬ����ĳ�ʼ��
LUT = zeros(1,M^(len));
count = zeros(1,M^(len));
%��ôҪ��Ӧ�ò�ͬ�������LUT�Ĺ�ϵ
x_inedx = mapminmax(x.',0,3).';

N = fix(len/2);
%��������
for i = N+1:length(x)-N
    %���ݲο��ź�
    %���ÿ��������������Ӧ���±�
    ind = num2str(x_inedx(i-N:i+N).','%d');
    index = base2dec(ind,M)+1;
    LUT(index) = LUT(index) + (y(i)-x(i));
    count(index) = count(index) + 1;
end
LUT_e = LUT./count;

end

