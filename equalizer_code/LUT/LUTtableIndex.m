function [LUT_e] = LUTtableIndex(x,y,len,M)
%LUTTABLEINDEX 此处显示有关此函数的摘要 PAM4的
%  主要是得出查找表
%  x 参考信号
%  y 受到损伤的信号
%  len 记忆长度  需要是奇数
%  M 调制阶数  

x = x(:);
y = y(:);
%首先建好不同情况的初始表
LUT = zeros(1,M^(len));
count = zeros(1,M^(len));
%那么要对应好不同的情况和LUT的关系
x_inedx = mapminmax(x.',0,3).';

N = fix(len/2);
%滑动窗口
for i = N+1:length(x)-N
    %根据参考信号
    %获得每个滑动窗口所对应的下标
    ind = num2str(x_inedx(i-N:i+N).','%d');
    index = base2dec(ind,M)+1;
    LUT(index) = LUT(index) + (y(i)-x(i));
    count(index) = count(index) + 1;
end
LUT_e = LUT./count;

end

