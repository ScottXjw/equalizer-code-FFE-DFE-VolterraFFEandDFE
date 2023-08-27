function [y] = LUT(x,LUT_e,len,M)
%  LUT 根据查找表来做预均衡
%  x 受到损伤的信号
%   LUT_e  表
%  len 记忆长度  需要是奇数
%  M 调制阶数  
x=x(:);
y = x;
x_inedx = mapminmax(x.',0,3).';

N = fix(len/2);
for i = N+1:length(x)-N
    %根据参考信号
    %获得每个滑动窗口所对应的下标
    ind = num2str(x_inedx(i-N:i+N).','%d');
    index = base2dec(ind,M)+1;
    y(i) = x(i)-LUT_e(index);
end

end

