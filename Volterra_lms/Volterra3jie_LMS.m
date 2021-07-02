function [ w,e ] = Volterra3jie_LMS( x,d,l1,l2,l3 )
% 二阶Volterra级数LMS算法
% x 滤波器输入信号,行向量
% d 期望信号，行向量
% N 抽头数， 奇数 
% l1、l2、l3分别是一阶、二阶、三阶的记忆长度{N,N(N+1)/2，N(N+1)(N+2)/6}
% 注意：l1\l2\l3都得是奇数，设计的时候就是按照奇数设计的。偶数没有试过，可能会出现不对齐等问题


%一阶部分
w1_length = l1;    %权重的数量
fix_d1 = fix(w1_length/2);
%为了收敛的效果，按最大的记忆长度来
d = d(fix_d1+1:end-(w1_length-fix_d1-1)); %% 对齐
training_length = length(d); %训练长度
x1_fact = zeros(w1_length,training_length); %初始化一阶输入

%二阶部分
w2_length = l2*(l2+1)/2;    %权重的数量
fix_d2 = fix(l2/2);
x2_fact = zeros(w2_length,training_length); %初始化二阶输入

%三阶部分
w3_length = l3*(l3+1)*(l3+2)/6;    %权重的数量
fix_d3 = fix(l3/2);
x3_fact = zeros(w3_length,training_length); %初始化三阶输入


%将x(k-N)到x(k) 全部 装进xi_fact
for i = 1:l1
    x1_fact(i,:) = x(i:i+training_length-1);  
end

index2 = 0;
for i = 1:l2
    x1 = x(i+(fix_d1-fix_d2):i+(fix_d1-fix_d2)+training_length-1);
    for j = i:l2
        x2 = x(j+(fix_d1-fix_d2):j+(fix_d1-fix_d2)+training_length-1);
        index2 = index2 + 1;
        x2_fact(index2,:) = x1.*x2;
    end  
end

index3 = 0;
for i = 1:l3
    x11 = x(i+(fix_d1-fix_d3):i+(fix_d1-fix_d3)+training_length-1);
    for j = i:l3
        x22 = x(j+(fix_d1-fix_d3):j+(fix_d1-fix_d3)+training_length-1);
        for z = j:l3
            x33 = x(z+(fix_d1-fix_d3):z+(fix_d1-fix_d3)+training_length-1);
            index3 = index3 + 1;
            x3_fact(index3,:) = x11.*x22.*x33;
        end
    end  
end

%滤波器的最终输入
x_fact = [x1_fact;x2_fact;x3_fact];


v = 100;%阻尼因子 10、100、1000
Rvv = (x*x')/length(d);
max_step_len = 2/((l1+l2+l3)*Rvv);
%初始化
w = zeros(w1_length+w2_length+w3_length,training_length);

for i=1:training_length 
    y(i)=w(:,i)'*x_fact(:,i);  
    e(i)=d(i)-y(i);   
    if i == 1
        ee = 0;
    else
        ee = abs(e(i)*e(i-1));
    end
    step_len = 0.5*max_step_len*(1 - 1/(1 + exp(v * ee)));
%     step_len = 0.2*max_step_len;
    u = diag([step_len*ones(1,w1_length) step_len*step_len*ones(1,w2_length) step_len*step_len*ones(1,w3_length)]);
    w(:,i+1)=w(:,i)+u*e(i)*x_fact(:,i);
end


end

