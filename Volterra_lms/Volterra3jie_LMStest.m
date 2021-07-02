function [ y ] = Volterra3jie_LMStest(x,w,training_length,l1,l2,l3)


%一阶部分
w1_length = l1;    %权重的数量
fix_d1 = fix(w1_length/2);
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


for i=1:training_length 
    y(i)=w'*x_fact(:,i);  
end

end

