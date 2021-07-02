function [ ber ] = BER03(x,y)
% 阈值在不同应用时需要调整

panjue(x >= 2.5) = 3;
panjue(x < 2.5 & x >= 1.5) = 2;
panjue(x < 1.5 & x >= 0.5) = 1;
panjue(x < 0.5) = 0;

ber = sum(y ~= panjue) / length(panjue);

end

