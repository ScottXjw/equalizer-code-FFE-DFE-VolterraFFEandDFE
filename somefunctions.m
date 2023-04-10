
%{
查看频谱：pwelch(x)
加噪声：awgn(x,20,'measured')

滤波：rcosdesign(cutoff_factor,fir_len,sps,'sqrt');

采样： kron(x,[1 1 1 1])
查看脉冲响应：fvtool(x,'impulse')。类似于plot
%}
