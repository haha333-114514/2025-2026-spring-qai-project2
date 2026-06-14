--optimization 经典优化器COBYLA

量子电路中，全连接意味着很大的计算开销，真机上的量子电路结果非常不好，有大量的swap，不适合用在实际的量子电路中

eta过小，会选不出来，不是最佳的4个，过大，4个的解会过于稳定

多比特编码：接近连续变量，表示概率
但是，意味着计算量的上涨

增加q意味着增加eta 增加eta可帮助更快收敛

layers12 111111

layers4eta6.0 110101 六位小数没有误差

layers4eta6.0currentbest 110101 六位小数没有误差

layers4eta6.0wrong result 101101 

layers6eta0.1 110101 第五位小数不同

layers8eta0.01 111111

layers8eta0.05 101111

layers8eta0.1 111111

layers8eta1hq5 011110

layers8eta0.1hq0.25 110101 第五位小数不同

layers8eta0.5 110101 第六位小数不同

layers8eta1.0 110101 第六位小数不同

layers8eta6.0 111100

layers8eta6.0 scale 1000 第五位小数不同