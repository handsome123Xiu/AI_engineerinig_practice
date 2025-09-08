import numpy as np

def logistic(X, y):
    '''
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    
    # 添加偏置项
    X_b = np.vstack([np.ones((1, N)), X])  # (P+1)-by-N
    
    # 设置超参数
    learning_rate = 0.01
    num_iterations = 10000
    
    # 梯度下降
    for i in range(num_iterations):
        # 计算预测概率
        z = w.T @ X_b  # 1-by-N
        y_pred = 1 / (1 + np.exp(-z))  # sigmoid函数
        
        # 计算梯度
        gradient = X_b @ (y_pred - y).T  # (P+1)-by-1
        
        # 更新权重
        w -= learning_rate * gradient / N
        
        # 可选：每1000次迭代打印损失
        if i % 1000 == 0:
            # 计算损失（对数似然损失）
            loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
            # print(f"Iteration {i}, Loss: {loss:.4f}")
    
    # end answer
    
    return w