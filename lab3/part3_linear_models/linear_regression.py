import numpy as np

def linear_regression(X, y):
    '''
    LINEAR_REGRESSION Linear Regression.

    INPUT:  X: training样本特征, P-by-N matrix.
            y: training样本标签, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    
    # 添加偏置项 (1-by-N)
    X_b = np.vstack([np.ones((1, N)), X])  # (P+1)-by-N
    
    # 使用正规方程: w = (X_b X_b^T)^(-1) X_b y^T
    # 因为 X_b 是 (P+1)-by-N, y 是 1-by-N
    
    # 计算 X_b X_b^T
    XXT = X_b @ X_b.T  # (P+1)-by-(P+1)
    
    # 计算 X_b y^T
    Xy = X_b @ y.T  # (P+1)-by-1
    
    # 求解线性方程组: (X_b X_b^T) w = X_b y^T
    try:
        # 使用最小二乘解，更稳定
        w = np.linalg.lstsq(XXT, Xy, rcond=None)[0]
    except:
        # 如果出现数值问题，使用伪逆
        w = np.linalg.pinv(XXT) @ Xy
    
    # 确保w的形状是(P+1)-by-1
    w = w.reshape(-1, 1)
    
    # end answer
    return w
