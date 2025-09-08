import numpy as np
from scipy.optimize import minimize

def svm(X, y):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    num = 0

    # YOUR CODE HERE
    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge wtih any method
    # that support constrain.
    # begin answer
    
    # 将标签y转换为1维数组
    y = y.flatten()
    
    # 构建二次规划问题的参数
    # 目标函数: min (1/2) * alpha^T * H * alpha - 1^T * alpha
    # 约束: y^T * alpha = 0, 0 <= alpha_i <= inf (但实际使用边界约束)
    
    # 计算Gram矩阵
    K = np.dot(X.T, X)
    H = np.outer(y, y) * K
    
    # 定义目标函数
    def objective(alpha):
        return 0.5 * np.dot(alpha, np.dot(H, alpha)) - np.sum(alpha)
    
    # 定义约束条件
    constraints = ({'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)})
    
    # 定义边界 (0 <= alpha_i <= inf)
    bounds = [(0, None) for _ in range(N)]
    
    # 初始值
    alpha0 = np.zeros(N)
    
    # 求解二次规划问题
    res = minimize(objective, alpha0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    # 获取最优拉格朗日乘子
    alpha = res.x
    
    # 计算权重向量 w = sum(alpha_i * y_i * x_i)
    w_vec = np.sum((alpha * y).reshape(-1, 1) * X.T, axis=0)
    
    # 寻找支持向量 (alpha_i > 1e-5)
    support_vectors = alpha > 1e-5
    num = np.sum(support_vectors)
    
    # 计算偏置项 b
    # 选择任意一个支持向量来计算 b: y_i*(w^T x_i + b) = 1
    sv_indices = np.where(support_vectors)[0]
    if len(sv_indices) > 0:
        # 使用第一个支持向量
        sv_index = sv_indices[0]
        b = y[sv_index] - np.dot(w_vec, X[:, sv_index])
    else:
        b = 0
    
    # 组合权重和偏置
    w = np.vstack((b, w_vec.reshape(-1, 1)))
    
    # end answer
    return w, num