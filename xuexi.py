# 训练过程示例：简单线性回归
import numpy as np
import matplotlib.pyplot as plt
# 生成训练数据
np.random.seed(42)
X = np.random.rand(50, 1) * 10
y = 3 * X + 2 + np.random.randn(50, 1) * 2
# 初始化模型参数
w, b = 0.0, 0.0
learning_rate = 0.01
epochs = 100
# 记录训练过程
loss_history = []
# 训练循环
for epoch in range(epochs):
    # 前向传播
    y_pred = w * X + b
    # 计算损失（均方误差）
    loss = np.mean((y_pred - y) ** 2)
    loss_history.append(loss)
    # 计算梯度
    dw = np.mean(2 * X * (y_pred - y))
    db = np.mean(2 * (y_pred - y))
    # 更新参数
    w -= learning_rate * dw
    b -= learning_rate * db
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")
# 可视化训练过程
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失变化')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', label='训练数据')
plt.plot(X, w * X + b, color='red', label='训练后的模型')
plt.xlabel('X')
plt.ylabel('y')
plt.title('训练结果')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print(f"最终模型参数：w = {w:.4f}, b = {b:.4f}")