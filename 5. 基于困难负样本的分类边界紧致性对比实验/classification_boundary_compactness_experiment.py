import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# 设置随机种子
np.random.seed(42)

# 生成"已知类"数据（模拟已知关系）
blob_data = make_blobs(n_samples=100, centers=[[-2, 0]], cluster_std=1.0, random_state=42)
X_known = blob_data[0]
y_known = np.zeros(len(X_known))  # label 0 表示 known

# 生成"困难负样本" —— 靠近已知类但属于 NOTA
X_hard_neg = np.random.randn(50, 2) * 0.8 + [-1, 0]  # 散布在已知类边缘
y_hard_neg = np.ones(len(X_hard_neg))  # label 1 表示 NOTA

# 合并数据
X_train = np.vstack([X_known, X_hard_neg])
y_train = np.hstack([y_known, y_hard_neg])

# 特征标准化（非必需，但稳定训练）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# --- 实验1：只用已知类训练（松散边界）---
# 创建一个虚拟的第二类，远离已知类，用于训练边界
X_dummy = np.array([[-5, -5], [-5, 5], [5, -5], [5, 5]])  # 远离已知类的点
y_dummy = np.ones(len(X_dummy))  # 标记为1（NOTA类）

# 合并已知类和虚拟类进行训练
X_loose_train = np.vstack([X_train_scaled[y_train == 0], X_dummy])
y_loose_train = np.hstack([y_train[y_train == 0], y_dummy])

clf_loose = SGDClassifier(loss='log_loss', alpha=0.0001, max_iter=1000, random_state=42)
clf_loose.fit(X_loose_train, y_loose_train)

# --- 实验2：用全部数据（含困难负样本）训练（紧凑边界）---
clf_tight = SGDClassifier(loss='hinge', alpha=0.0001, max_iter=1000, random_state=42)
clf_tight.fit(X_train_scaled, y_train)  # full supervision

# === 可视化函数 ===
def plot_decision_boundary(clf, X, y, title, ax):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
  
    # 计算所有网格点的预测分数（用于画 contour）
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
  
    # 绘制等高线：0 处是决策边界
    ax.contour(xx, yy, Z, levels=[0], colors='black', linestyles='-', linewidths=2)
  
    # 画点
    scatter = ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Known Relation', alpha=0.7)
    scatter = ax.scatter(X[y==1, 0], X[y==1, 1], c='red',  marker='x', label='Hard Negative (NOTA)', alpha=0.7)
  
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

# 创建画布
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 图1：松散边界
plot_decision_boundary(clf_loose, X_train_scaled, y_train, "Loose Boundary (Only Known Class)", axes[0])

# 图2：紧凑边界
plot_decision_boundary(clf_tight, X_train_scaled, y_train, "Tight Boundary (With Hard Negatives)", axes[1])

plt.tight_layout()
plt.show()

# === 计算"紧凑性指标"：样本到边界的距离方差 ===
def get_distances_to_boundary(clf, X, y):
    scores = clf.decision_function(X)
    # 对于 known 类 (label 0)，我们希望 score < 0 → distance = -score
    known_mask = (y == 0)
    distances = -scores[known_mask]  # since scores < 0 for known class
    return distances

distances_loose = get_distances_to_boundary(clf_loose, X_train_scaled, y_train)
distances_tight = get_distances_to_boundary(clf_tight, X_train_scaled, y_train)

print("=== Compactness Metrics ===")
print(f"Loose Model: Mean Distance = {np.mean(distances_loose):.3f}, Variance = {np.var(distances_loose):.3f}")
print(f"Tight Model: Mean Distance = {np.mean(distances_tight):.3f}, Variance = {np.var(distances_tight):.3f}")