import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# 设置随机种子以保证可重复性
np.random.seed(42)

# ============================================
# 第一步：生成更合理的数据分布
# ============================================

# 生成"已知类"数据（模拟已知关系） - 以原点为中心的高斯分布
# 使用椭圆形分布使数据更自然
n_known = 150
angles_known = np.random.uniform(0, 2*np.pi, n_known)
radius_known = np.random.normal(0.6, 0.2, n_known)  # 半径服从正态分布
radius_known = np.abs(radius_known)  # 确保为正

X_known = np.column_stack([
    radius_known * np.cos(angles_known) * 1.2,  # x方向稍微拉长
    radius_known * np.sin(angles_known) * 0.8   # y方向稍微压扁
])
y_known = np.zeros(len(X_known))

# 生成"困难负样本" —— 围绕已知类周边形成一个密集的环状分布
# 关键：距离要适中，不要太近，给已知类留出安全空间
n_hard = 100
angles_hard = np.random.uniform(0, 2*np.pi, n_hard)
# 环状分布：距离中心1.5-2.2的位置（在已知类外围，但保持适当距离）
radius_hard = np.random.uniform(1.5, 2.2, n_hard)

X_hard_neg = np.column_stack([
    radius_hard * np.cos(angles_hard) * 1.2,
    radius_hard * np.sin(angles_hard) * 0.8
])
# 添加轻微噪声
X_hard_neg += np.random.randn(n_hard, 2) * 0.2
y_hard_neg = np.ones(len(X_hard_neg))

# 合并所有数据
X_all = np.vstack([X_known, X_hard_neg])
y_all = np.hstack([y_known, y_hard_neg])

# 特征标准化
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

# 重新分离标准化后的数据
X_known_scaled = X_all_scaled[y_all == 0]
X_hard_neg_scaled = X_all_scaled[y_all == 1]

# ============================================
# 实验1：松散边界模型
# ============================================
# 策略：已知类 + 稀疏的、随机分布的负样本
# 负样本分布不均匀，导致决策边界松散，可能切割已知类区域

# 生成稀疏的负样本：
# 1. 在右侧和上方添加一些负样本，制造不对称性
# 2. 负样本分布稀疏且不规则
n_sparse_neg = 30
X_sparse_neg_right = np.random.randn(n_sparse_neg//2, 2) * 0.5 + [1.5, 0]  # 右侧
X_sparse_neg_left = np.random.randn(n_sparse_neg//2, 2) * 0.5 + [-1.5, 0]  # 左侧

# 关键：在已知类的上下边缘也添加一些负样本，迫使边界切割
X_sparse_neg_boundary = np.random.randn(20, 2) * [0.6, 0.3] + [0, 1.2]  # 上边缘

X_sparse_neg = np.vstack([X_sparse_neg_right, X_sparse_neg_left, X_sparse_neg_boundary])

# 标准化负样本（使用相同的scaler）
X_sparse_neg_scaled = scaler.transform(X_sparse_neg)

# 松散模型的训练数据：已知类 + 稀疏不规则负样本
X_loose_train = np.vstack([X_known_scaled, X_sparse_neg_scaled])
y_loose_train = np.hstack([np.zeros(len(X_known_scaled)), np.ones(len(X_sparse_neg_scaled))])

# 使用极小的正则化参数 + class_weight平衡，产生松散边界
clf_loose = SGDClassifier(
    loss='hinge', 
    alpha=0.00001,      # 极小的正则化，允许复杂边界
    max_iter=3000, 
    random_state=42,
    fit_intercept=True,
    class_weight='balanced',  # 平衡类别权重
    learning_rate='optimal'
)
clf_loose.fit(X_loose_train, y_loose_train)

# ============================================
# 实验2：紧凑边界模型  
# ============================================
# 策略：已知类 + 密集的困难负样本一起训练
# 关键改进：使用RBF核SVM，能够形成非线性的、环绕式的决策边界
# 困难负样本环绕在已知类周围，会"推"决策边界远离已知类，
# 让已知类样本"抱团"更紧密，同时整体远离边界（更安全）

X_tight_train = np.vstack([X_known_scaled, X_hard_neg_scaled])
y_tight_train = np.hstack([np.zeros(len(X_known_scaled)), np.ones(len(X_hard_neg_scaled))])

# 使用RBF核SVM + 精心调整的参数
# 关键：让边界在蓝点外围留出足够的margin，而不是紧贴
# C较小 → 允许一些容错，边界更平滑、更外扩
# gamma较小 → RBF核影响范围更大，边界更圆滑
clf_tight = SVC(
    kernel='rbf',        # 使用RBF核实现非线性边界
    C=1.0,               # 适中的C值，允许一些容错空间
    gamma=0.5,           # 较小的gamma，让边界更平滑
    class_weight='balanced',
    random_state=42
)
clf_tight.fit(X_tight_train, y_tight_train)

# ============================================
# 可视化函数
# ============================================
def plot_decision_boundary(clf, X_known, X_neg, title, ax, neg_label='Negative'):
    """
    绘制决策边界和数据点
    
    Parameters:
    - clf: 训练好的分类器
    - X_known: 已知类数据（标准化后）
    - X_neg: 负样本数据（标准化后）
    - title: 图标题
    - ax: matplotlib轴对象
    - neg_label: 负样本的标签名称
    """
    h = .03  # 网格步长
    x_min, x_max = -3.5, 3.5
    y_min, y_max = -3, 3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 计算决策函数值
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 使用颜色填充表示分类区域 - 淡蓝色表示已知类，淡红色表示负类
    ax.contourf(xx, yy, Z, levels=[-100, 0, 100], 
               colors=['#e3f2fd', '#ffebee'], alpha=0.6)
    
    # 绘制决策边界（Z=0的等高线） - 黑色实线
    ax.contour(xx, yy, Z, levels=[0], colors='black', 
              linestyles='-', linewidths=3.5, zorder=3)
    
    # 绘制支持向量边界（Z=±1的等高线） - 灰色虚线
    ax.contour(xx, yy, Z, levels=[-1, 1], colors='gray', 
              linestyles='--', linewidths=2, alpha=0.7, zorder=2)
    
    # 绘制数据点 - 蓝色圆点表示已知类
    ax.scatter(X_known[:, 0], X_known[:, 1], 
              c='#1976d2', label='Known Relation', 
              alpha=0.7, s=50, edgecolors='navy', linewidths=0.8, zorder=4)
    
    # 绘制负样本 - 红色叉号
    ax.scatter(X_neg[:, 0], X_neg[:, 1], 
              c='#d32f2f', marker='x', label=neg_label, 
              alpha=0.8, s=80, linewidths=2.5, zorder=4)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=1)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)

# 创建画布
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

fig, axes = plt.subplots(1, 2, figsize=(17, 7))
fig.patch.set_facecolor('white')

# 图1：松散边界（显示稀疏负样本）
plot_decision_boundary(clf_loose, X_known_scaled, X_sparse_neg_scaled, 
                      "Loose Boundary (Linear SVM + Sparse Negatives)", 
                      axes[0], neg_label='Sparse Negative')

# 图2：紧凑边界（显示密集的困难负样本）
plot_decision_boundary(clf_tight, X_known_scaled, X_hard_neg_scaled, 
                      "Tight Boundary (RBF SVM + Dense Hard Negatives)", 
                      axes[1], neg_label='Hard Negative (NOTA)')

plt.tight_layout()
plt.savefig('/Users/damon/myWork/myExps/5. 基于困难负样本的分类边界紧致性对比实验/boundary_comparison.png', 
            dpi=200, bbox_inches='tight', facecolor='white')
print("\n✓ 图像已保存至: boundary_comparison.png\n")
plt.show()

# ============================================
# 计算紧凑性指标
# ============================================

def get_distances_to_boundary(clf, X_known):
    """
    计算已知类样本到决策边界的距离
    对于SVM，decision_function的值就是到决策超平面的有符号距离（未归一化）
    我们期望已知类的score < 0（在正确的一侧）
    距离的绝对值越大表示越远离边界（越安全）
    """
    scores = clf.decision_function(X_known)
    # 对于已知类，正确的预测应该是负值
    # 距离 = |score|，表示到边界的距离
    distances = np.abs(scores)
    return distances, scores

# 计算两个模型的距离指标
distances_loose, scores_loose = get_distances_to_boundary(clf_loose, X_known_scaled)
distances_tight, scores_tight = get_distances_to_boundary(clf_tight, X_known_scaled)

print("\n" + "="*70)
print("【紧凑性对比分析】")
print("="*70)

print("\n🔬 实验设计说明:")
print("   • 松散边界: 线性SVM + 稀疏负样本")
print("     → 决策线穿过已知类区域，很多蓝点靠近甚至跨越边界")
print("")
print("   • 紧凑边界: RBF核SVM + 密集困难负样本")
print("     → 决策线远离蓝点，把它们'圈得更紧'，红点(NOTA)被推到外围")
print("")
print("   • 关键指标: 平均距离更大(更安全) + 方差更小(更集中)")
print("")

print("\n1️⃣  松散边界模型（Loose Boundary）:")
print(f"   - 平均距离: {np.mean(distances_loose):.4f}")
print(f"   - 距离标准差: {np.std(distances_loose):.4f}")
print(f"   - 距离方差: {np.var(distances_loose):.4f}")
print(f"   - 最小距离: {np.min(distances_loose):.4f}")
print(f"   - 最大距离: {np.max(distances_loose):.4f}")
wrong_side_loose = np.sum(scores_loose > 0)
print(f"   - 错误侧样本数: {wrong_side_loose}/{len(scores_loose)} ({100*wrong_side_loose/len(scores_loose):.1f}%)")

print("\n2️⃣  紧凑边界模型（Tight Boundary）:")
print(f"   - 平均距离: {np.mean(distances_tight):.4f}")
print(f"   - 距离标准差: {np.std(distances_tight):.4f}")
print(f"   - 距离方差: {np.var(distances_tight):.4f}")
print(f"   - 最小距离: {np.min(distances_tight):.4f}")
print(f"   - 最大距离: {np.max(distances_tight):.4f}")
wrong_side_tight = np.sum(scores_tight > 0)
print(f"   - 错误侧样本数: {wrong_side_tight}/{len(scores_tight)} ({100*wrong_side_tight/len(scores_tight):.1f}%)")

print("\n3️⃣  对比结论:")
print(f"   - 平均距离提升: {np.mean(distances_tight) - np.mean(distances_loose):.4f}")
print(f"   - 方差降低: {np.var(distances_loose) - np.var(distances_tight):.4f}")
print(f"   - 标准差降低: {np.std(distances_loose) - np.std(distances_tight):.4f}")

if np.mean(distances_tight) > np.mean(distances_loose) and np.var(distances_tight) < np.var(distances_loose):
    print("\n✅ 紧凑边界模型成功达到目标：")
    print("   • 平均距离更大 → 边界离已知类更远，更安全")
    print("   • 方差更小 → 样本分布更集中，边界更紧凑")
else:
    print("\n⚠️  结果未达预期，可能需要调整参数")

print("\n4️⃣  模型参数对比:")
print(f"\n   松散模型（线性SVM）:")
print(f"   - 权重向量: {clf_loose.coef_[0]}")
print(f"   - 截距: {clf_loose.intercept_[0]:.4f}")
print(f"   - 权重向量模长: {np.linalg.norm(clf_loose.coef_):.4f}")

print(f"\n   紧凑模型（RBF核SVM）:")
print(f"   - 核函数: RBF (高斯核)")
print(f"   - 支持向量数量: {len(clf_tight.support_vectors_)}")
print(f"   - C参数: {clf_tight.C}")
print(f"   - Gamma: {clf_tight.gamma if isinstance(clf_tight.gamma, float) else 'scale (auto)'}")

print("\n" + "="*70)
print("💡 核心思想与实验结论：")
print("="*70)
print("\n   1. 【负样本分布的重要性】")
print("      • 稀疏、随机的负样本 → 决策边界松散，可能切割已知类区域")
print("      • 密集、环绕的困难负样本 → 决策边界紧凑，紧贴已知类外围")
print("\n   2. 【困难负样本的作用机制】")
print("      • 困难负样本像'围栏'一样环绕已知类")
print("      • 它们'推'着决策边界向内收缩")
print("      • 最终形成紧致的、安全距离更大的分类边界")
print("\n   3. 【实际应用价值】")
print("      • 在关系抽取、意图识别等任务中")
print("      • 通过挖掘困难负样本（接近已知类但属于NOTA的样本）")
print("      • 可以显著提升模型对未知类的识别能力")
print("      • 同时让已知类的边界更加紧凑和安全！")
print("\n" + "="*70 + "\n")