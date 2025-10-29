## 可复用模块：Create Dataset (PyTorch)


## **1.从 NumPy 创建 PyTorch Dataset**

**一句话总结：** 一个通用模板，将内存中的 NumPy 数据 `(X, y)` 快速打包成 PyTorch 能用的 `Dataset` 和 `DataLoader`。

---

#### 1. 核心代码模板 (直接复制)

```python
import torch
from torch.utils.data import Dataset
import numpy as np

class GenericDataset(Dataset):
    """一个通用的、从 NumPy 数组创建 PyTorch 数据集的模板。"""
    def __init__(self, X, y=None):
        # 1. 将 NumPy 特征数据转换为 PyTorch Float Tensor
        self.data = torch.from_numpy(X).float()
        
        # 2. 处理可选的标签
        if y is not None:
            # 将 NumPy 标签转换为 PyTorch Long Tensor (用于分类)
            self.label = torch.LongTensor(y.astype(np.int))
        else:
            self.label = None

    def __getitem__(self, idx):
        # 3. 按索引返回数据
        if self.label is not None:
            return self.data[idx], self.label[idx] # 返回 (特征, 标签)
        else:
            return self.data[idx] # 仅返回特征

    def __len__(self):
        # 4. 返回数据集总长度
        return len(self.data)
```

#### 2. 标准使用三步走

假设你已有 NumPy 数据 `train_features_np` 和 `train_labels_np`。

```python
from torch.utils.data import DataLoader

# 步骤 1: 实例化你的 Dataset
# X 是特征, y 是标签
my_dataset = GenericDataset(X=train_features_np, y=train_labels_np)

# 步骤 2: 将 Dataset 传入 DataLoader
# batch_size 和 shuffle 是最关键的参数
my_loader = DataLoader(my_dataset, batch_size=32, shuffle=True)

# 步骤 3: 在训练循环中使用
for batch_data, batch_labels in my_loader:
    # ...送入模型进行训练...
    pass
```

---

#### 3. 下次修改的「热点区域」

当你复用此模板时，90% 的修改都集中在以下几点：

1.  **数据来源 (`__init__`)**:
    *   如果数据不是 NumPy 而是**文件路径**，在此处添加文件读取逻辑（如 `PIL.Image.open()`）。

2.  **数据类型 (`__init__`)**:
    *   特征的 `.float()` 可能需要根据模型输入改变。
    *   标签的 `torch.LongTensor` 用于分类，如果是**回归任务**，应改为 `torch.FloatTensor`。

3.  **数据增强 (`__getitem__`)**:
    *   在此处对返回的 `self.data[idx]` 应用各种数据变换 (Transforms)。

4.  **`DataLoader` 参数**:
    *   **`batch_size`**: 根据你的 GPU 显存调整。
    *   **`shuffle`**: 训练集设为 `True`，验证/测试集设为 `False`。
    *   **`num_workers`**: 设为大于 0 的整数（如 `4` 或 `8`）可以加速数据加载。



---
## **2.分割训练集与验证集**

**一句话总结：** 将一份完整的数据集，按比例（如此处的80/20）切分成 **训练集 (Training Set)** 和 **验证集 (Validation Set)**。

```python
# --- 要点代码 ---

# 1. 定义验证集所占的比例 (e.g., 20%)
VAL_RATIO = 0.2

# 2. 计算训练集的样本数量，从而找到分割点
percent = int(train.shape[0] * (1 - VAL_RATIO))

# 3. 使用数组切片，将特征(train)和标签(train_label)同时分割
#    [:percent] -> 从头到分割点 (新的训练集)
#    [percent:] -> 从分割点到结尾 (验证集)
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]

# 4. (可选但推荐) 打印形状以验证分割是否正确
print('Size of training set: {}'.format(train_x.shape))
print('Size of validation set: {}'.format(val_x.shape))
```

### 关键要点 & 可复用性

*   **目的**：训练时用 `train_x, train_y`，每训练一阵就用 `val_x, val_y` 评估一下模型效果，防止过拟合。
*   **核心技术**：Python 的**数组切片 (Slicing)**。 `array[:index]` 取前面部分，`array[index:]` 取后面部分。
*   **可复用性**：**极高**。这是划分数据集最基本、最常用的方法。
    *   **下次你需要改的**：基本只有 `VAL_RATIO` 这个比例值。
    *   **代码结构完全不用变**。
