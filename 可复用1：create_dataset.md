## 可复用模块：Create Dataset (PyTorch)

### **标签:** `可复用`, `Create Dataset`

### **目的:**
##### 这个模块提供了一个通用的 PyTorch `Dataset` 类模板，用于加载和整理具有可选标签的数值型数据（如 NumPy 数组）。它能够方便地与 PyTorch 的 `DataLoader` 一起使用，为模型训练和评估提供数据。
#
#### **适用场景:**
##### *   当你拥有预先处理好的、以 NumPy 数组形式存储的特征数据 `X` 时。
##### *   当你的数据可能有对应的标签 `y` 时（例如分类任务）。
##### *   当你想将 NumPy 数据轻松转换为 PyTorch Tensor 以便模型使用时。
##### *   适用于特征数据是数值型，且标签是整数 ID 的情况。
#
#### **核心逻辑:**
##### 1.  **初始化 (`__init__`)**: 接收 NumPy 数组 `X` (特征) 和可选的 NumPy 数组 `y` (标签)。将它们转换为 PyTorch Tensor (`float` 类型用于特征，`LongTensor` 类型用于标签)，并存储在类的实例中。
##### 2.  **按索引取数据 (`__getitem__`)**: 根据传入的索引 `idx`，返回对应的单个数据样本（特征）及其标签（如果存在）。
##### 3.  **返回总数 (`__len__`)**: 返回数据集中总共包含的样本数量。

```
import torch
from torch.utils.data import Dataset
import numpy as np # 引入 numpy 用于类型转换
```

### 数据集类定义
#
##### 这是核心的 `Dataset` 类。你可以直接复制这个类到你的项目中，然后根据你的具体数据进行实例化。

```
class GenericDataset(Dataset):
    """
    一个通用的 PyTorch Dataset 类，用于加载 NumPy 数组数据，支持可选的标签。

    Attributes:
        data (torch.Tensor): 加载的特征数据，转换为 float 类型 Tensor。
        label (torch.Tensor or None): 加载的标签数据，转换为 LongTensor 类型。如果初始化时未提供标签，则为 None。
    """
    def __init__(self, X, y=None):
        """
        初始化 GenericDataset。

        Args:
            X (np.ndarray): 特征数据，通常是 NumPy 数组。
            y (np.ndarray, optional): 标签数据，通常是 NumPy 数组。默认为 None。
        """
        # 将 NumPy 数组转换为 PyTorch Tensor，并确保是 float 类型（适用于大部分模型输入）
        self.data = torch.from_numpy(X).float()

        # 如果提供了标签 y
        if y is not None:
            # 确保标签是整数类型，这里转换为 NumPy 的 int 类型先，再转换到 PyTorch LongTensor
            # LongTensor 是 PyTorch 中表示整数标签的常用类型
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            # 如果没有标签，则 label 设置为 None
            self.label = None

    def __getitem__(self, idx):
        """
        获取指定索引 idx 的数据样本。

        Args:
            idx (int): 需要获取的数据样本的索引。

        Returns:
            tuple or torch.Tensor:
                如果存在标签，返回 (data_sample, label_sample)。
                如果不存在标签，仅返回 data_sample。
        """
        # 如果存在标签，则返回该索引对应的数据和标签
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            # 如果不存在标签，则只返回该索引对应的数据
            return self.data[idx]

    def __len__(self):
        """
        返回数据集中总的样本数量。

        Returns:
            int: 数据集的样本总数。
        """
        # 返回数据的长度，即样本的总数
        return len(self.data)
```

### 如何使用这个模块
##### 1. 准备数据
#
###### 假设你已经加载了训练数据 `train_data` (NumPy 数组) 和对应的训练标签 `train_labels` (NumPy 数组)。
```
# **示例数据 (实际项目中请替换为你的真实数据加载)**:

# %%
# 模拟一些 NumPy 数据
num_samples_train = 1000
num_features = 784
num_classes = 10

# 模拟训练数据 (特征)
train_features_np = np.random.rand(num_samples_train, num_features).astype(np.float32)
# 模拟训练标签 (整数 ID)
train_labels_np = np.random.randint(0, num_classes, num_samples_train)

# 模拟测试数据 (特征) - 通常测试集可能没有标签
num_samples_test = 200
test_features_np = np.random.rand(num_samples_test, num_features).astype(np.float32)

print(f"模拟训练数据形状: {train_features_np.shape}")
print(f"模拟训练标签形状: {train_labels_np.shape}")
print(f"模拟测试数据形状: {test_features_np.shape}")

```
# ### 2. 实例化 Dataset
#
# 使用上面定义的 `GenericDataset` 类来创建你的数据集对象。

```
# 创建训练数据集实例
train_dataset = GenericDataset(X=train_features_np, y=train_labels_np)

# 创建测试数据集实例 (假设测试集只有特征，没有标签)
test_dataset = GenericDataset(X=test_features_np, y=None)

print(f"\n训练数据集已创建。")
print(f"- 样本数量: {len(train_dataset)}")
print(f"- 第一个样本数据形状: {train_dataset[0][0].shape}") # [0] 访问第一个样本, [0] 访问该样本的 data
print(f"- 第一个样本标签: {train_dataset[0][1]}")      # [0] 访问第一个样本, [1] 访问该样本的 label

print(f"\n测试数据集已创建。")
print(f"- 样本数量: {len(test_dataset)}")
print(f"- 第一个样本数据形状: {test_dataset[0].shape}")  # test_dataset[0] 直接返回 data，因为没有 label
```

#### 3. 与 DataLoader 结合使用
#
###### PyTorch 的 `DataLoader` 可以方便地对 `Dataset` 进行批处理、打乱等操作。

```
from torch.utils.data import DataLoader

# 定义批次大小
batch_size = 32

# 创建训练 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 创建测试 DataLoader
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # 测试集通常不需要打乱

print(f"\nDataLoader 已创建。")
print(f"- 训练 DataLoader 批次大小: {batch_size}, 是否打乱: True")
print(f"- 测试 DataLoader 批次大小: {batch_size}, 是否打乱: False")
```

# #### 演示如何从 DataLoader 中获取一个批次数据
#
###### 在模型训练循环中，你通常会这样做：
```

# 获取一个批次的训练数据和标签
# `batch_data` 将是一个 tensor，形状类似于 (batch_size, num_features)
# `batch_labels` 将是一个 tensor，形状类似于 (batch_size,)
try:
    batch_data, batch_labels = next(iter(train_loader))
    print(f"\n从训练 DataLoader 获取一个批次:")
    print(f"- 批次数据形状: {batch_data.shape}")
    print(f"- 批次标签形状: {batch_labels.shape}")
    print(f"- 批次数据 (前 5 个): {batch_data[0][:5]}")
    print(f"- 批次标签 (前 5 个): {batch_labels[:5]}")
except StopIteration:
    print("\nDataLoader 中没有更多数据（这在你第一次运行时不应该发生）。")

# 获取一个批次的测试数据
try:
    batch_test_data = next(iter(test_loader))
    print(f"\n从测试 DataLoader 获取一个批次:")
    print(f"- 批次数据形状: {batch_test_data.shape}")
    print(f"- 批次数据 (前 5 个): {batch_test_data[0][:5]}")
except StopIteration:
    print("\nDataLoader 中没有更多数据（这在你第一次运行时不应该发生）。")
```

#### *   **`GenericDataset` 类 (代码本身)**: 这是你以后项目里**最可能直接复制粘贴**的部分。
##### *   **`__init__` 方法的核心框架**: 接收 `X` 和 `y`，并进行 PyTorch 类型的转换 (`torch.from_numpy().float()`, `torch.LongTensor()`)。
##### *   **`__getitem__` 的框架**: 根据 `self.label` 是否为 `None` 来决定返回一个还是两个值。
##### *   **`__len__` 方法**: 简单地返回 `len(self.data)`。
#
#### **你下次需要根据实际情况调整的部分是：**
#
##### 1.  **类名**: 如果你希望保持唯一性，可以修改 `GenericDataset` 的名字（比如 `MyAudioDataset`, `ImageDataset`）。
##### 2.  **数据加载方式 (`__init__` 内部)**:
#####     *   如果你的数据不是 NumPy 数组，而是文件路径列表，你需要在这里实现加载文件的逻辑（例如，使用 Pillow 加载图片，使用 `torchaudio` 加载音频）。
#####     *   如果你的数据需要更复杂的预处理（如图像增强、音频特征提取），也在此方法中完成。
##### 3.  **数据类型转换 (`__init__` 内部)**:
#####     *   `X` 的 `.float()` 可能需要改为 `.long()`, `.bool()` 等，或者不需要转换。
#####     *   `y` 的 `.astype(np.int)` 和 `torch.LongTensor()` 可能需要根据你的标签类型调整（例如，回归任务的标签可能是 `float`）。
##### 4.  **`__getitem__` 返回的内容**:
#####     *   如果你的数据是成对的（如图像+文本），`__getitem__` 需要返回这对数据。
#####     *   如果你的任务不需要标签（如无监督学习），`__getitem__` 只返回 `self.data[idx]`。
##### 5.  **`DataLoader` 的参数**:
#####     *   `batch_size` 是根据你的内存和模型大小来定的。
#####     *   `shuffle` 对于训练集通常是 `True`，对于验证集和测试集通常是 `False`。
#####     *   `num_workers` 可以调整数据加载的并行程度。



---

## 分割训练集与验证集

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
