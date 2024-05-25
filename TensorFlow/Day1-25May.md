# Day1 

## 1 AI & machine Leaning & Deep learning & Nerual network 的概念区别

- AI 最大，包含后面概念：
1. 也许人们并不这么认为，但Tic tac toe, Pac Man 吃豆人都是属于AI的范畴

- Mahine Learning: 


- Deep Learning：
是机器学习的一个重要领域，特别关注使用神经网络
相较于让程序员来指定 “游戏规则”，DL 只会给出 input 和 output 的数据，其中的规律由机器自己来学习发现
![alt text](./图片库/image.png)

- Neural network神经网络:
是深度学习的核心板块
1. 相比上面，是最小的概念，是某种特殊的机器学习
2. 定义： it is a form of machine learning that use the layed representation of data

数据集：
会有 输入值（ feature ） 输出值 （ label ）

测试集：
只有 feature 没有 label
使用 feature 来预测 label 的值

## 2 ML 机器学习的种类；

### 2.1. Unsupervised Learning 无监督学习 
数据 只有 feature ，没有没有对应的 label 

例子：
聚类问题。 平面上有很多点，哪些点可以归为一类呢？
把 所有点的 x、y坐标传递给模型，然后用模型来决定 哪些点是一类

### 2.2. Supervised Learning 监督学习
模型会先根据 feature 给出预测
监督者会比较 预测值和实际值的偏差，对模型做出指导和调整（Tweak）

例子：
预测期末成绩是 90分
但实际的成绩的 80分
那么此时监督者会指示模型，你的预测结果比实际数据大
模型再根据你的信息来慢慢调整

### 2.3. Reinforcement Learning 加强学习
连数据都没有。需要自己去创造
模型会有：
 1. agent 中介
 2. environment 环境
 3. reward 奖励
在环境中，由中介去随机地探索，但是越符合预期的，奖励就会越大，中介会去倾向于尝试奖励值高的方法
但是一般比较依赖于 环境，如果环境变化，比较难适应。

例子： 训练 AI 来玩游戏

## 3 TensorFlow 相关概念

### 3.1 什么是 TensorFlow 
tensorflow 是一个由谷歌开发管理的拥有的开源的machine learning platform， 它是最大的machine learning library之一
可以和 Google Colab 配合食用

主要可以拿来干这些事情：
- 图像分类
- 数据聚类
- 回归分析
- 加强学习
- 自然语言处理


### 3.2 工作原理
它主要依靠两个东西工作

- Graphs 影像
    TensorFlow works by building graphs of predefined computation.
    我们在写 tensorflow 代码的过程，其实就是在建立 graphs。 但是 graphs 只代表了计算的方式，
    给出了公式。但并不真的进行了运算，也不会存储任何数据

如： s1 = s2 + s3
只是说明了s1 等于 s2 和 s3的和，并没有真的把s1算出来，也不会存储s1的数据

- Session 会话
    Session 会使得part of graphs executed，它会分配资源，内存来处理 graphs中的指令

### 3.3 导入tensorflow

1. 如果是使用 notebook （ 比如 google colab ）
```python
%tensorflow_version 2.x # 如果是其他 IDE 就不用

# 这是一个代码示例
```
2. 如果使用其他开发环境
先用 pip 来下载 tensorflow 包
（ 如果你有 cuda 的 gpu，还有专门版本的tensorflow 供你选择下载）


### 3.4 Tensor（张量） 的概念
**a tensor is a generalization of vector and metrics to potential higher dimension.**
张量是向量和矩阵向更高维度的拓展
> 标量是 一种 0 维张量 1
> 
> 向量是 一种 1 维张量 [1,2,3]
> 
> 矩阵是 一种 2 维张量 [ [1,2,3], [4,5,6] ]

Tensor 是tensorflow的主要操作对象，它会被传递，运算等等。

Each tensor represent a partially defined computation that will eventually produce a value

**每个tensor有两个重要特征**
1. data types included： float32，int32， string and other
2. Shape： 代表了数据的维度 ， represent the dimension of data


## 4 如何创建 张量 tensor
```python
string = tf.Variable("This is a string", tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.234, tf.float64)
# 前面是数据，后面是数据类型
```
### 4.1 Rank/Degree of Tensors 张量的秩
张量的秩就是其中的维度数量
上面的几个 张量，就是 rank 0， 秩为0，它们是标量 scalar

现在我们试一试更高秩的张量
```python
rank1_tensor = tf.Variable(["test"], tf.string)
rank2_tensor = tf.Variable([["test","ok"],["test","yes"]], tf.string)
# 一定是矩形的，比如[[1,2], [3,4], [1]] 就会报错。第三个也应该有两个元素才对
```

可以调用函数，来得知一个张量的秩
```python
tf.rank(rank2_tensor)
# 这个函数就会返回张量的信息： shape， dtype，rank
```

### 4.2 Shape of tensor 张量的形状
张量的形状，指的是张量每个维度里面元素的个数
```python

rank2_tensor.shape
"""
返回结果
>>> TensorShape(2,2)
第一个2，表示这里有两个 row
第二个2，表示这里每一个 row 里面有 2个元素
"""
# 有一说一，其实上面使用tf.rank（）函数的时候已经返回过它的shape了

```

### 4.3 Changing of shape 改变张量的形状
一个张量，其内部的元素个数就是它形状的数字的乘积。 同样的 元素个数，可以有几种不同的形状。
我们可以把张量做相应的形状改变，而保留元素的个数不变
```python
tensor1 = tf.ones([1,2,3]) # 这个函数ones（[1,2,3]）是指创造一个形状为[1,2,3] 的张量， 且每个元素都是 1。和 Matlab 里面的一样
tensor2 = tf.reshape(tensor1,[2,3,1]) # 把张量整型为形状 [2,3,1]
tensor3 = tf.reshape(tensor2,[3,-1]) # -1 的意思是让程序自己算该是多少合适， 这里 共 6个， 除以前面的 3，得2. 所有应该是[3,2]

print(tensor1)
print(tensor2)
print(tensor3)
"""
>>> tf.Tensor(
    [[[1. 1. 1.]
      [1. 1. 1.]]], shape=(1,2,3), dtype=float32 )

>>> tf.Tensor(
    [[[1.]
      [1.]
      [1.]]
      
      [[1.]
       [1.]
       [1.]]], shape(2,3,1), dtype=float32 )
"""
```

### 4.4 Typs of Tensors 张量的种类
最常用的 tensor 种类有这些：
- Varible
- Constant
- Placeholder
- SparseTensor 

以上这些，除了Varibale 其他的tensor都是immutable，即它们的值在运行时不会变化

### 4.5 Evaluate Tensors 计算tensor

```python
with tf.Session() as sess: # create a session with default graph
    tensor.eval() # tensor will of course be the name of your tensor
```

#### Reference:
1. tensorflow 官方文档
2. FreeCode camp 
