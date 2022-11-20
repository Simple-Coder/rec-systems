import numpy as np


# CN（common neighbors）：仅需计算两个集合的交集长度
def CN(set1, set2):
    return len(set1 & set2)


# Jaccard 相似度：在CN的基础上除以样本间的并集
def Jaccard(set1, set2):
    return len(set1 & set2) / len(set1 | set2)


# Cos相似度：两个向量间的cos相似度 ，两个向量的内积/L2范数的乘积
def cos4Vector(v1, v2):
    return (np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# 两个集合间的cos相似度
def cos4Set(set1, set2):
    return len(set1 & set2) / (len(set1) * len(set2)) ** 0.5


# Pearson相似度：两个向量
def Pearson(v1, v2):
    v1_mean = np.mean(v1)
    v2_mean = np.mean(v2)

    return (np.dot(v1 - v1_mean, v2 - v2_mean)) / (np.linalg.norm(v1 - v1_mean) * np.linalg.norm(v2 - v2_mean))


# Pearson 相似度：先让每个向量的值都减去该向量的所有值的平均值，然后求夹角余弦值
def PearsonSimple(v1, v2):
    v1 -= np.mean(v1)
    v2 -= np.mean(v2)
    return cos4Vector(v1, v2)


if __name__ == '__main__':
    a = {1, 2, 3}
    b = {2, 3, 4}
    # cn相似度
    # print(CN(a, b))

    # Jaccard相似度
    # print(Jaccard(a, b))

    # 余弦相似度
    c = [1, 2, 3]
    d = [2, 3, 4]
    # print(cos4Vector(c, d))
    print(Pearson(c, d))
    print(PearsonSimple(c, d))
