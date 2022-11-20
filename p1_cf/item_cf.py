"""
基于物品（item）的协同过滤
基于物品的 CF 的原理和基于用户的 CF 类似，只是在计算邻居时采用物品本身，而不是从用户的角度，即基于用户对物品的偏好找到相似的物品，
然后根据用户的历史偏好，推荐相似的物品给他。计算上，就是计算物品之间的相似度，
得到物品的相似物品后，根据用户历史的偏好预测当前用户还没有表示偏好的物品，计算得到一个排序的物品列表作为推荐。

算法过程：
1。建立 物品—用户 倒排表；
2。计算各个物品之间的相似度（使用余弦定理相似性度量
3。根据物品的相似度和用户的历史行为给用户生成推荐列表
4。对于物品b，它的相似物品为物品a，但是购买过物品b的用户都购买过物品a；
对于物品c，它的相似物品为物品a，但是购买过物品c的用户也都购买过物品a。


用户/物品 | 物品a | 物品b | 物品c
用户A     |   √  |       |   √
用户B     |   √  |       |   √
用户C     |   √  |   √   |
"""
from math import sqrt
import numpy as np

user = ['用户A', '用户B', '用户C']
item = ['物品a', '物品b', '物品c']
n = len(user)  # n个用户
m = len(item)  # m个物品


# 定义余弦相似性度量计算
def cosine(ls_1, ls_2):
    Numerator = 0  # 公式中的分子
    abs_1 = abs_2 = 0  # 分母中两向量的绝对值
    for i in range(m):
        Numerator += ls_1[i] * ls_2[i]
        if ls_1[i] == 1:
            abs_1 += ls_1[i]
        if ls_2[i] == 1:
            abs_2 += ls_2[i]

    Denominator = sqrt(abs_1 * abs_2)  # 公式中的分母
    return Numerator / Denominator


# 定义预测函数
def predict(w_uv, r_vi=1):
    p = w_uv * r_vi
    return p


def loadData():
    # 建立用户-物品矩阵
    user_item_matrix = np.array([[1, 0, 1],
                                 [1, 0, 1],
                                 [1, 1, 0]])
    print("用户-物品矩阵：")
    print(user_item_matrix)
    return user_item_matrix


def getSimilarMatrix(item_user_matrix):
    # 构建物品-物品相似度矩阵
    sim = np.zeros((n, n))  # 相似度矩阵，默认全为0
    for i in range(n):
        for j in range(n):
            if i < j:
                sim[i][j] = cosine(item_user_matrix[i], item_user_matrix[j])
                sim[j][i] = sim[i][j]
    print("得到的物品-物品相似度矩阵：")
    print(sim)  # 打印物品-物品相似度矩阵
    return sim


def getRecommandItems(item_similar_matrix, item_user_matrix):
    # 推荐物品
    max_sim = [0, 0, 0]  # 存放每个物品的相似物品
    r_list = [[], [], []]  # 存放推荐给每个用户的物品
    p = [[], [], []]  # 每个用户被推荐物品的预测值列表
    for i in range(m):  # m个物品循环m次
        # 找到与物品i最相似的物品
        for j in range(len(item_similar_matrix[i])):  # range()里面写m也可以
            if max(item_similar_matrix[i]) != 0 and item_similar_matrix[i][j] == max(item_similar_matrix[i]):
                max_sim[i] = item[j]  # 此时的j就是相似物品的编号
                break  # break目的：一是结束当前循环，二是当前的j后面有用
        if max_sim[i] == 0:
            continue  # 等于0，表明当前物品无相似物品，继续下个物品
        # 找出应该推荐的物品，并计算预测值
        for x in range(n):  # n个用户循环n次
            if item_user_matrix[i][x] == 1 and item_user_matrix[j][x] == 0:  # 当前物品用户知道，而相似物品该用户不知道
                r_list[x].append(max_sim[i])
                p[x].append(predict(item_similar_matrix[i][j]))
    return r_list, p


if __name__ == '__main__':
    # 1.建立用户-物品矩阵
    user_item_matrix = loadData()
    # 2.建立物品-用户倒排表
    item_user_matrix = user_item_matrix.T
    print("物品-用户矩阵：")
    print(item_user_matrix)

    # 2.计算相似度矩阵,找到每个物品最相似的物品
    item_similar_matrix = getSimilarMatrix(item_user_matrix)

    # 3.推荐物品
    recommandItems, p = getRecommandItems(item_similar_matrix, item_user_matrix)

    # 打印结果
    for i in range(n):  # n个用户循环n次
        if len(recommandItems[i]) > 0:  # 当前用户有被推荐的物品
            print("向{:}推荐的物品有：".format(user[i]), end='')
            print(recommandItems[i])
            print("该用户对以上物品该兴趣的预测值为：", end='')
            print(p[i])
        print()
