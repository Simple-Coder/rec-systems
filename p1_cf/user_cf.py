'''
https://jiang-hs.github.io/post/2D2skmyaX/
用户/物品 | 物品a | 物品b | 物品c | 物品d
用户A     |   √  |       |   √   |
用户B     |   √  |   √   |       |   √
用户C     |   √  |   √   |   √   |   √

1.计算各个用户之间的相似度,相似度矩阵
2.根据相似度的高低找到各用户的相似用户
用户A的相似用户为用户C；
用户B的相似用户为用户C；
用户C的相似用户为用户B。
3.找到相似用户购买过而目标用户不知道的物品，计算目标用户对这样的物品感兴趣的预测值（就是预测目标用户购买的可能性），向目标用户推荐这些物品。
4.给用户生成推荐列表，将被推荐物品展示给相应的用户

'''

# 定义数据集， 也就是那个表格， 注意这里我们采用字典存放数据， 因为实际情况中数据是非常稀疏的， 很少有情况是现在这样
from math import sqrt
import numpy as np

user = ['用户A', '用户B', '用户C']
item = ['物品a', '物品b', '物品c', '物品d']
n = len(user)  # n个用户
m = len(item)  # m个物品


def loadData():
    # 建立用户-物品矩阵
    user_item_matrix = np.array([[1, 0, 1, 0],
                                 [1, 1, 0, 1],
                                 [1, 1, 1, 1]])
    print("用户-物品矩阵：")
    print(user_item_matrix)
    return user_item_matrix


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


def getSimilarMatrix(user_item):
    # 用户-用户相似度矩阵
    similarity_matrix = np.zeros((n, n))  # 相似度矩阵，默认全为0
    for i in range(n):
        for j in range(n):
            if i < j:
                similarity_matrix[i][j] = cosine(user_item[i], user_item[j])
                similarity_matrix[j][i] = similarity_matrix[i][j]
    print("得到的用户-用户相似度矩阵：")
    print(similarity_matrix)  # 打印用户-用户相似度矩阵
    return similarity_matrix


def getRecommandItems(similarity_matrix, user_item):
    # 推荐物品
    max_sim = [0, 0, 0]  # 存放每个用户的相似用户
    r_list = [[], [], []]  # 存放推荐给每个用户的物品
    # 寻找每个用户相似的用户
    for i in range(n):
        for j in range(len(similarity_matrix[i])):
            if max(similarity_matrix[i]) != 0 and similarity_matrix[i][j] == max(similarity_matrix[i]):
                max_sim[i] = user[j]  # 此时的j就是相似用户的编号
                break  # break目的：一是结束当前循环，二是当前的j后面有用
        if max_sim[i] == 0:
            continue  # 等于0，表明当前用户无相似用户，无需推荐，继续下个用户
        for x in range(m):  # m个物品需要循环m次
            if user_item[i][x] == 0 and user_item[j][x] == 1:  # 目标用户不知道，而相似用户知道
                r_list[i].append(item[x])
    return r_list


if __name__ == '__main__':
    # 1.建立用户-物品矩阵
    user_item_matrix = loadData()
    # 2.计算相似度矩阵,找到每个用户最相似的用户
    user_similar_matrix = getSimilarMatrix(user_item_matrix)
    # 3.推荐物品
    recommandItems = getRecommandItems(user_similar_matrix, user_item_matrix)
    # 打印结果
    for i in range(n):  # n个用户循环n次
        if len(recommandItems[i]) > 0:  # 当前用户有被推荐的物品
            print("向{:}推荐的物品有：".format(user[i]), end='')
            print(recommandItems[i])
        print()