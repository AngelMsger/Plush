from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean

from data_source import critics


def sim_euclidean(prefs, person1, person2):
    """基于欧几里得距离，输出[0, 1]之间的值"""
    # print('In: %s, %s' % (person1, person2))
    shared_items = [item for item in prefs[person1] if item in prefs[person2]]
    if len(shared_items) == 0:
        return 0

    person1_pref = [prefs[person1][item] for item in shared_items]
    person2_pref = [prefs[person2][item] for item in shared_items]

    # print('Out: %s' % str(1 / (1 + euclidean(person1_pref, person2_pref))))
    return 1 / (1 + pow(euclidean(person1_pref, person2_pref), 2))


def sim_pearson(prefs, person1, person2):
    """基于皮尔逊相关系数，即 协方差 与 标准差乘积 的比值，输出[-1, 1]之间的值"""
    shared_items = [item for item in prefs[person1] if item in prefs[person2]]
    if len(shared_items) == 0:
        return 1

    person1_pref = [prefs[person1][item] for item in shared_items]
    person2_pref = [prefs[person2][item] for item in shared_items]

    return pearsonr(person1_pref, person2_pref)[0]


def top_matches(prefs, person, n=5, similarity=sim_pearson):
    """寻找最相似的n个人"""
    scores = [(similarity(prefs, person, other), other) for other in prefs if other != person]
    scores.sort(reverse=True)
    return scores[0:n]


def recommendations(prefs, person, similarity=sim_pearson):
    """以相似度较高的人的评价为标准，利用加权平均分给出推荐列表"""
    total, sims_sum = {}, {}

    for other in prefs:
        if other == person:
            continue

        sim = similarity(prefs, other, person)
        if sim <= 0:
            continue

        for item in prefs[other]:
            if item not in prefs[person]:
                total.setdefault(item, 0)
                total[item] += sim * prefs[other][item]
                sims_sum.setdefault(item, 0)
                sims_sum[item] += sim

    ranks = [(total[item] / sims_sum[item], item) for item in total]
    ranks.sort(reverse=True)
    return ranks


def transform_prefs(prefs):
    """倒置评价主客体"""
    result = {}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})
            result[item][person] = prefs[person][item]
    return result


def calculate_similar_items(prefs, n=10, similarity=sim_pearson):
    """计算并缓存最相近客体，客体关系稳定不易变动，因此可重复使用"""
    items_prefs = transform_prefs(prefs)

    result = {}
    for item in items_prefs:
        ranks = top_matches(items_prefs, item, n, similarity)
        result[item] = ranks
    return result


print(calculate_similar_items(critics, similarity=sim_euclidean))
# print(top_matches(critics, 'Toby', 3))

# movies = transform_prefs(critics)
# print(top_matches(movies, 'Superman Returns'))
