import numpy as np

from scipy.stats import pearsonr
from PIL import Image, ImageDraw


class Node:
    """分级聚类中的树节点"""

    def __init__(self, left=None, right=None, vec=None, nid=None, distance=0.) -> None:
        # 左子节点
        self.left = left
        # 右子节点
        self.right = right
        # 数据向量, 若为叶子节点则对应真实数据，若为枝干则对应子节点平均数据
        self.vec = vec
        # ID, 若为正则为真实数据节点, 也即叶节点, 否则则为枝干节点
        self.nid = nid
        # 子节点间距离
        self.distance = distance
        # 下属多少个真实数据节点
        self.elements_size = 1 if (self.left is None and self.right is None) \
            else (self.left.elements_size if self.left is not None else 0) + \
                 (self.right.elements_size if self.right is not None else 0)


def read_file(filename):
    """从文件中读取数据"""
    with open(filename) as f:
        col_names = f.readline().split('\t')[1:]
        row_names, data = [], []
        for line in f:
            split = line.split('\t')
            row_names.append(split[0])
            data.append([float(count) for count in split[1:]])
    return row_names, col_names, np.array(data).T


def pearson(v1, v2):
    """利用皮尔逊相关系数计算向量距离"""
    return 1 - pearsonr(v1, v2)[0]


def hierarchical_classify(data, calc_distance=pearson):
    """分级聚类, 结果为树状结构"""

    # 缓存已得距离信息以备下次循环使用
    distance_cache = {}
    # 生成节点ID为负且依次减小
    new_node_id = -1
    _, cols = data.shape
    # 数据矩阵的每一列对应一个真实数据, 可以生成一个叶节点, 所有叶节点构成初始cluster列表
    cluster = [Node(vec=data[:, i], nid=i) for i in range(cols)]

    while len(cluster) > 1:
        # 每次寻找出最近的两个节点并合并, 直到cluster列表中仅剩一个根节点

        closest_indexes = (0, 1)
        closest_distance = calc_distance(cluster[0].vec, cluster[1].vec)

        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                current_nid_pair = (cluster[i].nid, cluster[j].nid)
                if current_nid_pair not in distance_cache:
                    distance_cache[current_nid_pair] = calc_distance(cluster[i].vec, cluster[j].vec)
                distance = distance_cache[current_nid_pair]
                if distance < closest_distance:
                    closest_indexes = (i, j)
                    closest_distance = distance

        left, right = cluster[closest_indexes[0]], cluster[closest_indexes[1]]
        merge_vec = (left.vec + right.vec) / 2
        merge_node = Node(left=left, right=right, vec=merge_vec, nid=new_node_id, distance=closest_distance)

        cluster.remove(left)
        cluster.remove(right)
        cluster.append(merge_node)
        new_node_id -= 1

    return cluster[0]


def print_cluster(node, convert=None, level=0):
    """在命令行中输出分级聚类结果树"""
    if node is not None:
        content = '-' if node.nid < 0 else (convert(node.nid) if convert is not None else node.nid)
        print('%s%s' % (' ' * level, content))

        print_cluster(node.left, convert, level + 1)
        print_cluster(node.right, convert, level + 1)


def get_distance_depth(node):
    """计算子节点累计距离，距离用以在画图中实现距离越远的数据线段越长"""
    return node.distance + max((get_distance_depth(node.left) if node.left is not None else 0),
                               (get_distance_depth(node.right) if node.right is not None else 0))


def draw_node(draw, node, x, y, scaling, convert=None):
    """递归绘制分级聚类结果树节点"""
    if node.nid < 0:
        half = node.elements_size * 10
        top = y - half + node.left.elements_size * 10
        bottom = y + half - node.right.elements_size * 10
        print(top, bottom)
        width = node.distance * scaling
        horizontal_offset = x + width
        color = (255, 0, 0)

        draw.line((x, top, x, bottom), fill=color)
        draw.line((x, top, horizontal_offset, top), fill=color)
        draw.line((x, bottom, horizontal_offset, bottom), fill=color)

        draw_node(draw, node.left, horizontal_offset, top, scaling, convert=convert)
        draw_node(draw, node.right, horizontal_offset, bottom, scaling, convert=convert)
    else:
        draw.text((x + 5, y - 7), convert(node.nid) if convert is not None else node.nid, (0, 0, 0))


def plot_tree(node, width=1200, convert=None, filename='clusters.jpg'):
    """绘制分级聚类结果树图像并保存"""
    height = len(titles) * 20
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    centre = height / 2
    draw.line((0, centre, 10, centre), fill=(255, 0, 0))
    scaling = (width - 150) / get_distance_depth(node)
    draw_node(draw, node, 10, centre, scaling, convert=convert)
    img.save(filename, 'JPEG')


if __name__ == '__main__':
    titles, words, counts = read_file('blogdata.txt')
    result = hierarchical_classify(counts)

    # 命令行中输出聚类结果
    # print_cluster(classify(counts), convert=lambda x: titles[x])

    # 通过树状图输出聚类结果
    plot_tree(result, convert=lambda x: titles[x])
