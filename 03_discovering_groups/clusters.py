import numpy as np

from scipy.stats import pearsonr


class Node:

    def __init__(self, left=None, right=None, vec=None, nid=None, distance=0.) -> None:
        self.left = left
        self.right = right
        self.vec = vec
        self.nid = nid
        self.distance = distance


def read_file(filename):
    with open(filename) as f:
        col_names = f.readline().split('\t')[1:]
        row_names, data = [], []
        for line in f:
            split = line.split('\t')
            row_names.append(split[0])
            data.append([float(count) for count in split[1:]])
    return row_names, col_names, np.array(data).T


def pearson(v1, v2):
    return 1 - pearsonr(v1, v2)[0]


def classify(data, calc_distance=pearson):
    distance_cache = {}
    new_node_id = -1
    _, cols = data.shape
    cluster = [Node(vec=data[:, i], nid=i) for i in range(cols)]

    while len(cluster) > 1:
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
    if node is not None:
        content = '-' if node.nid < 0 else (convert(node.nid) if convert is not None else node.nid)
        print('%s%s' % (' ' * level, content))

        print_cluster(node.left, convert, level + 1)
        print_cluster(node.right, convert, level + 1)


if __name__ == '__main__':
    titles, words, counts = read_file('blogdata.txt')
    print_cluster(classify(counts), convert=lambda x: titles[x])
