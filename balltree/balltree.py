# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import warnings
import subprocess
from queue import PriorityQueue


warnings.simplefilter('ignore', UserWarning)
global fig_cnt

plt.rcParams['figure.figsize'] = (8, 6)
default_size = 40
centroid_size = 80


def euler_dist(p1, p2):
    """ 计算欧拉距离
    """
    p1, p2 = np.array(p1), np.array(p2)
    diff = p1 - p2
    diff = diff**2
    return np.sqrt(np.sum(diff))


def cal_radius_next_point(pc, data):
    """
    计算质心的半径以及离质心最远的点
    """
    radius = 0
    pm = None
    for p in data:
        # print(p.shape)
        p = np.expand_dims(p, axis=0)   # add dimension so p can keep with pc
        dist = euler_dist(pc, p)
        if radius < dist:
            radius = dist
            pm = p
    return radius, pm


class Point(object):

    def __init__(self, data, dist):
        self.data = data
        self.dist = dist

    def __lt__(self, other):
        return self.dist < other.dist

    '''
    def __cmp__(self, other):
        return -1 if self.dist < other.dist else 1

    def __eq__(self, other):
        return self.dist == other.dist

    def __gt__(self, other):
        return self.dist > other.dist

    def __ge__(self, other):
        return self.dist >= other.dist

    def __le__(self, other):
        return self.dist <= other.dist
    '''


class BallNode(object):

    def __init__(self, **kwargs):
        self.centroid = kwargs.get('centroid', None)
        self.radius = kwargs.get('radius', 0)
        self.is_leaf = kwargs.get('is_leaf', False)
        self.left = None
        self.right = None
        self.node_data = np.array([])

        # use to trace
        self.is_class_a = np.array([])
        self.c_meta = []


class BallTree(object):

    def __init__(self, leaf_max_node):
        self.leaf_max_node = leaf_max_node
        self.root = BallNode()

    def build(self, data):
        self._build(self.root, data)

    def _build(self, b_node, raw_data):
        """ data array contains feature vector sample
        """
        data = raw_data.copy()

        if len(data) == 0 or b_node is None:
            return

        # 如果数据没有叶子规定的最大个数
        elif len(data) < self.leaf_max_node:
            bn = b_node
            # bn = BallNode(is_leaf=True)
            bn.is_leaf = True
            bn.centroid = np.mean(data, axis=0, keepdims=True)
            bn.node_data = data
            bn.radius, _ = cal_radius_next_point(bn.centroid, data)
            b_node = bn
            return

        bn = b_node
        bn.centroid = np.mean(data, axis=0, keepdims=True)
        bn.radius, next_p = cal_radius_next_point(bn.centroid, data)
        np_radius, np_next = cal_radius_next_point(next_p, data)

        # 划分数据
        is_class_a = np.array([True for _ in range(len(data))])
        for idx, p in enumerate(data):
            if (p == next_p).all():   # a 类
                continue
            elif (p == np_next).all():
                is_class_a[idx] = False   # b 类
                continue
            dist_a = euler_dist(next_p, p)
            dist_b = euler_dist(np_next, p)
            if dist_b < dist_a:
                is_class_a[idx] = False

        data_a = data[is_class_a]
        data_b = data[~is_class_a]

        # add trace data to plot
        # traces.append((bn.centroid, bn.radius, next_p, np_next))
        bn.node_data = data
        bn.is_class_a = is_class_a
        bn.c_meta = [next_p, np_next]

        del data

        if len(data_a) > 0:
            bn.left = BallNode()
            self._build(bn.left, data_a)

        if len(data_b) > 0:
            bn.right = BallNode()
            self._build(bn.right, data_b)

    def search(self, v, k, bn, pq):
        """
        v: query vector
        k: k nearst neigboor number
        bn: balltree node
        pq: priority queue
        """
        if bn is None:
            # print("none return")
            return
        elif bn.is_leaf:
            # print("leaf return")
            for data in bn.node_data:
                dist = euler_dist(v, data)
                # print(pq.qsize())
                if pq.qsize() >= k and pq.queue[-1].dist < dist:
                    continue
                # bn.distance = dist
                p = Point(data, dist)
                pq.put(p)
                if pq.qsize() > k:
                    pq.get()
            return

        # left and right distance
        ld = euler_dist(v, bn.left.centroid)
        rd = euler_dist(v, bn.right.centroid)

        if ld < rd:
            s1, s2 = bn.left, bn.right
        else:
            s1, s2 = bn.right, bn.left
            ld, rd = rd, ld

        if pq.empty() or (ld - s1.radius) < pq.queue[-1].dist:
            # print("search tree 1")
            self.search(v, k, s1, pq)
        # else:
        #     print("search tree 1 not")

        if pq.empty() or (rd - s2.radius) < pq.queue[-1].dist:
            # print("search tree 2")
            self.search(v, k, s2, pq)
        # else:
        #     print("search tree 2 not")


def check_bt(bt):
    node = [bt.root]
    leaf_cnt = 0
    while node:
        tmp = node.pop()
        if tmp.left is not None:
            node.append(tmp.left)

        if tmp.right is not None:
            node.append(tmp.right)

        if tmp.is_leaf:
            # print("LEAF data: ", tmp.node_data)
            leaf_cnt += 1
    print("leaves number: ", leaf_cnt)


def get_leaves(bt):
    node = [bt.root]
    leaves = []
    while node:
        tmp = node.pop()
        if tmp.left is not None:
            node.append(tmp.left)

        if tmp.right is not None:
            node.append(tmp.right)

        if tmp.is_leaf:
            leaves.append(tmp)
    return leaves



def get_custom_layout():

    plt.clf()
    plt.cla()
    plt.close()
    # fig, ax = plt.subplots(figsize=(16, 12))
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    plt.grid()

    return fig, ax


def save_fig(fig, savefile):
    fig.savefig(savefile, transparent=False, dpi=160, bbox_inches='tight')
    global fig_cnt
    print("Processed %d fig" % fig_cnt)
    fig_cnt += 1


def show_or_save(fig, show, savefile):
    if show:
        plt.show()

    if savefile is not None:
        save_fig(fig, savefile)


def add_scatter(ax, data, s=default_size, facecolors='none', edgecolors='k', alpha=1, **kwargs):
    ax.scatter(data[:, 0], data[:, 1], s=s, facecolors=facecolors,
            edgecolors=edgecolors, alpha=alpha, **kwargs)


def add_circle(ax, p, r, color, alpha=.2):
    pc = plt.Circle(xy=p, radius=r, color=color, alpha=alpha)
    ax.add_artist(pc)


def add_circles(ax, ps, rs, colors, alphas):
    n = len(ps)
    if isinstance(alphas, float):
        alphas = [alphas for _ in range(n)]
    for i in range(n):
        add_circle(ax, ps[i], rs[i], colors[i], alphas[i])


def show_default_data(data, show=False, savefile=None):
    fig, ax = get_custom_layout()
    add_scatter(ax, data)
    show_or_save(fig, show, savefile)
    return fig, ax

def get_bg_fig(data, show=False, savefile=None):
    fig, ax = get_custom_layout()
    add_scatter(ax, data, s=default_size, alpha=.3)
    return fig, ax


def show_final_balls(bt, show=False, savefile=None):

    fig, ax = get_custom_layout()
    leaves = get_leaves(bt)
    colors = cm.Set1(np.linspace(0, 1, len(leaves)))
    for i, leaf in enumerate(leaves):
        c, r, d = leaf.centroid, leaf.radius, leaf.node_data
        # ax.scatter(d[:, 0], d[:, 1], s=80, facecolors=colors[i], edgecolors='none', alpha=.7)
        # ax.scatter(c[0], c[1], s=160, facecolors=colors[i], edgecolors='none')
        add_scatter(ax, d, s=default_size, facecolors=colors[i], edgecolors='none', alpha=.7)
        add_scatter(ax, c, s=centroid_size, facecolors=colors[i], edgecolors='none', alpha=.7)
        pc = plt.Circle(xy=c[0], radius=r, color=colors[i], alpha=.2)
        ax.add_artist(pc)

    show_or_save(fig, show, savefile)


def show_build_path(bt):

    global fig_cnt
    def _fn():
        if fig_cnt < 10:
            s = '00%d' % fig_cnt
        elif fig_cnt < 100:
            s = '0%d' % fig_cnt
        else:
            s = str(fig_cnt)
        return 'f%s.png' % s

    if fig_cnt == 0:
        data = bt.root.node_data
        is_class_a = bt.root.is_class_a
        data_a = data[is_class_a]
        data_b = data[~is_class_a]
        # c, r, next_p, np_next = traces[0]
        c, r = bt.root.centroid, bt.root.radius
        next_p, np_next = bt.root.c_meta

        show_default_data(data, savefile=_fn())

        fig, ax = get_bg_fig(data)
        # centroid
        add_scatter(ax, c, facecolors='b', edgecolors='none')
        save_fig(fig, _fn())

        # centroid circle
        add_circle(ax, c[0], r, 'b')
        save_fig(fig, _fn())

        fig, ax = get_bg_fig(data)
        # centroid
        add_scatter(ax, c, facecolors='b', edgecolors='none')
        save_fig(fig, _fn())

        # next point
        add_scatter(ax, next_p, facecolors='r', edgecolors='none')
        save_fig(fig, _fn())

        # next next point
        add_scatter(ax, np_next, facecolors='g', edgecolors='none')
        save_fig(fig, _fn())

        fig, ax = get_custom_layout()
        add_scatter(ax, data_a, s=default_size, facecolors='r', edgecolors='none', alpha=.3)
        add_scatter(ax, next_p, facecolors='r', edgecolors='none')
        add_scatter(ax, data_b, s=default_size, facecolors='g', edgecolors='none', alpha=.3)
        add_scatter(ax, np_next, facecolors='g', edgecolors='none')
        save_fig(fig, _fn())

    node = [bt.root]
    leaves = []
    node_trav = []

    while node:
        tmp = node.pop()
        if tmp.left is not None:
            node.append(tmp.left)

        if tmp.right is not None:
            node.append(tmp.right)

        if tmp.is_leaf:
            leaves.append(tmp)
        node_trav.append(tmp)

    node_trav = node_trav
    for i in range(1, len(node_trav)):
        tmp = node_trav[i]
        if tmp.is_leaf:
            fig, ax = get_bg_fig(bt.root.node_data)
            data = tmp.node_data
            c, r = tmp.centroid, tmp.radius
            add_scatter(ax, data, s=default_size, facecolors='b', edgecolors='b', alpha=.3)
            save_fig(fig, _fn())
            add_scatter(ax, c, s=centroid_size, facecolors='b', edgecolors='b')
            add_circle(ax, c[0], r, color='b')
            save_fig(fig, _fn())
        else:
            fig, ax = get_bg_fig(bt.root.node_data)
            data = tmp.node_data
            c, r = tmp.centroid, tmp.radius
            add_scatter(ax, data, s=default_size, facecolors='b', edgecolors='b', alpha=.3)
            save_fig(fig, _fn())
            add_scatter(ax, c, s=centroid_size, facecolors='b', edgecolors='b')
            save_fig(fig, _fn())
            add_circle(ax, c[0], r, color='b')
            save_fig(fig, _fn())

            ia = tmp.is_class_a
            data_a, data_b = data[ia], data[~ia]
            next_p, np_next = tmp.c_meta

            fig, ax = get_bg_fig(bt.root.node_data)
            add_scatter(ax, data_a, s=default_size, facecolors='r', edgecolors='none', alpha=.3)
            add_scatter(ax, next_p, facecolors='r', edgecolors='none')
            add_scatter(ax, data_b, s=default_size, facecolors='g', edgecolors='none', alpha=.3)
            add_scatter(ax, np_next, facecolors='g', edgecolors='none')
            save_fig(fig, _fn())

    show_final_balls(bt, savefile=_fn())


if __name__ == "__main__":

    np.random.seed(123)
    data = np.random.normal(size=200).reshape(100, 2)
    c = np.mean(data, axis=0)
    bt = BallTree(leaf_max_node=20)

    # traces = []
    fig_cnt = 0
    bt.build(data)

    check_bt(bt)

    k = 5
    pq = PriorityQueue(maxsize=k+1)
    # bt.search([0, 0], k, bt.root, pq)
    bt.search(c, k, bt.root, pq)

    show_build_path(bt)
    # print("check the k nn:")
    # print([e.data for e in pq.queue])
    cmd = "convert -delay 20 -loop 1 *.png bt.gif"
    subprocess.call(cmd, shell=True)

