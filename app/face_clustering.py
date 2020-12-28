#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from tqdm import tqdm

import networkx
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_distances

class ChineseWhisperClustering(networkx.Graph):
    """
    继承networkx.Graph类, 主要实现动态聚类
    """
    def __init__(self, nrof_iter=20, **attrs_kw):
        super(ChineseWhisperClustering, self).__init__(None, **attrs_kw)
        self.nrof_iter = nrof_iter
        self.__cluster_id_tag = '__cluster_id'
        self._id_tag = 'id'

    def add_node(self, node, **attrs_kw):
        attrs_kw.update({self.__cluster_id_tag: node})
        super().add_node(node, **attrs_kw)

    def add_edge(self, u_node, v_node, weight):
        return super().add_edge(u_node, v_node, weight=weight)

    def update_cluster(self, nrof_iter=None, node_li=None, seed=814):
        """
        迭代更新节点类标签
        :param nrof_iter:
        :param node_li:
        :return:
        """
        nrof_iter = self.nrof_iter if nrof_iter is None else nrof_iter
        update_nodes_info = {}
        np.random.seed(seed)
        nrof_node = len(self.nodes)

        for i in tqdm(range(nrof_iter), desc='Clustering feature: '):
            if node_li is None:
                node_li = [
                    node for node in self.nodes()
                    if len(list(self.neighbors(node)))
                ]
            if len(node_li) == 0:
                break
            np.random.shuffle(node_li)
            counter = 0
            for cur_node in node_li:
                cur_cluster_id = self.nodes[cur_node][self.__cluster_id_tag]
                neighbors = self.neighbors(cur_node)
                cluster_weight_sum = defaultdict(float)
                for ne in neighbors:
                    cluster_weight_sum[self.nodes[ne][
                        self.__cluster_id_tag]] += self.adj[cur_node][ne]['weight']
                if cluster_weight_sum:
                    st_cluster_weights = sorted(
                        cluster_weight_sum.items(), key=lambda x: x[1])
                    winning_group_id, edge_weight_sum = st_cluster_weights[-1]

    def _get_clusters(self):
        clusters_dic = defaultdict(list)
        for node_id in self.nodes:
            cluster_id = self.nodes[node_id][self.__cluster_id_tag]
            clusters_dic[cluster_id].append(node_id)
        return clusters_dic

    def get_clusters(self, num_per_cluster=1, attr_li=None):
        """
        获取聚类结果
        :param attr_li:
        :return:
        """
        clusters_dic = self._get_clusters()
        clusters_dic = {
            cid: [
                self.get_node_attrs(node, attr_li, add_id_tag=True)
                for node in node_li
            ]
            for cid, node_li in clusters_dic.items()
            if len(node_li) >= num_per_cluster
        }
        return clusters_dic

    def get_node_attrs(self, node_id, node_attr_li, add_id_tag=False):
        res = {}
        if add_id_tag:
            res[self._id_tag] = node_id
        if node_id in self and node_attr_li is not None:
            for k in node_attr_li:
                if k not in res and k in self.nodes[node_id]:
                    res[k] = self.nodes[node_id][k]
        return res

    @property
    def cluster_id_tag(self):
        return self.__cluster_id_tag

    @property
    def id_tag(self):
        return self._id_tag


class FaceClusteringAlgo(object):
    """
    基于Chinese Whisper Clustering的动态人脸分堆算法
    """

    def __init__(self, threshold=0.45, nrof_iter=10, debug=False):
        """
        创建分堆算法对象
        :param threshold: 人脸相似度阈值, 默认0.45
        :param nrof_iter: 迭代次数, 默认为10
        :param debug: 调试模式
        """
        super(FaceClusteringAlgo, self).__init__()
        self._threshold = threshold
        self.debug = debug
        self.emb_size = 512
        self._graph = ChineseWhisperClustering(nrof_iter)

    def process(self, faceinfo_li):
        nrof_face = len(faceinfo_li)
        print('Build graph ....')
        feats = np.vstack([f.feature.reshape(-1, self.emb_size) for f in faceinfo_li])
        distances = cosine_distances(feats)

        for i in range(nrof_face):
            face_info = faceinfo_li[i]
            self._graph.add_node(i, face_id=face_info.face_id)
            for j in range(i, nrof_face):
                weight = distances[i][j]
                self._graph.add_edge(i, j, weight)
        # iter to do clustering
        self._graph.update_cluster()
        clusters = self._graph.get_clusters(attr_li=['face_id'])
        results = {}
        for cluster_id, node_li in clusters.items():
            for node in node_li:
                face_id = node['face_id']
                results[face_id] = cluster_id
        return results

