#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tqdm import tqdm

import networkx
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import DBSCAN


def cal_distance(embeddings1, embeddings2, distance_metric=1):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        min_dist = np.min(similarity)
        dist = 1 - similarity
        # dist = np.arccos(similarity) / np.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric
    return dist


class ChineseWhisperClustering(networkx.Graph):
    """
    ChineseWhisperClustering 继承networkx.Graph类, 主要实现无监督聚类
    """
    def __init__(self, nrof_iter=20, **attrs_kw):
        """
        init

        Args:
            nrof_iter (int, optional): [description]. Defaults to 20.
        """
        super(ChineseWhisperClustering, self).__init__(None, **attrs_kw)
        self.nrof_iter = nrof_iter
        self.__cluster_id_tag = '__cluster_id'
        self._id_tag = 'id'
        self._stop_thresh = 0.01

    def add_node(self, node, **attrs_kw):
        """
        add_node 

        Args:
            node ([type]): [description]
        """
        attrs_kw.update({self.__cluster_id_tag: node})
        super().add_node(node, **attrs_kw)

    def add_edge(self, u_node, v_node, weight):
        """
        add_edge 

        Args:
            u_node ([type]): [description]
            v_node ([type]): [description]
            weight ([type]): [description]

        Returns:
            [type]: [description]
        """
        return super().add_edge(u_node, v_node, weight=weight)

    def update_cluster(self, nrof_iter=None, node_li=None, seed=814):
        """
        update_cluster 迭代更新节点类标签

        Args:
            nrof_iter ([type], optional): [description]. Defaults to None.
            node_li ([type], optional): [description]. Defaults to None.
            seed (int, optional): [description]. Defaults to 814.
        """
        # print(self)
        nrof_iter = self.nrof_iter if nrof_iter is None else nrof_iter
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
                    # update current node
                    if winning_group_id != cur_cluster_id:
                        self.nodes[cur_node][self.__cluster_id_tag] = winning_group_id
                        counter += 1

            thresh = 1 if counter == 0 else counter / float(nrof_node)
            if thresh < self._stop_thresh:
                print(f'Update number: {counter}/{nrof_node} ratio: {thresh} < {self._stop_thresh}')
                break

    def _get_clusters(self):
        clusters_dic = defaultdict(list)
        for node_id in self.nodes:
            cluster_id = self.nodes[node_id][self.__cluster_id_tag]
            clusters_dic[cluster_id].append(node_id)
        return clusters_dic

    def get_clusters(self, num_per_cluster=1, attr_li=None):
        """
        get_clusters 获取聚类结果

        Args:
            num_per_cluster (int, optional): [description]. Defaults to 1.
            attr_li ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        clusters_dic = self._get_clusters()
        clusters_dic = {
            cid: [
                self.get_node_attrs(node, attr_li, add_id_tag=True)
                for node in node_li
            ]
            for cid, node_li in clusters_dic.items()
            if len(node_li) > num_per_cluster
        }
        return clusters_dic

    def get_node_attrs(self, node_id, node_attr_li, add_id_tag=False):
        """
        get_node_attrs 

        Args:
            node_id ([type]): [description]
            node_attr_li ([type]): [description]
            add_id_tag (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
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

    def __str__(self):
        s = ''
        for cur_node in self.nodes:
            cluster_id = self.nodes[cur_node][self.__cluster_id_tag]
            neighbors = self.neighbors(cur_node)
            cur_s = 'Node={} cluster_id={} \nneighbors=[\n'.format(cur_node, cluster_id)
            for ne in neighbors:
               weight = self.adj[cur_node][ne]['weight']
               ne_cluster_id = self.nodes[ne][self.__cluster_id_tag]
               cur_s += 'Node={} cluster_id={} weight={:.3f}\n'.format(ne, ne_cluster_id, weight)
            s += cur_s + '\n]\n'
            s += 25 * '-*-' + '\n'
        return s
    
    def __repr__(self):
        return self.__str__()


class FaceClusteringAlgo(object):
    """
    基于Chinese Whispers的无监督聚类算法
    """

    def __init__(self, threshold=0.65, nrof_iter=10, debug=False):
        """
        创建分堆算法对象
        Args:
            threshold (float, optional): 人脸相似度阈值, cosine 距离.
            nrof_iter (int, optional): 迭代次数.
            debug (bool, optional): debug.
        """
        super(FaceClusteringAlgo, self).__init__()
        self._threshold = threshold
        self.debug = debug
        self.emb_size = 512
        self._graph = ChineseWhisperClustering(nrof_iter)

    def process(self, faceinfo_li):
        """
        主处理函数

        Args:
            faceinfo_li ([type]): [description]

        Returns:
            [type]: [description]
        """
        nrof_face = len(faceinfo_li)
        print('Build graph ....')
        feats = np.vstack([f.feature.reshape(-1, self.emb_size) for f in faceinfo_li])
        # distances = cosine_distances(feats)


        for i in range(nrof_face):
            face_info = faceinfo_li[i]
            self._graph.add_node(i, face_id=face_info.face_id)
            for j in range(i+1, nrof_face):
                # weight = distances[i][j]
                weight = cal_distance(feats[i].reshape(1, -1), feats[j].reshape(1, -1))[0]
                # print(f'({i}, {j})={weight}')
                if weight <= self._threshold:
                    self._graph.add_edge(i, j, weight)
                    self._graph.add_edge(j, i, weight)

        # iter to do clustering
        self._graph.update_cluster()
        clusters = self._graph.get_clusters(attr_li=['face_id'])
        results = {}
        for cluster_id, node_li in clusters.items():
            for node in node_li:
                face_id = node['face_id']
                results[face_id] = cluster_id
        return results


class FaceClusteringDBSCAN(object):
    def __init__(self, emb_size=512):
        super(FaceClusteringDBSCAN, self).__init__()
        self.emb_size = emb_size
        self.classifier = DBSCAN(metric="cosine", n_jobs=10, min_samples=2)
    
    def process(self, faceinfo_li):
        """
        process [summary]

        Args:
            faceinfo_li ([type]): [description]

        Returns:
            [type]: [description]
        """
        embeddings = np.vstack([f.feature.reshape(-1, self.emb_size) for f in faceinfo_li])
        self.classifier.fit(embeddings)
        pred_labels = self.classifier.labels_
        results = {faceinfo_li[i].face_id : pred_labels[i] for i in range(len(faceinfo_li))}
        return results
