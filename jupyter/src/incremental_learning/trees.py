#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#
from graphviz import Digraph
import numpy as np
from math import isclose

class TreeNode:
    def __init__(self, json_node, feature_names):
        self.id = json_node['node_index']
        self.number_samples = json_node['number_samples']
        self.is_leaf = False
        self.left_child_id = json_node['left_child']
        self.right_child_id = json_node['right_child']
        self.threshold = json_node['threshold']
        self.split_feature = feature_names[json_node['split_feature']]
        self.split_gain = json_node['split_gain']

    def label(self):
        label = "id={}\nsamples={}\n{}<{:.4f}".format(self.id, self.number_samples,
                                                      self.split_feature, self.threshold)
        return label

    def __eq__(self, value):
        return self.id == value.id \
            and self.is_leaf == value.is_leaf \
            and self.left_child_id == value.left_child_id \
            and self.right_child_id == value.right_child_id \
            and isclose(self.threshold,  value.threshold, abs_tol=1e-6) \
            and self.split_feature == value.split_feature \
            and isclose(self.split_gain,  value.split_gain, abs_tol=1e-6)
    
    def __ne__(self, value):
        return not self == value
    
    def __str__(self):
        return "Node {}: is_leaf {}, left_child_id {}, right_child_id {}\nthreshold {}, split_feature {}, split_gain {}".format(
            self.id, self.is_leaf, self.left_child_id, self.right_child_id, self.threshold, self.split_feature, self.split_gain)


class Leaf:
    def __init__(self, json_node):
        self.id = json_node['node_index']
        self.number_samples = json_node['number_samples']
        self.value = json_node['leaf_value']
        self.is_leaf = True

    def label(self):
        label = "id={}\nsamples={}\nvalue={:.4f}".format(self.id, self.number_samples,
                                                         self.value)
        return label

    def __eq__(self, value):
        return self.id == value.id \
            and isclose(self.value, value.value, abs_tol=1e-6) \
            and self.is_leaf == value.is_leaf
    
    def __ne__(self, value):
        return not self == value

    def __str__(self):
        return "Leaf {}: number_samples {}, value {}".format(self.id, self.number_samples, self.value)


class Tree:
    def __init__(self, json_tree):
        self.feature_names = json_tree['feature_names']
        self.dot = Digraph()
        self.dot.attr(size='10,8')
        self.nodes = {}
        self.number_nodes = 0
        self.number_leaves = 0
        for json_node in json_tree['tree_structure']:
            if 'leaf_value' in json_node:
                leaf = Leaf(json_node)
                self.nodes[leaf.id] = leaf
                self.dot.attr('node', shape='oval')
                self.dot.node(str(leaf.id), label=leaf.label())
                self.number_leaves += 1
            else:
                node = TreeNode(json_node, self.feature_names)
                self.nodes[node.id] = node
                self.dot.attr('node', shape='box')
                self.dot.node(str(node.id), label=node.label())
                self.number_nodes += 1
        for node in self.nodes.values():
            if not node.is_leaf:
                self.dot.edge(str(node.id), str(node.left_child_id))
                self.dot.edge(str(node.id), str(node.right_child_id))
        self.traverse(self.nodes[0], 0)

    def traverse(self, node, height=0):
        node.height = height
        if not node.is_leaf:
            self.traverse(self.nodes[node.left_child_id], height+1)
            self.traverse(self.nodes[node.right_child_id], height+1)

    def max_index(self):
        return max(map(lambda x: x.id, self.nodes.values()))

    def min_gain(self):
        if self.number_nodes > 0:
            return min(map(lambda x: x.split_gain, self.nodes.values()))
        return 0

    def max_gain(self):
        if self.number_nodes > 0:
            return max(map(lambda x: x.split_gain, self.nodes.values()))
        return 0

    def total_gain(self):
        if self.number_nodes > 0:
            return sum(map(lambda x: x.split_gain, self.nodes.values()))
        return 0

    def size(self):
        return len(self.nodes)

    def depth(self):
        return max(map(lambda x: x.height, self.nodes.values()))+1

    def get_leaf_path(self, x):
        path = []
        node = self.nodes[0]
        path.append(node.id)
        while not node.is_leaf:
            if x[node.split_feature] < node.threshold:
                node = self.nodes[node.left_child_id]
            else:
                node = self.nodes[node.right_child_id]
            path.append(node.id)
        return path

    def get_leaf_id(self, x):
        return self.get_leaf_path(x)[-1]

    def get_tree_distance(self, x1, x2):
        path1 = self.get_leaf_path(x1)
        path2 = self.get_leaf_path(x2)
        return len(np.setdiff1d(path1, path2))+len(np.setdiff1d(path2, path1))

    def get_leaves_num_samples(self) -> dict:
        num_samples = {}  # np.zeros(self.number_leaves)
        for idx, node in self.nodes.items():
            if node.is_leaf:
                num_samples[node.id] = node.number_samples
        return num_samples

    def __eq__(self, other):
        if self.number_leaves != other.number_leaves \
                or self.number_nodes != other.number_nodes:
            return False
        for i in range(len(self.nodes)):
            if self.nodes[i] != other.nodes[i]:
                print("Nodes {} not equal\n{}\n{}".format(i, self.nodes[i], other.nodes[i]))
                return False
        return True


class Forest:
    def __init__(self, trained_models):
        self.trees = []
        for trained_model in trained_models:
            self.trees.append(Tree(trained_model['tree']))

    def __len__(self):
        return len(self.trees)

    def max_tree_length(self):
        return max(map(lambda x: x.number_nodes, self.trees))

    def min_gain(self):
        return min(map(lambda x: x.min_gain(), self.trees))

    def max_gain(self):
        return max(map(lambda x: x.max_gain(), self.trees))

    def total_gain(self):
        return [x.total_gain() for x in self.trees]

    def tree_sizes(self):
        return [x.size() for x in self.trees]

    def tree_depths(self):
        return [x.depth() for x in self.trees]

    def show_tree(self, tree_index=0):
        if tree_index >= len(self):
            return
        return self.trees[tree_index].dot
