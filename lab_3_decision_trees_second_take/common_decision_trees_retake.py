import os
from collections import deque

from IPython.display import display
from graphviz import Digraph
import pandas as pd

import decision_trees_utils as utils

os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"


def get_training_data_set():
    attributesName = ["attr_1", "attr_2", "attr_3", "attr_4", "attr_5", "attr_6"]
    data = pd.DataFrame([[0, 0, 1, 1, 0, 1, 0],
                         [1, 1, 1, 1, 0, 1, 0],
                         [0, 1, 1, 0, 1, 0, 1],
                         [1, 1, 0, 0, 0, 0, 1],
                         [0, 1, 1, 1, 0, 1, 1],
                         [1, 1, 0, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1, 1, 0],
                         [0, 1, 0, 0, 0, 0, 1],
                         [1, 0, 1, 0, 1, 1, 1],
                         [1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 0, 1, 1],
                         [1, 1, 1, 0, 1, 1, 0],
                         [1, 1, 0, 1, 1, 1, 1],
                         [0, 1, 0, 0, 0, 0, 0],
                         [1, 1, 0, 1, 1, 0, 0],
                         [0, 1, 1, 0, 1, 1, 0],
                         [1, 0, 1, 0, 1, 1, 1],
                         [1, 1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 1, 0, 1, 0]], columns=attributesName + ["cl"])

    return attributesName, data


def get_validation_data_set():
    attribute_name_list = ["attr_1", "attr_2", "attr_3", "attr_4", "attr_5", "attr_6"]
    data = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 1, 1, 1, 1],
                         [0, 1, 1, 0, 0, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [1, 0, 0, 1, 1, 1, 1],
                         [1, 1, 1, 1, 0, 1, 1],
                         [1, 1, 0, 1, 0, 1, 1],
                         [1, 1, 0, 0, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 1]], columns=attribute_name_list + ["cl"])

    return attribute_name_list, data


class Node:
    def __init__(self, attr, left, right, value):
        self.attr = attr
        self.left = left
        self.right = right
        self.value = value

    def __call__(self, series: pd.Series):
        if self.value is None:
            if series[self.attr] == 0:
                return self.left(series)
            else:
                return self.right(series)
        else:
            return self.value


def get_error_rate(root, data: pd.DataFrame):
    correct = sum(root(series) == series["cl"] for i, series in data.iterrows())
    return 1 - correct / len(data)


def addNode(dgraph, node, data):
    if data is None:
        stats = ""
    else:
        cl_zero_count = len(data[data['cl'] == 0])
        cl_one_count = len(data[data['cl'] == 1])
        stats = f"\ncl 0: {cl_zero_count}\ncl 1: {cl_one_count}"

    node_name = str(node.nodeId)
    if node.value is None:
        node_label = f"node id: {node_name}\n{node.attr} {stats}"
        dgraph.node(node_name, label=node_label, fillcolor="yellow", style="filled")

        left_node = addNode(dgraph, node.left, None if data is None else data[data[node.attr] == 0])
        right_node = addNode(dgraph, node.right, None if data is None else data[data[node.attr] == 1])

        dgraph.edge(node_name, left_node, label="0")
        dgraph.edge(node_name, right_node, label="1")

    else:

        node_label = f"node id: {node_name}\ncl = {node.value} {stats}"
        color = "green" if node.value == 0 else "red"
        dgraph.node(node_name, label=node_label, fillcolor=color, style="filled")
    return node_name


def add_id(root: Node):
    nodes = deque([root])
    node_id = 0
    while nodes:
        node = nodes.popleft()
        node.nodeId = node_id
        node_id += 1
        if node.value is None:
            nodes.append(node.left)
            nodes.append(node.right)


def add_children_to_leaf(leaf: Node, attribute_to_split: str) -> None:
    leaf.attr = attribute_to_split
    leaf.left = Node(None, None, None, 0)
    leaf.right = Node(None, None, None, 1)
    leaf.value = None


def build_tree(data: pd.DataFrame, depth=1, max_depth=10) -> Node:
    node = Node(None, None, None, None)

    attribute_to_split = utils.choose_attribute_to_split(data)
    if depth <= max_depth and attribute_to_split:
        node.attr = attribute_to_split
        left_data, right_data = utils.split_data_by_the_best_attribute(data)
        node.left = build_tree(left_data, depth=depth + 1, max_depth=max_depth)
        node.right = build_tree(right_data, depth=depth + 1, max_depth=max_depth)
    else:

        most_common = max(utils.get_percentages_of_occurrence(data["cl"]).items(), key=lambda x: x[1])[0]

        node.value = most_common
    return node


def print_graph(root: Node, data=None, size=10, filename="DecisionTree"):
    dgraph = Digraph(format="png", filename=filename)
    dgraph.attr(size=f"{size},{size}")
    dgraph.node_attr.update()
    add_id(root)
    addNode(dgraph, root, data)
    display(dgraph)
