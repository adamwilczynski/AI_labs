import math
from collections import defaultdict, Counter
from typing import Any

import pandas as pd


def log_2(b: float) -> float:
    if b == 0:
        return 0
    return math.log(b, 2)


def get_percentages_of_occurrence(vector: list):
    counter_list = dict(Counter(vector))
    percentages_of_occurrences = defaultdict(int)
    for element, count in counter_list.items():
        percentages_of_occurrences[element] = count / len(vector)
    return percentages_of_occurrences


def get_entropy(classifier_vector: list) -> float:
    percentage_list = get_percentages_of_occurrence(classifier_vector).values()
    return - sum(occurrence_percentage * log_2(occurrence_percentage) for occurrence_percentage in percentage_list)


def get_conditional_entropy(data: pd.DataFrame, attribute: str):
    attribute_value_list = data[attribute]
    classification_value_list = data["cl"]

    x_occurrence_percent = get_percentages_of_occurrence(attribute_value_list)

    x_y_pair = list(zip(attribute_value_list, classification_value_list))
    x_y_pair_occurrence_percent = get_percentages_of_occurrence(x_y_pair)

    sum_ = 0
    for x in attribute_value_list.unique():
        probability_of_x = x_occurrence_percent[x]
        for y in classification_value_list.unique():
            probability_of_x_and_y = x_y_pair_occurrence_percent[(x, y)]

            sum_ += probability_of_x_and_y * log_2(probability_of_x_and_y / probability_of_x)
    return -sum_


def get_information_gain(data: pd.DataFrame, attribute_name) -> float:
    return get_entropy(data["cl"]) - get_conditional_entropy(data, attribute_name)


def get_information_gain_for_attributes(data: pd.DataFrame) -> dict:
    return {a: get_information_gain(data, a) for a in data.columns if a != "cl"}


def choose_attribute_to_split(data: pd.DataFrame, assert_information_gain=True) -> str | None:
    information_gain_for_attributes = get_information_gain_for_attributes(data)

    if assert_information_gain and max(information_gain_for_attributes.values()) == 0:
        return None
    return max(information_gain_for_attributes.items(), key=lambda x: x[1])[0]


def split_data_by_the_best_attribute(data: pd.DataFrame) -> str:
    attribute_column = data[choose_attribute_to_split(data)]

    left = data[attribute_column == 0]
    right = data[attribute_column == 1]
    return left, right
