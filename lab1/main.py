import math
import pandas as pd
from anytree import Node, RenderTree
from collections import Counter


def entropy(lst: list) -> float:
    p = 0
    for x in lst:
        if x:
            p -= x * math.log2(x)
    return p


def gain(e, c_e):
    return e - c_e


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def get_attributes(data: pd.DataFrame) -> list:
    return data.keys().tolist()


def get_entropy_by_attribute(data: pd.DataFrame, attribute) -> float:
    return entropy(data[attribute].value_counts(normalize=True).tolist())


def get_conditional_entropy(data: pd.DataFrame, attribute: str, decision: str) -> float:
    occurrences = dict(Counter(data[attribute].tolist()))
    size = len(data[attribute].tolist())

    c_e = 0
    for key, value in occurrences.items():
        c_e += value / size * entropy(data[data[attribute] == key][decision].value_counts(normalize=True).tolist())

    return c_e


# WERSJA 1 - podział wieku wg podanych kategorii
#
# def age_to_category(age: int) -> str:
#     if age <= 20:
#         return 1
#     if age <= 40:
#         return 2
#     return 3
#
#
# def get_best_attribute(data: pd.DataFrame, decision: str) -> str:
#     decision_entropy = get_entropy_by_attribute(data, decision)
#     possible_attributes = get_attributes(data)
#     possible_attributes.remove(decision)
#
#     gain_values = {}
#     for attribute in possible_attributes:
#         cond_entropy = get_conditional_entropy(data, attribute, decision)
#         gain_values[attribute] = gain(decision_entropy, cond_entropy)
#
#     return max(gain_values, key=gain_values.get)


# WERSJA 2 - dynamiczny podział wieku
def get_best_attribute(data: pd.DataFrame, decision: str) -> str:
    decision_entropy = get_entropy_by_attribute(data, decision)
    possible_attributes = get_attributes(data)
    possible_attributes.remove(decision)

    threshold = None
    gain_values = {}
    for attribute in possible_attributes:
        tmp_data = data.copy()
        if attribute == 'Age':
            threshold = data['Age'].median()  # mediana jako próg
            tmp_data['Age'] = tmp_data['Age'].apply(lambda x: 0 if x < threshold else 1)
        cond_entropy = get_conditional_entropy(tmp_data, attribute, decision)
        gain_values[attribute] = gain(decision_entropy, cond_entropy)

    best_attribute = max(gain_values, key=gain_values.get)
    if best_attribute == 'Age':
        new_column_name = 'Age<' + str(threshold)
        data.loc[:, new_column_name] = 1  # wartość 1 - false (warunek age nie spełniony)
        data.loc[data['Age'] < threshold, new_column_name] = 0  # wartość 0 - true (warunek age spełniony)
        best_attribute = new_column_name

    return best_attribute


def create_leafs(data: pd.DataFrame, decision: Node, main_decision: str):
    if len(set(data[main_decision].tolist())) == 1:
        return Node(data[main_decision].tolist()[0])

    best_attrib = Node(get_best_attribute(data, main_decision))
    best_attrib_values = set(data[best_attrib.name].tolist())
    data_to_pass = data.drop(decision.name, axis=1)

    for value in best_attrib_values:
        opt = Node(value, parent=best_attrib)
        a = create_leafs(data_to_pass[data_to_pass[best_attrib.name] == value], best_attrib, main_decision)
        a.parent = opt

    return best_attrib


def create_tree(data: pd.DataFrame, decision: str):
    best_attrib = Node(get_best_attribute(data, decision))
    best_attrib_values = set(data[best_attrib.name].tolist())

    for value in best_attrib_values:
        opt = Node(value, parent=best_attrib)
        a = create_leafs(data[data[best_attrib.name] == value], best_attrib, decision)
        a.parent = opt

    for pre, fill, node in RenderTree(best_attrib):
        print("%s%s" % (pre, node.name))


def main():
    # wyłączenie warningów https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    pd.options.mode.chained_assignment = None

    data = load_data("titanic-homework.csv")
    data = data.drop("PassengerId", axis=1)
    data = data.drop("Name", axis=1)

    # WERSJA 1 - podział wieku wg podanych kategorii
    # data["Age"] = data["Age"].apply(age_to_category)

    create_tree(data, "Survived")


if __name__ == "__main__":
    main()
