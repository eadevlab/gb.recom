import numpy as np


def precision(recommended_list, bought_list):
    return np.sum(np.isin(bought_list, recommended_list)) / len(recommended_list)


def precision_at_k(recommended_list, bought_list, k=5):
    return np.sum(np.isin(bought_list, recommended_list[:k])) / len(
        recommended_list[:k])


def recall(recommended_list, bought_list):
    return np.sum(np.isin(bought_list, recommended_list))/len(bought_list)


def recall_at_k(recommended_list, bought_list, k=5):
    return np.sum(np.isin(bought_list, recommended_list[:k]))/len(bought_list)
