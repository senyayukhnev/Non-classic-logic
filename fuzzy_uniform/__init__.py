"""Библиотека работы с нечёткими величинами равномерного вида и их кластеризации."""
import logging

from .number import (UniformFuzzyNumber, compare, distance, jaccard,
                     necessity_ge, possibility_ge, prob_greater)
from .collection import from_array, generate, load_csv, save_csv, to_array
from .clustering import (ClusterResult, auto_fcm, fcm, similarity_clusters,
                         similarity_matrix, xie_beni)

logging.getLogger("fuzzy_uniform").addHandler(logging.NullHandler())


def setup_log(path="fuzzy.log", level=logging.DEBUG):
    """Включает журнал библиотеки в файл (для внешних программ)."""
    h = logging.FileHandler(path, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    lg = logging.getLogger("fuzzy_uniform")
    lg.addHandler(h)
    lg.setLevel(level)
    return h
