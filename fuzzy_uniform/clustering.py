"""Кластерный анализ множества нечётких величин.

Каждая величина U[a; b] представлена точкой (a, b); расстояние —
`number.distance` (метрика по альфа-срезам), что с точностью до множителя
1/sqrt(2) совпадает с евклидовым расстоянием в плоскости (a, b).

* fcm               — нечёткие c-средние на заданное число кластеров k;
* similarity_clusters — по критерию: альфа-срез транзитивного замыкания
                      нечёткого отношения сходства (порог alpha);
* auto_fcm          — число кластеров выбирается по критерию Се–Бени.
"""
import logging
from dataclasses import dataclass, field

import numpy as np

from .collection import from_array, to_array

log = logging.getLogger("fuzzy_uniform")


@dataclass
class ClusterResult:
    labels: np.ndarray            # номер кластера для каждой величины
    centers: list                 # прототипы кластеров (UniformFuzzyNumber)
    membership: np.ndarray = None  # n x k степени принадлежности (для FCM)
    iterations: int = 0
    scores: dict = field(default_factory=dict)  # значения критерия (auto_fcm)

    @property
    def k(self):
        return len(self.centers)

    def sizes(self):
        return np.bincount(self.labels, minlength=self.k).tolist()


def _dist2(X, V):
    """Квадраты расстояний n x k по метрике number.distance."""
    return ((X[:, None, :] - V[None, :, :]) ** 2).sum(axis=2) / 2


def fcm(numbers, k, m=2.0, eps=1e-6, max_iter=500, seed=None):
    X = to_array(numbers)
    n = len(X)
    if not 1 <= k <= n:
        raise ValueError(f"k должно быть в [1; {n}]")
    if m <= 1:
        raise ValueError("m должно быть > 1")
    rng = np.random.default_rng(seed)
    U = rng.random((n, k))
    U /= U.sum(axis=1, keepdims=True)
    for it in range(1, max_iter + 1):
        W = U ** m
        V = W.T @ X / W.sum(axis=0)[:, None]
        D = np.fmax(_dist2(X, V), 1e-12)
        inv = D ** (-1 / (m - 1))
        U_new = inv / inv.sum(axis=1, keepdims=True)
        delta = np.abs(U_new - U).max()
        U = U_new
        if delta < eps:
            break
    V = np.sort(V, axis=1)  # гарантия a <= b у прототипа
    log.info("FCM: k=%d, m=%.2f, итераций=%d", k, m, it)
    return ClusterResult(U.argmax(axis=1), from_array(V), U, it)


def xie_beni(numbers, res, m=2.0):
    """Индекс Се–Бени: компактность / разделимость. Чем меньше, тем лучше."""
    X, V = to_array(numbers), to_array(res.centers)
    if len(V) < 2:
        return float("inf")
    comp = ((res.membership ** m) * _dist2(X, V)).sum()
    sep = min(_dist2(V[i:i + 1], np.delete(V, i, 0)).min() for i in range(len(V)))
    return comp / (len(X) * max(sep, 1e-12))


def auto_fcm(numbers, k_min=2, k_max=10, m=2.0, seed=None):
    """Перебирает k и возвращает разбиение с минимальным индексом Се–Бени."""
    k_max = min(k_max, len(numbers) - 1)
    best, scores = None, {}
    for k in range(k_min, k_max + 1):
        r = fcm(numbers, k, m, seed=seed)
        scores[k] = xie_beni(numbers, r, m)
        if best is None or scores[k] < scores[best.k]:
            best = r
    best.scores = scores
    log.info("Авто-выбор по Се–Бени: k=%d, индексы=%s", best.k,
             {k: round(v, 4) for k, v in scores.items()})
    return best


def similarity_matrix(numbers):
    """R(i,j) = 1 - d(i,j)/d_max — нечёткое отношение сходства (рефлексивно, симметрично)."""
    X = to_array(numbers)
    D = np.sqrt(_dist2(X, X))
    dmax = D.max()
    return 1 - D / dmax if dmax > 0 else np.ones_like(D)


def similarity_clusters(numbers, alpha):
    """Кластеры = классы эквивалентности альфа-среза max-min транзитивного замыкания R.

    Альфа-срез замыкания совпадает с компонентами связности графа с рёбрами
    R(i,j) >= alpha, поэтому замыкание явно не строится (O(n^2) вместо O(n^3 log n)).
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha должно быть в [0; 1]")
    R = similarity_matrix(numbers)
    n = len(R)
    labels = -np.ones(n, dtype=int)
    k = 0
    for s in range(n):
        if labels[s] >= 0:
            continue
        stack, labels[s] = [s], k
        while stack:
            i = stack.pop()
            for j in np.flatnonzero((R[i] >= alpha) & (labels < 0)):
                labels[j] = k
                stack.append(j)
        k += 1
    X = to_array(numbers)
    centers = from_array([X[labels == c].mean(axis=0) for c in range(k)])
    log.info("Кластеризация по порогу сходства alpha=%.3f: %d кластеров", alpha, k)
    return ClusterResult(labels, centers)
