"""Множество нечётких величин: генерация, сохранение, загрузка."""
import csv
import logging

import numpy as np

from .number import UniformFuzzyNumber

log = logging.getLogger("fuzzy_uniform")


def generate(n=200, groups=4, center_range=(0, 100), width_range=(1, 10), spread=5.0, seed=None):
    """Генерирует n величин (100..300 по заданию), сгруппированных вокруг `groups` центров.

    groups=0 — центры равномерно случайны по всему диапазону (без структуры).
    """
    if not 1 <= n <= 100_000:
        raise ValueError("n должно быть в [1; 100000]")
    rng = np.random.default_rng(seed)
    lo, hi = center_range
    if groups > 0:
        means = rng.uniform(lo, hi, groups)
        centers = rng.choice(means, n) + rng.normal(0, spread, n)
    else:
        centers = rng.uniform(lo, hi, n)
    widths = rng.uniform(*width_range, n)
    res = [UniformFuzzyNumber(c - w / 2, c + w / 2) for c, w in zip(centers, widths)]
    log.info("Сгенерировано %d величин (групп=%d, seed=%s)", n, groups, seed)
    return res


def to_array(numbers):
    """Матрица n x 2 параметров [a, b]."""
    return np.array([x.to_tuple() for x in numbers], dtype=float).reshape(-1, 2)


def from_array(arr):
    return [UniformFuzzyNumber(a, b) for a, b in arr]


def save_csv(numbers, path, labels=None):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["a", "b"] + (["cluster"] if labels is not None else []))
        for i, x in enumerate(numbers):
            w.writerow([x.a, x.b] + ([labels[i]] if labels is not None else []))
    log.info("Сохранено %d величин в %s", len(numbers), path)


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter=";"))
    res = [UniformFuzzyNumber(float(r[0]), float(r[1])) for r in rows[1:] if r]
    log.info("Загружено %d величин из %s", len(res), path)
    return res
