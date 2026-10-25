"""Нечёткая величина равномерного вида (прямоугольная функция принадлежности).

Плотность равномерного распределения на [a, b] постоянна: f(x) = 1/(b-a).
Нормированная к высоте 1, она даёт функцию принадлежности
    mu(x) = 1, если a <= x <= b, иначе 0.
Все альфа-срезы такой величины совпадают с [a, b], поэтому принцип
обобщения Заде сводится к интервальной арифметике, а результат любой
арифметической операции снова является величиной того же вида.
"""
import logging
import math

log = logging.getLogger("fuzzy_uniform")


class UniformFuzzyNumber:
    __slots__ = ("a", "b")

    def __init__(self, a, b=None):
        if b is None:
            b = a  # чёткое число как вырожденный интервал
        a, b = float(a), float(b)
        if math.isnan(a) or math.isnan(b) or a > b:
            raise ValueError(f"Нужно a <= b, получено a={a}, b={b}")
        self.a, self.b = a, b

    # ---- характеристики ----
    def mu(self, x):
        return 1.0 if self.a <= x <= self.b else 0.0

    def alpha_cut(self, alpha=1.0):
        if not 0 < alpha <= 1:
            raise ValueError("alpha должно быть в (0, 1]")
        return self.a, self.b

    @property
    def center(self):
        """Центр тяжести (дефаззификация) = мат. ожидание равномерного распределения."""
        return (self.a + self.b) / 2

    @property
    def width(self):
        return self.b - self.a

    @property
    def variance(self):
        return self.width ** 2 / 12

    # ---- арифметика (интервальная, по альфа-срезам) ----
    @staticmethod
    def _coerce(x):
        return x if isinstance(x, UniformFuzzyNumber) else UniformFuzzyNumber(x)

    def __add__(self, o):
        o = self._coerce(o)
        r = UniformFuzzyNumber(self.a + o.a, self.b + o.b)
        log.debug("%s + %s = %s", self, o, r)
        return r

    def __sub__(self, o):
        o = self._coerce(o)
        r = UniformFuzzyNumber(self.a - o.b, self.b - o.a)
        log.debug("%s - %s = %s", self, o, r)
        return r

    def __mul__(self, o):
        o = self._coerce(o)
        p = (self.a * o.a, self.a * o.b, self.b * o.a, self.b * o.b)
        r = UniformFuzzyNumber(min(p), max(p))
        log.debug("%s * %s = %s", self, o, r)
        return r

    def __truediv__(self, o):
        o = self._coerce(o)
        if o.a <= 0 <= o.b:
            raise ZeroDivisionError(f"Делитель {o} содержит 0")
        r = self * UniformFuzzyNumber(1 / o.b, 1 / o.a)
        log.debug("%s / %s = %s", self, o, r)
        return r

    def __radd__(self, o):
        return self._coerce(o) + self

    def __rsub__(self, o):
        return self._coerce(o) - self

    def __rmul__(self, o):
        return self._coerce(o) * self

    def __rtruediv__(self, o):
        return self._coerce(o) / self

    def __neg__(self):
        return UniformFuzzyNumber(-self.b, -self.a)

    # ---- нечёткие операции над множествами (min / max) ----
    def intersection(self, o):
        """A ∩ B: mu = min(muA, muB). None, если пересечение пусто."""
        a, b = max(self.a, o.a), min(self.b, o.b)
        return UniformFuzzyNumber(a, b) if a <= b else None

    def hull(self, o):
        """Наименьшая величина того же вида, содержащая A ∪ B."""
        return UniformFuzzyNumber(min(self.a, o.a), max(self.b, o.b))

    # ---- сравнение ----
    def __eq__(self, o):
        return isinstance(o, UniformFuzzyNumber) and self.a == o.a and self.b == o.b

    def __hash__(self):
        return hash((self.a, self.b))

    def __lt__(self, o):
        return compare(self, self._coerce(o)) < 0

    def __le__(self, o):
        return compare(self, self._coerce(o)) <= 0

    def __gt__(self, o):
        return compare(self, self._coerce(o)) > 0

    def __ge__(self, o):
        return compare(self, self._coerce(o)) >= 0

    def __repr__(self):
        return f"U[{self.a:.4g}; {self.b:.4g}]"

    def to_tuple(self):
        return self.a, self.b


def compare(x, y):
    """Сравнение по центру тяжести, при равенстве — по ширине (уже = меньше)."""
    kx, ky = (x.center, x.width), (y.center, y.width)
    return (kx > ky) - (kx < ky)


def possibility_ge(x, y):
    """Poss(X >= Y) = sup min(muX(u), muY(v)), u >= v."""
    return 1.0 if x.b >= y.a else 0.0


def necessity_ge(x, y):
    """Nec(X >= Y) = 1 - Poss(X < Y)."""
    return 1.0 if x.a >= y.b else 0.0


def prob_greater(x, y):
    """P(X > Y) для независимых X~U[x.a,x.b], Y~U[y.a,y.b] — степень предпочтения X над Y."""
    def F(t):  # функция распределения Y
        if y.width == 0:
            return 1.0 if t > y.a else (0.5 if t == y.a else 0.0)
        return min(1.0, max(0.0, (t - y.a) / y.width))

    if x.width == 0:
        return F(x.a)
    pts = sorted({x.a, x.b} | {p for p in (y.a, y.b) if x.a < p < x.b})
    # F кусочно-линейна между точками излома -> формула средней точки точна
    return sum(F((p + q) / 2) * (q - p) for p, q in zip(pts, pts[1:])) / x.width


def jaccard(x, y):
    """Степень сходства |A∩B| / |A∪B| (для чётких точек — 1 при совпадении)."""
    i = x.intersection(y)
    u = max(x.b, y.b) - min(x.a, y.a)
    if u == 0:
        return 1.0
    return 0.0 if i is None else i.width / u


def distance(x, y):
    """Метрика по альфа-срезам: sqrt(∫((a1-a2)^2 + (b1-b2)^2)/2 dα) — для U-величин без интеграла."""
    return math.sqrt(((x.a - y.a) ** 2 + (x.b - y.b) ** 2) / 2)
