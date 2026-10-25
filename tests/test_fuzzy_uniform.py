import os
import tempfile
import unittest

import fuzzy_uniform as fu
from fuzzy_uniform import UniformFuzzyNumber as U


class TestNumber(unittest.TestCase):
    def test_arith(self):
        A, B = U(2, 5), U(1, 3)
        self.assertEqual(A + B, U(3, 8))
        self.assertEqual(A - B, U(-1, 4))
        self.assertEqual(A * B, U(2, 15))
        self.assertEqual(A / B, U(2 / 3, 5))
        self.assertEqual(U(-2, 3) * U(-1, 4), U(-8, 12))
        self.assertEqual(A + 1, U(3, 6))
        self.assertEqual(10 - A, U(5, 8))
        with self.assertRaises(ZeroDivisionError):
            A / U(-1, 1)
        with self.assertRaises(ValueError):
            U(3, 1)

    def test_compare(self):
        A, B = U(2, 5), U(1, 3)
        self.assertTrue(A > B and B < A and A >= A)
        self.assertAlmostEqual(fu.prob_greater(U(0, 1), U(0, 1)), 0.5)
        self.assertAlmostEqual(fu.prob_greater(U(0, 2), U(1, 3)), 0.125)
        self.assertAlmostEqual(fu.prob_greater(U(0, 2), U(1, 3)) + fu.prob_greater(U(1, 3), U(0, 2)), 1)
        self.assertEqual(fu.prob_greater(U(5), U(0, 10)), 0.5)
        self.assertEqual(fu.possibility_ge(B, A), 1.0)
        self.assertEqual(fu.necessity_ge(B, A), 0.0)
        self.assertAlmostEqual(fu.jaccard(U(0, 2), U(1, 3)), 1 / 3)
        self.assertIsNone(U(0, 1).intersection(U(2, 3)))


class TestClustering(unittest.TestCase):
    def setUp(self):
        self.data = fu.generate(200, groups=3, center_range=(0, 300), spread=3, seed=7)

    def test_generate(self):
        self.assertEqual(len(self.data), 200)
        self.assertTrue(all(x.a <= x.b for x in self.data))

    def test_fcm(self):
        r = fu.fcm(self.data, 3, seed=0)
        self.assertEqual(r.k, 3)
        self.assertEqual(sum(r.sizes()), 200)
        self.assertTrue(((r.membership.sum(axis=1) - 1) ** 2 < 1e-12).all())

    def test_auto(self):
        self.assertEqual(fu.auto_fcm(self.data, 2, 8, seed=0).k, 3)

    def test_alpha(self):
        self.assertEqual(fu.similarity_clusters(self.data, 0).k, 1)
        self.assertEqual(fu.similarity_clusters(self.data, 1).k, len(set(self.data)))
        self.assertEqual(fu.similarity_clusters(self.data, 0.9).k, 3)

    def test_csv(self):
        p = os.path.join(tempfile.mkdtemp(), "d.csv")
        fu.save_csv(self.data, p)
        self.assertEqual(fu.load_csv(p), self.data)


if __name__ == "__main__":
    unittest.main()
