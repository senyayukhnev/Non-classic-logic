"""Пример использования библиотеки fuzzy_uniform внешней программой (без GUI)."""
import fuzzy_uniform as fu

fu.setup_log("example.log")

A, B = fu.UniformFuzzyNumber(2, 5), fu.UniformFuzzyNumber(1, 3)
print("A =", A, " B =", B)
print("A+B =", A + B, " A-B =", A - B, " A*B =", A * B, " A/B =", A / B)
print("A > B:", A > B, " P(A>B) =", round(fu.prob_greater(A, B), 4))

data = fu.generate(200, groups=4, seed=1)
r = fu.fcm(data, k=4, seed=1)
print("FCM k=4, размеры:", r.sizes(), "прототипы:", r.centers)
r = fu.auto_fcm(data, 2, 8, seed=1)
print("Се–Бени выбрал k =", r.k)
r = fu.similarity_clusters(data, alpha=0.95)
print("Порог alpha=0.95 ->", r.k, "кластеров")
fu.save_csv(data, "data.csv", r.labels.tolist())
