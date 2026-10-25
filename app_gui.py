"""Графический интерфейс комплекса: операции, генерация набора, кластеризация."""
import logging
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import fuzzy_uniform as fu

log = logging.getLogger("fuzzy_uniform")


class TextHandler(logging.Handler):
    def __init__(self, widget):
        super().__init__(logging.INFO)
        self.widget = widget
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))

    def emit(self, record):
        self.widget.insert("end", self.format(record) + "\n")
        self.widget.see("end")


def plot_numbers(ax, numbers, labels=None, title=""):
    """Каждая величина — прямоугольная функция принадлежности высоты 1 (смещена по y для наглядности)."""
    cmap = matplotlib.colormaps["tab10"]
    order = sorted(range(len(numbers)), key=lambda i: numbers[i].center)
    for y, i in enumerate(order):
        x = numbers[i]
        c = cmap(labels[i] % 10) if labels is not None else "tab:blue"
        ax.hlines(y, x.a, x.b, color=c, lw=2)
    ax.set_xlabel("x")
    ax.set_ylabel("величины (по возрастанию центра)")
    ax.set_title(title)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Нечёткие величины равномерного вида и кластерный анализ")
        self.geometry("1150x780")
        self.numbers, self.labels = [], None

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self._tab_ops(nb)
        self._tab_set(nb)
        self._tab_cluster(nb)

        logbox = tk.Text(self, height=8)
        logbox.pack(fill="x")
        fu.setup_log("fuzzy.log")
        log.addHandler(TextHandler(logbox))
        log.info("Программа запущена, журнал: fuzzy.log")

    # ---------- вкладка 1: операции ----------
    def _tab_ops(self, nb):
        f = ttk.Frame(nb)
        nb.add(f, text="Операции")
        left = ttk.Frame(f, padding=8)
        left.pack(side="left", fill="y")
        self.ent = {}
        for r, (name, a, b) in enumerate([("A", 2, 5), ("B", 1, 3)]):
            ttk.Label(left, text=f"{name}:  a =").grid(row=r, column=0, sticky="e")
            for c, (key, val) in enumerate([("a", a), ("b", b)]):
                if c:
                    ttk.Label(left, text="b =").grid(row=r, column=2)
                e = ttk.Entry(left, width=8)
                e.insert(0, str(val))
                e.grid(row=r, column=1 + 2 * c, pady=2)
                self.ent[name + key] = e
        ops = [("A + B", lambda x, y: x + y), ("A − B", lambda x, y: x - y),
               ("A × B", lambda x, y: x * y), ("A ÷ B", lambda x, y: x / y),
               ("A ∩ B", lambda x, y: x.intersection(y)), ("Оболочка A ∪ B", lambda x, y: x.hull(y))]
        for i, (t, fn) in enumerate(ops):
            ttk.Button(left, text=t, command=lambda t=t, fn=fn: self._do_op(t, fn)).grid(
                row=2 + i // 2, column=(i % 2) * 2, columnspan=2, sticky="we", pady=2)
        ttk.Button(left, text="Сравнить A и B", command=self._do_cmp).grid(
            row=5, column=0, columnspan=4, sticky="we", pady=6)
        self.op_out = tk.Text(left, width=46, height=16)
        self.op_out.grid(row=6, column=0, columnspan=4)

        self.op_fig = Figure(figsize=(6, 5))
        self.op_canvas = FigureCanvasTkAgg(self.op_fig, f)
        self.op_canvas.get_tk_widget().pack(side="right", fill="both", expand=True)

    def _read_ab(self):
        g = lambda k: float(self.ent[k].get().replace(",", "."))
        return fu.UniformFuzzyNumber(g("Aa"), g("Ab")), fu.UniformFuzzyNumber(g("Ba"), g("Bb"))

    def _draw_mu(self, items):
        self.op_fig.clear()
        ax = self.op_fig.add_subplot()
        for (name, x), dy in zip(items, (0, 0.03, 0.06)):
            xs = [x.a - 1e-9, x.a, x.b, x.b + 1e-9]
            ax.plot(xs, [dy, 1 - dy, 1 - dy, dy], lw=2, label=f"{name} = {x}")
        ax.set_ylim(-0.05, 1.1)
        ax.set_ylabel("μ(x)")
        ax.legend()
        ax.grid(alpha=0.3)
        self.op_canvas.draw()

    def _do_op(self, title, fn):
        try:
            A, B = self._read_ab()
            R = fn(A, B)
        except Exception as e:
            log.error("%s: %s", title, e)
            messagebox.showerror("Ошибка", str(e))
            return
        log.info("%s: A=%s, B=%s -> %s", title, A, B, R)
        self.op_out.delete("1.0", "end")
        if R is None:
            self.op_out.insert("end", f"{title} = ∅ (пустое множество)\n")
            self._draw_mu([("A", A), ("B", B)])
            return
        self.op_out.insert("end", f"{title} = {R}\nцентр = {R.center:.4g}\nширина = {R.width:.4g}\n"
                                  f"дисперсия = {R.variance:.4g}\n")
        self._draw_mu([("A", A), ("B", B), ("R", R)])

    def _do_cmp(self):
        try:
            A, B = self._read_ab()
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            return
        sign = {1: "A > B", 0: "A = B", -1: "A < B"}[fu.compare(A, B)]
        txt = (f"По центру тяжести: {sign}\n"
               f"  центр A = {A.center:.4g}, центр B = {B.center:.4g}\n"
               f"P(A > B) = {fu.prob_greater(A, B):.4f}\n"
               f"Poss(A ≥ B) = {fu.possibility_ge(A, B):.0f}\n"
               f"Nec(A ≥ B) = {fu.necessity_ge(A, B):.0f}\n"
               f"Сходство Жаккара = {fu.jaccard(A, B):.4f}\n"
               f"Расстояние d(A,B) = {fu.distance(A, B):.4f}\n")
        log.info("Сравнение A=%s, B=%s: %s", A, B, txt.replace("\n", "; "))
        self.op_out.delete("1.0", "end")
        self.op_out.insert("end", txt)
        self._draw_mu([("A", A), ("B", B)])

    # ---------- вкладка 2: набор величин ----------
    def _tab_set(self, nb):
        f = ttk.Frame(nb)
        nb.add(f, text="Набор величин")
        left = ttk.Frame(f, padding=8)
        left.pack(side="left", fill="y")
        self.gen = {}
        for r, (k, t, v) in enumerate([("n", "Количество (100–300)", 200), ("groups", "Число групп (0 = нет)", 4),
                                       ("lo", "Центр от", 0), ("hi", "Центр до", 100),
                                       ("wlo", "Ширина от", 1), ("whi", "Ширина до", 10),
                                       ("spread", "Разброс внутри группы", 5), ("seed", "seed (пусто = случайно)", 1)]):
            ttk.Label(left, text=t).grid(row=r, column=0, sticky="w")
            e = ttk.Entry(left, width=10)
            e.insert(0, str(v))
            e.grid(row=r, column=1, pady=1)
            self.gen[k] = e
        ttk.Button(left, text="Сгенерировать", command=self._generate).grid(row=8, column=0, columnspan=2, sticky="we", pady=4)
        ttk.Button(left, text="Загрузить CSV…", command=self._load).grid(row=9, column=0, columnspan=2, sticky="we")
        ttk.Button(left, text="Сохранить CSV…", command=self._save).grid(row=10, column=0, columnspan=2, sticky="we", pady=4)
        self.tree = ttk.Treeview(left, columns=("i", "a", "b", "c", "cl"), show="headings", height=18)
        for c, t, w in [("i", "№", 40), ("a", "a", 70), ("b", "b", 70), ("c", "центр", 70), ("cl", "кластер", 60)]:
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="e")
        self.tree.grid(row=11, column=0, columnspan=2, sticky="ns")

        self.set_fig = Figure(figsize=(6, 5))
        self.set_canvas = FigureCanvasTkAgg(self.set_fig, f)
        self.set_canvas.get_tk_widget().pack(side="right", fill="both", expand=True)

    def _generate(self):
        try:
            g = {k: e.get().strip() for k, e in self.gen.items()}
            n = int(g["n"])
            if not 100 <= n <= 300:
                log.info("Внимание: по заданию n должно быть 100–300, задано %d", n)
            self.numbers = fu.generate(n, int(g["groups"]), (float(g["lo"]), float(g["hi"])),
                                       (float(g["wlo"]), float(g["whi"])), float(g["spread"]),
                                       int(g["seed"]) if g["seed"] else None)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            return
        self.labels = None
        self._refresh_set()

    def _load(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if p:
            try:
                self.numbers, self.labels = fu.load_csv(p), None
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))
                return
            self._refresh_set()

    def _save(self):
        if not self.numbers:
            return messagebox.showinfo("Нет данных", "Сначала сгенерируйте набор")
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            fu.save_csv(self.numbers, p, None if self.labels is None else self.labels.tolist())

    def _refresh_set(self):
        self.tree.delete(*self.tree.get_children())
        for i, x in enumerate(self.numbers):
            cl = "" if self.labels is None else int(self.labels[i]) + 1
            self.tree.insert("", "end", values=(i + 1, f"{x.a:.3f}", f"{x.b:.3f}", f"{x.center:.3f}", cl))
        self.set_fig.clear()
        plot_numbers(self.set_fig.add_subplot(), self.numbers, self.labels, f"Набор: {len(self.numbers)} величин")
        self.set_canvas.draw()

    # ---------- вкладка 3: кластеризация ----------
    def _tab_cluster(self, nb):
        f = ttk.Frame(nb)
        nb.add(f, text="Кластеризация")
        left = ttk.Frame(f, padding=8)
        left.pack(side="left", fill="y")
        self.method = tk.StringVar(value="fcm")
        for r, (v, t) in enumerate([("fcm", "Нечёткие c-средние, заданное k"),
                                    ("auto", "По критерию Се–Бени (авто k)"),
                                    ("alpha", "По порогу сходства α")]):
            ttk.Radiobutton(left, text=t, variable=self.method, value=v).grid(row=r, column=0, columnspan=2, sticky="w")
        self.cl = {}
        for r, (k, t, v) in enumerate([("k", "k (число кластеров)", 4), ("m", "m (фаззификатор)", 2),
                                       ("kmax", "k max (для авто)", 10), ("alpha", "α (порог 0..1)", 0.95),
                                       ("seed", "seed", 1)], start=3):
            ttk.Label(left, text=t).grid(row=r, column=0, sticky="w")
            e = ttk.Entry(left, width=8)
            e.insert(0, str(v))
            e.grid(row=r, column=1, pady=1)
            self.cl[k] = e
        ttk.Button(left, text="Выполнить", command=self._cluster).grid(row=8, column=0, columnspan=2, sticky="we", pady=6)
        self.cl_out = tk.Text(left, width=42, height=26)
        self.cl_out.grid(row=9, column=0, columnspan=2)

        self.cl_fig = Figure(figsize=(7, 5))
        self.cl_canvas = FigureCanvasTkAgg(self.cl_fig, f)
        self.cl_canvas.get_tk_widget().pack(side="right", fill="both", expand=True)

    def _cluster(self):
        if not self.numbers:
            return messagebox.showinfo("Нет данных", "Сначала сгенерируйте или загрузите набор")
        try:
            g = {k: e.get().strip().replace(",", ".") for k, e in self.cl.items()}
            seed = int(g["seed"]) if g["seed"] else None
            meth = self.method.get()
            if meth == "fcm":
                res = fu.fcm(self.numbers, int(g["k"]), float(g["m"]), seed=seed)
            elif meth == "auto":
                res = fu.auto_fcm(self.numbers, 2, int(g["kmax"]), float(g["m"]), seed=seed)
            else:
                res = fu.similarity_clusters(self.numbers, float(g["alpha"]))
        except Exception as e:
            log.error("Кластеризация: %s", e)
            messagebox.showerror("Ошибка", str(e))
            return
        self.labels = res.labels
        out = [f"Кластеров: {res.k}"]
        if res.iterations:
            out.append(f"Итераций FCM: {res.iterations}")
        if res.membership is not None:
            out.append(f"Индекс Се–Бени: {fu.xie_beni(self.numbers, res, float(g['m'])):.4f}")
        for k, v in res.scores.items():
            out.append(f"  k={k}: XB={v:.4f}" + ("  <- лучший" if k == res.k else ""))
        out.append("\nКластер: размер, прототип")
        for c, (sz, v) in enumerate(zip(res.sizes(), res.centers)):
            out.append(f"{c + 1:>3}: {sz:>4}  {v}")
        self.cl_out.delete("1.0", "end")
        self.cl_out.insert("end", "\n".join(out))
        log.info("Результат кластеризации: размеры %s", res.sizes())

        self.cl_fig.clear()
        ax1, ax2 = self.cl_fig.subplots(1, 2)
        plot_numbers(ax1, self.numbers, res.labels, "Величины по кластерам")
        cmap = matplotlib.colormaps["tab10"]
        ax2.scatter([x.center for x in self.numbers], [x.width for x in self.numbers],
                    c=[cmap(l % 10) for l in res.labels], s=14)
        ax2.scatter([v.center for v in res.centers], [v.width for v in res.centers],
                    marker="X", s=120, c="k", label="прототипы")
        ax2.set_xlabel("центр (a+b)/2")
        ax2.set_ylabel("ширина b−a")
        ax2.legend()
        self.cl_fig.tight_layout()
        self.cl_canvas.draw()
        self._refresh_set()


if __name__ == "__main__":
    App().mainloop()
