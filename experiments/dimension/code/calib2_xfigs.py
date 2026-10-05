import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
import calib2 as C
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
rng = np.random.default_rng(0)
# 1. Takens on Lorenz
tr = C.CT.trajectories("lorenz", 1, 6000, rng, None)[0]; x = tr[:, 0]; tau = 25
fig = plt.figure(figsize=(10, 3.2))
ax = fig.add_subplot(1, 3, 1); ax.plot(x[:1500], lw=.7, color="#2f6db5"); ax.set_title("k = 1: только x(t) — всё слиплось в отрезок", fontsize=8.5); ax.set_xlabel("время")
ax = fig.add_subplot(1, 3, 2); ax.plot(x[:-tau], x[tau:], lw=.3, color="#2f6db5"); ax.set_title("k = 2: (x(t), x(t+τ)) — почти аттрактор,\nно траектории пересекаются", fontsize=8.5)
ax = fig.add_subplot(1, 3, 3, projection="3d"); ax.plot(x[:-2*tau], x[tau:-tau], x[2*tau:], lw=.3, color="#2f6db5"); ax.set_title("k = 3: пересечения исчезли", fontsize=8.5)
for a in fig.axes[2:]: a.set_xticks([]); a.set_yticks([]); a.set_zticks([])
fig.tight_layout(); fig.savefig("x1_takens.png", dpi=200)
# 2. gasket and zoom
centres, cv, w = C.simplex_cells(2, 9, None)
fig, axs = plt.subplots(1, 3, figsize=(10, 3.4))
axs[0].scatter(centres[:, 0], centres[:, 1], s=.05, c="k"); axs[0].set_title("салфетка Серпинского (19 683 клетки)", fontsize=8.5)
sub = centres[(centres[:, 0] < .13) & (centres[:, 1] < .13)]
axs[1].scatter(sub[:, 0], sub[:, 1], s=1, c="k"); axs[1].set_title("увеличение ×8: снова салфетка", fontsize=8.5)
sub = centres[(centres[:, 0] < .02) & (centres[:, 1] < .02)]
axs[2].scatter(sub[:, 0], sub[:, 1], s=12, c="k"); axs[2].set_title("увеличение ×50: видны отдельные клетки,\nвыстроенные в линии", fontsize=8.5)
for a in axs: a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig("x2_gasket.png", dpi=200)
# 3. multifractal measure
centres, cv, w = C.simplex_cells(2, 7, (0.6, 0.25, 0.15))
fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.4))
_, _, w0 = C.simplex_cells(2, 7, None)
for ax, ww, ttl in zip(axs, (w0, w), ("равномерная мера: все клетки одинаково «населены»", "мультифрактал: вес 0,6 / 0,25 / 0,15")):
    idx = rng.choice(len(centres), 6000, p=ww)
    ax.scatter(centres[idx, 0], centres[idx, 1], s=.3, c="k", alpha=.5); ax.set_title(ttl, fontsize=8.5); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig("x3_multifractal.png", dpi=200)
# 4. FNN/FNF schematic: circle projected to a line; long window folds
t = np.linspace(0, 2 * np.pi, 400)
fig, axs = plt.subplots(1, 3, figsize=(10, 3.0))
axs[0].plot(np.cos(t), np.sin(t), color="#2f6db5"); axs[0].plot([np.cos(1), np.cos(-1)], [np.sin(1), np.sin(-1)], "o", color="#c0392b")
axs[0].set_title("окружность: две красные точки далеко", fontsize=8.5)
axs[1].plot(np.cos(t), np.zeros_like(t), color="#2f6db5"); axs[1].plot([np.cos(1)], [0], "o", color="#c0392b", ms=9)
axs[1].set_title("проекция на прямую: они слиплись —\nложные ближайшие соседи (FNN)", fontsize=8.5)
s = np.linspace(0, 1, 600); y = np.sin(14 * np.pi * s) * (1 - s) ** 0.3
axs[2].plot(s, y, color="#2f6db5"); axs[2].set_title("слишком длинное окно: кривая «гармошкой»,\nдальние участки рядом — FNF", fontsize=8.5)
for a in axs: a.set_xticks([]); a.set_yticks([]); a.set_aspect("auto")
axs[0].set_aspect("equal")
fig.tight_layout(); fig.savefig("x4_fnn_fnf.png", dpi=200)
print("ok")
