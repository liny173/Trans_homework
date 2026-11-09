# -*- coding: utf-8 -*-
"""
03_validation.py —— 结果检验（内部验证）

在 D1、D2 上模拟 D3 的有限观测情形（仅保留 J03/J04/J06/J10/J13 五个已知点，
隐藏其余 13 个点），检验模型的空间迁移能力与跨日动态校准能力。

模型对照：
    M0  历史基线        参考日同时段占比（不使用目标日任何信息）
    M1  已知点简单均值  目标日 5 个已知点的算术平均（不使用历史信息）
    M2  IDW 空间插值    目标日已知点的反距离加权平均（不使用历史信息）
    M3d 距离校准        历史基线 + 已知点变化量的 IDW 加权校准（M3 的消融）
    M3  时空动态校准    历史基线 + 已知点变化量的 距离×历史相似性 加权校准（最终模型）

验证层次：
    第一层 空间迁移能力：D1/D2 日内，仅用当日 5 个已知点预测 13 个隐藏点；
    第二层 跨日动态校准：D1、D2 互为参考日与目标日，模拟完整预测流程。
    注意：D1 为工作日、D2 为非工作日，跨日验证存在日类型差异，是 D3（同为
    工作日的 D1 作基线）真实任务难度的保守上界。

评价指标：MAE、RMSE、Jensen-Shannon 散度（JSD）。

输出：
    outputs/validation_metrics.csv  模型对照指标表
    outputs/grid_search.csv         参数网格搜索结果
    figures/fig9_模型对比.png、fig10_参数网格.png、fig11_误差分析.png

"""

# ---------------------------------------------------------------- copy自01_data_analysis.py
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "course_data.csv"
OUT_DIR = BASE_DIR / "outputs"
FIG_DIR = BASE_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

COUNT_COLS = [
    "count_small_nev_passenger", "count_small_nev_freight",
    "count_small_ice_passenger", "count_small_ice_freight",
    "count_large_nev_passenger", "count_large_nev_freight",
    "count_large_ice_passenger", "count_large_ice_freight",
]
SHARE_COLS = [c.replace("count_", "share_") for c in COUNT_COLS]
CLASS_NAMES = ["小型新能源客车", "小型新能源货车", "小型燃油客车", "小型燃油货车",
               "大型新能源客车", "大型新能源货车", "大型燃油客车", "大型燃油货车"]

KNOWN_POINTS = ["J03", "J04", "J06", "J10", "J13"]
IDW_ALPHA = 2.0     # M2/M3d 的反距离幂次
EPS = 1e-9


# ---------------------------------------------------------------- 数据工具
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    for c in COUNT_COLS:
        df[c.replace("count_", "share_")] = df[c] / df["total_count"]
    return df


class DayData:
    """预计算三天的占比数组、距离矩阵与组成相似性矩阵。"""

    def __init__(self, df: pd.DataFrame):
        self.points = sorted(df["point_id"].unique())
        self.hidden = [p for p in self.points if p not in KNOWN_POINTS]
        self.known_idx = [self.points.index(p) for p in KNOWN_POINTS]
        self.hidden_idx = [self.points.index(p) for p in self.hidden]

        coords = df.groupby("point_id")[["coord_x", "coord_y"]].first()
        xy = coords.loc[self.points].to_numpy()
        self.dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))

        idx = pd.MultiIndex.from_product([self.points, range(24)],
                                         names=["point_id", "hour"])
        self.arr, self.sim = {}, {}
        for day in ["D1", "D2"]:
            a = (df[df["day_id"] == day]
                 .set_index(["point_id", "hour"])[SHARE_COLS]
                 .reindex(idx).to_numpy().reshape(len(self.points), 24, 8))
            self.arr[day] = a
            self.sim[day] = np.corrcoef(a.reshape(len(self.points), -1))


# ---------------------------------------------------------------- 评价指标
def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.abs(pred - truth).mean())


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(((pred - truth) ** 2).mean()))


def jsd(pred: np.ndarray, truth: np.ndarray, eps: float = 1e-12) -> float:
    """逐条记录（每行一个分布）计算 JSD 后取平均。"""
    p = np.clip(truth, eps, 1)
    q = np.clip(pred, eps, 1)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)
    val = 0.5 * np.sum(p * np.log(p / m), axis=1) + \
        0.5 * np.sum(q * np.log(q / m), axis=1)
    return float(val.mean())


def metric_dict(pred: np.ndarray, truth: np.ndarray) -> dict:
    return {"MAE": mae(pred, truth), "RMSE": rmse(pred, truth),
            "JSD": jsd(pred, truth)}


# ---------------------------------------------------------------- 模型
def combo_weights(data: DayData, i: int, sim: np.ndarray,
                  h: float, gamma: float) -> np.ndarray:
    """距离核 × 历史相似性 的组合权重（对 5 个已知点归一化）。"""
    d = data.dist[i, data.known_idx]
    s = np.maximum(sim[i, data.known_idx], 0.0)
    w = np.exp(-d / h) * np.exp(gamma * s)
    return w / w.sum()


def idw_weights(data: DayData, i: int, alpha: float = IDW_ALPHA) -> np.ndarray:
    d = data.dist[i, data.known_idx]
    w = (d + 1e-6) ** (-alpha)
    return w / w.sum()


def eval_spatial(data: DayData, day: str, sim_other: np.ndarray,
                 h: float, gamma: float) -> dict[str, dict]:
    """第一层：日内空间迁移。M1 均值 / M2 IDW / M3 组合权重（不做跨日校准）。"""
    arr = data.arr[day]
    preds = {"M1": [], "M2": [], "M3": []}
    truth = []
    for i in data.hidden_idx:
        w_idw = idw_weights(data, i)
        w_cmb = combo_weights(data, i, sim_other, h, gamma)
        for t in range(24):
            vals = arr[data.known_idx, t, :]
            preds["M1"].append(vals.mean(axis=0))
            preds["M2"].append(w_idw @ vals)
            preds["M3"].append(w_cmb @ vals)
            truth.append(arr[i, t])
    truth = np.array(truth)
    return {m: metric_dict(np.array(v), truth) for m, v in preds.items()}


def eval_calibration(data: DayData, target: str, ref: str,
                     h: float, gamma: float, lam: float
                     ) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    """第二层：跨日动态校准。参考日基线 + 目标日已知点变化量的加权修正。"""
    tar, bas, sim = data.arr[target], data.arr[ref], data.sim[ref]
    preds = {"M0": [], "M3d": [], "M3": []}
    truth = []
    for i in data.hidden_idx:
        w_idw = idw_weights(data, i)
        w_cmb = combo_weights(data, i, sim, h, gamma)
        for t in range(24):
            delta = tar[data.known_idx, t, :] - bas[data.known_idx, t, :]
            base = bas[i, t]
            preds["M0"].append(base)
            for name, w in [("M3d", w_idw), ("M3", w_cmb)]:
                p = base + lam * (w @ delta)
                p = np.maximum(p, EPS)
                preds[name].append(p / p.sum())
            truth.append(tar[i, t])
    truth = np.array(truth)
    metrics = {m: metric_dict(np.array(v), truth) for m, v in preds.items()}
    arrays = {m: np.array(v) for m, v in preds.items()}
    arrays = {"truth": truth, **arrays}
    return metrics, arrays


# ---------------------------------------------------------------- 参数网格搜索
def grid_search(data: DayData) -> pd.DataFrame:
    rows = []
    for h in [0.2, 0.4, 0.6, 0.8, 1.0]:
        for gamma in [0.0, 0.5, 1.0, 2.0]:
            for lam in [0.0, 0.5, 1.0, 1.5]:
                m1, _ = eval_calibration(data, "D1", "D2", h, gamma, lam)
                m2, _ = eval_calibration(data, "D2", "D1", h, gamma, lam)
                rows.append({"h": h, "gamma": gamma, "lambda": lam,
                             "MAE": np.mean([m1["M3"]["MAE"], m2["M3"]["MAE"]]),
                             "RMSE": np.mean([m1["M3"]["RMSE"], m2["M3"]["RMSE"]]),
                             "JSD": np.mean([m1["M3"]["JSD"], m2["M3"]["JSD"]])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 可视化
def fig9_model_compare(level1: dict, level2: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {"D1": "#1f77b4", "D2": "#ff7f0e"}

    models1 = ["M1", "M2", "M3"]
    labels1 = ["M1 已知点均值", "M2 IDW插值", "M3 组合权重插值"]
    x = np.arange(len(models1))
    for k, day in enumerate(["D1", "D2"]):
        v = [level1[day][m]["MAE"] for m in models1]
        ax1.bar(x + (k - 0.5) * 0.36, v, 0.36, label=f"目标日 {day}",
                color=colors[day], alpha=0.85)
        for xi, vi in zip(x + (k - 0.5) * 0.36, v):
            ax1.text(xi, vi, f"{vi:.4f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x, labels1)
    ax1.set_ylabel("MAE")
    ax1.set_title("第一层：空间迁移能力验证")
    ax1.legend()
    ax1.grid(alpha=0.3, axis="y")

    models2 = ["M0", "M1", "M2", "M3d", "M3"]
    labels2 = ["M0\n历史基线", "M1\n已知点均值", "M2\nIDW插值",
               "M3d\n距离校准", "M3\n时空动态校准"]
    x = np.arange(len(models2))
    for k, day in enumerate(["D1", "D2"]):
        v = [level2[day][m]["MAE"] for m in models2]
        ax2.bar(x + (k - 0.5) * 0.36, v, 0.36, label=f"目标日 {day}",
                color=colors[day], alpha=0.85)
        for xi, vi in zip(x + (k - 0.5) * 0.36, v):
            ax2.text(xi, vi, f"{vi:.4f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x, labels2, fontsize=9)
    ax2.set_ylabel("MAE")
    ax2.set_title("第二层：跨日动态校准验证")
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9_模型对比.png", dpi=150)
    plt.close(fig)


def fig10_grid(grid: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    sub = grid[np.isclose(grid["lambda"], 0.5)]
    pivot = sub.pivot(index="gamma", columns="h", values="MAE")
    im = ax1.imshow(pivot.to_numpy(), cmap="viridis_r", aspect="auto")
    ax1.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax1.set_yticks(range(len(pivot.index)), pivot.index)
    ax1.set_xlabel("距离核带宽 h")
    ax1.set_ylabel("相似性权重 γ")
    ax1.set_title("MAE 网格热图（λ=0.5，两日平均）")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax1.text(j, i, f"{pivot.to_numpy()[i, j]:.4f}",
                     ha="center", va="center", fontsize=8, color="w")
    fig.colorbar(im, ax=ax1, shrink=0.85)

    for gamma, mk in [(0.0, "o--"), (0.5, "s--"), (1.0, "^--"), (2.0, "d-")]:
        sub = grid[(grid["gamma"] == gamma) & np.isclose(grid["h"], 0.2)]
        ax2.plot(sub["lambda"], sub["MAE"], mk, lw=1.4, ms=5, label=f"γ={gamma}")
    ax2.set_xlabel("校准系数 λ")
    ax2.set_ylabel("MAE")
    ax2.set_title("MAE 随 λ 变化（h=0.2，两日平均）")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig10_参数网格.png", dpi=150)
    plt.close(fig)


def fig11_error_analysis(arr1: dict, arr2: dict, data: DayData) -> None:
    truth = np.vstack([arr1["truth"], arr2["truth"]])
    m0 = np.vstack([arr1["M0"], arr2["M0"]])
    m3 = np.vstack([arr1["M3"], arr2["M3"]])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.4))

    x = np.arange(8)
    ax1.bar(x - 0.18, np.abs(m0 - truth).mean(axis=0), 0.36, label="M0 历史基线")
    ax1.bar(x + 0.18, np.abs(m3 - truth).mean(axis=0), 0.36, label="M3 时空动态校准")
    ax1.set_xticks(x, CLASS_NAMES, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("MAE")
    ax1.set_title("分车型误差（两日合并）")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, axis="y")

    e0 = np.abs(m0 - truth).reshape(2, 13, 24, 8).mean(axis=(0, 1, 3))
    e3 = np.abs(m3 - truth).reshape(2, 13, 24, 8).mean(axis=(0, 1, 3))
    ax2.plot(range(24), e0, "-o", ms=4, lw=1.3, label="M0 历史基线")
    ax2.plot(range(24), e3, "-s", ms=4, lw=1.3, label="M3 时空动态校准")
    ax2.set_xlabel("小时")
    ax2.set_ylabel("MAE")
    ax2.set_title("分小时误差（两日合并）")
    ax2.set_xticks(range(0, 24, 4))
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    pe0 = np.abs(m0 - truth).reshape(2, 13, 24, 8).mean(axis=(0, 2, 3))
    pe3 = np.abs(m3 - truth).reshape(2, 13, 24, 8).mean(axis=(0, 2, 3))
    x = np.arange(13)
    ax3.bar(x - 0.18, pe0, 0.36, label="M0 历史基线")
    ax3.bar(x + 0.18, pe3, 0.36, label="M3 时空动态校准")
    ax3.set_xticks(x, data.hidden, rotation=45, fontsize=8)
    ax3.set_ylabel("MAE")
    ax3.set_title("分交叉口误差（两日合并）")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig11_误差分析.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 主流程
def main() -> None:
    df = load_data()
    data = DayData(df)
    records = []  # 汇总指标表

    print("=" * 66)
    print("第一层：空间迁移能力验证（日内 5 点观测 → 13 点预测）")
    print("=" * 66)
    level1 = {}
    for day, other in [("D1", "D2"), ("D2", "D1")]:
        level1[day] = eval_spatial(data, day, data.sim[other],
                                   h=0.2, gamma=2.0)
        for m, v in level1[day].items():
            records.append({"验证层次": "第一层·空间迁移", "目标日": day,
                            "参考日": "-", "模型": m, **v})
        print(f"  目标日 {day}: " + "  ".join(
            f"{m}: MAE={v['MAE']:.5f} RMSE={v['RMSE']:.5f} JSD={v['JSD']:.5f}"
            for m, v in level1[day].items()))

    print("\n" + "=" * 66)
    print("第二层：跨日动态校准验证（D1、D2 互为参考日与目标日）")
    print("=" * 66)
    level2, arrays = {}, {}
    for target, ref in [("D1", "D2"), ("D2", "D1")]:
        m, a = eval_calibration(data, target, ref, h=0.2, gamma=2.0, lam=0.5)
        level2[target] = {**m}
        arrays[target] = a
        # M1/M2 不依赖参考日，直接取第一层结果并入对照
        for base_m in ["M1", "M2"]:
            level2[target][base_m] = level1[target][base_m]
        for m in ["M0", "M1", "M2", "M3d", "M3"]:
            records.append({"验证层次": "第二层·跨日校准", "目标日": target,
                            "参考日": ref, "模型": m, **level2[target][m]})
        print(f"  目标日 {target}（参考日 {ref}）:")
        for m in ["M0", "M1", "M2", "M3d", "M3"]:
            v = level2[target][m]
            print(f"    {m:<4}: MAE={v['MAE']:.5f}  RMSE={v['RMSE']:.5f}  "
                  f"JSD={v['JSD']:.5f}")

    avg = {m: {k: np.mean([level2["D1"][m][k], level2["D2"][m][k]])
               for k in ["MAE", "RMSE", "JSD"]} for m in ["M0", "M3d", "M3"]}
    print("\n  两日平均（校准类模型）:")
    for m, v in avg.items():
        print(f"    {m:<4}: MAE={v['MAE']:.5f}  RMSE={v['RMSE']:.5f}  "
              f"JSD={v['JSD']:.5f}")
    imp = {k: (1 - avg["M3"][k] / avg["M0"][k]) * 100 for k in avg["M3"]}
    print(f"  M3 相对 M0: MAE↓{imp['MAE']:.1f}%  RMSE↓{imp['RMSE']:.1f}%  "
          f"JSD↓{imp['JSD']:.1f}%")

    print("\n" + "=" * 66)
    print("参数网格搜索（h × γ × λ，跨日两日平均，按 MAE 排序前 10）")
    print("=" * 66)
    grid = grid_search(data)
    grid.to_csv(OUT_DIR / "grid_search.csv", index=False, encoding="utf-8-sig")
    print(grid.sort_values("MAE").head(10).to_string(
        index=False, float_format=lambda x: f"{x:.5f}"))
    best = grid.sort_values("MAE").iloc[0]
    print(f"\n  最优参数: h={best['h']}, γ={best['gamma']}, λ={best['lambda']} "
          f"→ 用于 02_prediction.py 的最终模型")

    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(OUT_DIR / "validation_metrics.csv",
                      index=False, encoding="utf-8-sig")

    fig9_model_compare(level1, level2)
    fig10_grid(grid)
    fig11_error_analysis(arrays["D1"], arrays["D2"], data)

    print("\n  D1(工作日)/D2(非工作日)跨日验证存在日类型差异，")
    print("        真实任务中 D3 与基线日 D1 同为工作日，误差应低于上述结果；")
    print("        因此跨日验证指标是 D3 预测误差的保守上界。")
    print(f"\n  指标表: {OUT_DIR / 'validation_metrics.csv'}")
    print(f"  网格表: {OUT_DIR / 'grid_search.csv'}")
    print(f"  图件目录: {FIG_DIR}")


if __name__ == "__main__":
    main()
