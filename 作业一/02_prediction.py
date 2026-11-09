# -*- coding: utf-8 -*-
"""
02_prediction.py
预测 D3 目标日 13 个未知交叉口 24 小时的 8 类车型占比（312 条记录）

模型（与 03_validation.py 内部验证保持一致）：
    M0 基线：D1 历史同时段占比        p_hat(i,t) = p_D1(i,t)
    M3 最终：时空动态校准模型
        1) 基线：D1 与 D3 同为工作日，取 D1 同时段占比作为基线；
        2) 目标日变化：D3 已知 5 点逐小时占比相对 D1 的差异 Δ(j,t)；
        3) 空间传递：距离核 exp(-d/h) 与历史组成相似性 exp(γ·sim) 的乘积权重；
        4) 修正：p_hat = p_D1 + λ·Σ w_j·Δ(j,t)，再做非负截断与归一化。
    超参数 h=0.2, γ=2.0, λ=0.5 由 03_validation.py 的跨日网格搜索确定。

输出：
    outputs/predictions_d3.csv      312 条预测记录（含行和自检列）
    outputs/submission_filled.xlsx  按 submission_template.xlsx 填写好的提交文件
    figures/fig7_预测示例.png、fig8_修正强度空间分布.png

"""

# ---------------------------------------------------------------- copy自01_data_analysis.py
import re
import zipfile
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
TEMPLATE_PATH = BASE_DIR / "submission_template.xlsx"
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
PRED_COLS = [c.replace("count_", "pred_share_") for c in COUNT_COLS]
CLASS_NAMES = ["小型新能源客车", "小型新能源货车", "小型燃油客车", "小型燃油货车",
               "大型新能源客车", "大型新能源货车", "大型燃油客车", "大型燃油货车"]

KNOWN_POINTS = ["J03", "J04", "J06", "J10", "J13"]

# 最终模型超参数（由 03_validation.py 跨日网格搜索确定）
H_KERNEL = 0.2    # 距离核带宽
GAMMA_SIM = 2.0   # 历史相似性权重
LAMBDA_CAL = 0.5  # 目标日变化校准系数
EPS = 1e-9        # 非负截断下界


# ---------------------------------------------------------------- 数据工具
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    for c in COUNT_COLS:
        df[c.replace("count_", "share_")] = df[c] / df["total_count"]
    return df


def share_array(df: pd.DataFrame, day: str, points: list[str]) -> np.ndarray:
    """指定日期、指定交叉口的占比数组，形状 (len(points), 24, 8)。"""
    idx = pd.MultiIndex.from_product([points, range(24)], names=["point_id", "hour"])
    arr = (df[df["day_id"] == day]
           .set_index(["point_id", "hour"])[SHARE_COLS]
           .reindex(idx).to_numpy())
    return arr.reshape(len(points), 24, len(SHARE_COLS))


def get_distance_matrix(df: pd.DataFrame, points: list[str]) -> pd.DataFrame:
    coords = df.groupby("point_id")[["coord_x", "coord_y"]].first()
    xy = coords.loc[points].to_numpy()
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
    return pd.DataFrame(d, index=points, columns=points)


# ---------------------------------------------------------------- 模型
def baseline_d1(arr_d1: np.ndarray, all_points: list[str],
                hidden: list[str]) -> dict[tuple[str, int], np.ndarray]:
    """M0：D1 历史同时段占比基线。"""
    return {(p, t): arr_d1[all_points.index(p), t] for p in hidden for t in range(24)}


def spatiotemporal_calibration(df: pd.DataFrame, all_points: list[str],
                               hidden: list[str],
                               h: float = H_KERNEL, gamma: float = GAMMA_SIM,
                               lam: float = LAMBDA_CAL) -> tuple[dict, pd.DataFrame]:
    """M3：D1 基线 + D3 已知点变化量的 距离×相似性 加权校准。"""
    arr_d1 = share_array(df, "D1", all_points)
    arr_d3k = share_array(df, "D3", KNOWN_POINTS)          # (5, 24, 8)
    arr_d1k = share_array(df, "D1", KNOWN_POINTS)          # (5, 24, 8)
    delta_known = arr_d3k - arr_d1k                        # 目标日变化 (5, 24, 8)

    # 历史组成相似性：由历史参考日 D1 的 24h×8类 剖面相关得到
    sim = np.corrcoef(arr_d1.reshape(len(all_points), -1))
    dist = get_distance_matrix(df, all_points)

    preds, weight_rows = {}, []
    for p in hidden:
        i = all_points.index(p)
        d_ik = dist.loc[p, KNOWN_POINTS].to_numpy()
        s_ik = np.maximum(sim[i, [all_points.index(j) for j in KNOWN_POINTS]], 0.0)
        w = np.exp(-d_ik / h) * np.exp(gamma * s_ik)
        w = w / w.sum()
        weight_rows.append(dict(zip(KNOWN_POINTS, w.round(4)), point_id=p,
                                最近已知点=KNOWN_POINTS[int(np.argmax(w))]))
        for t in range(24):
            base = arr_d1[i, t]
            p_hat = base + lam * (w @ delta_known[:, t, :])
            p_hat = np.maximum(p_hat, EPS)                 # 非负约束
            p_hat = p_hat / p_hat.sum()                    # 归一化
            preds[(p, t)] = p_hat
    return preds, pd.DataFrame(weight_rows).set_index("point_id")


# ---------------------------------------------------------------- 输出
def build_prediction_frame(preds: dict[tuple[str, int], np.ndarray],
                           hidden: list[str]) -> pd.DataFrame:
    rows = []
    for p in hidden:
        for t in range(24):
            row = {"day_id": "D3", "hour": t, "point_id": p}
            row.update(dict(zip(PRED_COLS, preds[(p, t)])))
            rows.append(row)
    out = pd.DataFrame(rows)
    out["row_sum"] = out[PRED_COLS].sum(axis=1)
    return out

# -> None没有return
def check_predictions(pred_df: pd.DataFrame, hidden: list[str]) -> None:
    """输出约束核验：312 条、关键字唯一、占比∈[0,1]、行和=1。"""
    print("\n  输出约束核验:")
    n = len(pred_df)
    dup = pred_df.duplicated(["day_id", "point_id", "hour"]).sum()
    lo = pred_df[PRED_COLS].min().min()
    hi = pred_df[PRED_COLS].max().max()
    dev = (pred_df["row_sum"] - 1).abs().max()
    points_ok = sorted(pred_df["point_id"].unique()) == sorted(hidden)
    print(f"    记录数 = {n}（要求 312）: {'通过' if n == 312 else '不通过'}")
    print(f"    重复关键字 = {dup}: {'通过' if dup == 0 else '不通过'}")
    print(f"    待预测点覆盖正确: {'通过' if points_ok else '不通过'}")
    print(f"    占比范围 [{lo:.6f}, {hi:.6f}] ⊆ [0,1]: "
          f"{'通过' if lo >= 0 and hi <= 1 else '不通过'}")
    print(f"    行和最大偏差 = {dev:.2e}（容差 1e-3）: {'通过' if dev <= 1e-3 else '不通过'}")
    assert n == 312 and dup == 0 and lo >= 0 and hi <= 1 and dev <= 1e-3 and points_ok


def fill_submission_template(pred_df: pd.DataFrame) -> Path:
    """不依赖 openpyxl，直接改写模板 XML：仅填充 D-K 列并同步公式缓存值。"""
    key2val = {(r["point_id"], int(r["hour"])): [f"{v:.10f}" for v in r[PRED_COLS]]
               for _, r in pred_df.iterrows()}
    col_letters = ["D", "E", "F", "G", "H", "I", "J", "K"]

    with zipfile.ZipFile(TEMPLATE_PATH) as zin:
        items = {n: zin.read(n) for n in zin.namelist()}

    sheet = items["xl/worksheets/sheet2.xml"].decode("utf-8")

    def fill_row(m: re.Match) -> str:
        row_xml = m.group(0)
        r = m.group(1)
        if r == "1":  # 表头行不动
            return row_xml
        point = re.search(rf'<x:c r="C{r}"[^>]*><x:v>([^<]+)</x:v>', row_xml).group(1)
        hour = int(re.search(rf'<x:c r="B{r}"[^>]*><x:v>(\d+)</x:v>', row_xml).group(1))
        vals = key2val[(point, hour)]
        for col, v in zip(col_letters, vals):
            row_xml = row_xml.replace(
                f'<x:c r="{col}{r}" s="18" />',
                f'<x:c r="{col}{r}" s="18" t="n"><x:v>{v}</x:v></x:c>')
        row_sum = f"{sum(float(v) for v in vals):.10f}".rstrip("0").rstrip(".")
        row_xml = row_xml.replace(
            f'<x:f>SUM(D{r}:K{r})</x:f><x:v>0</x:v>',
            f'<x:f>SUM(D{r}:K{r})</x:f><x:v>{row_sum}</x:v>')
        row_xml = row_xml.replace("<x:v>未填写</x:v>", "<x:v>通过</x:v>")
        return row_xml

    sheet = re.sub(r'<x:row r="(\d+)"[^>]*>.*?</x:row>', fill_row, sheet,
                   flags=re.DOTALL)
    items["xl/worksheets/sheet2.xml"] = sheet.encode("utf-8")

    # sheet3 提交前检查：同步缓存值
    sheet3 = items["xl/worksheets/sheet3.xml"].decode("utf-8")
    max_dev = float((pred_df["row_sum"] - 1).abs().max())
    for ref, new_v in [("B5", "0"), ("B6", "0"),
                       ("B8", f"{max_dev:.2e}"), ("B9", "可以提交")]:
        sheet3 = re.sub(rf'(<x:c r="{ref}"[^>]*>.*?<x:v>)[^<]*(</x:v>)',
                        rf"\g<1>{new_v}\g<2>", sheet3, flags=re.DOTALL)
    items["xl/worksheets/sheet3.xml"] = sheet3.encode("utf-8")

    # 打开文件时强制重算公式，避免任何缓存残留
    wb = items["xl/workbook.xml"].decode("utf-8")
    if "calcPr" not in wb:
        wb = wb.replace("</x:workbook>",
                        '<x:calcPr fullCalcOnLoad="1"/></x:workbook>')
    items["xl/workbook.xml"] = wb.encode("utf-8")

    out_path = OUT_DIR / "submission_filled.xlsx"
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in items.items():
            zout.writestr(name, data)
    return out_path


# ---------------------------------------------------------------- 可视化
def fig7_examples(pred_df: pd.DataFrame, base_preds: dict, hidden: list[str],
                  df: pd.DataFrame) -> None:
    """距已知点最近/最远的两个待预测点：基线 vs 最终预测的逐小时曲线。"""
    dist = get_distance_matrix(df, sorted(df["point_id"].unique()))
    min_d = dist.loc[hidden, KNOWN_POINTS].min(axis=1)
    samples = [min_d.idxmin(), min_d.idxmax()]
    show = [(0, "小型新能源客车"), (2, "小型燃油客车")]  # 占比最大的两类

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, p in zip(axes, samples):
        for k, name in show:
            base_v = [base_preds[(p, t)][k] for t in range(24)]
            pred_v = pred_df[pred_df["point_id"] == p].sort_values("hour")[PRED_COLS[k]]
            ax.plot(range(24), base_v, "--", lw=1.3,
                    label=f"{CLASS_NAMES[k]}·D1基线")
            ax.plot(range(24), pred_v, "-", lw=1.8,
                    label=f"{CLASS_NAMES[k]}·D3预测")
        tag = "距已知点最近" if p == samples[0] else "距已知点最远"
        ax.set_title(f"{p}（{tag}，d={min_d[p]:.2f}）")
        ax.set_xlabel("小时")
        ax.set_xticks(range(0, 24, 4))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("占比")
    axes[0].legend(fontsize=8)
    fig.suptitle("最终预测 vs D1 历史基线（示例待预测点）")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_预测示例.png", dpi=150)
    plt.close(fig)


def fig8_correction_map(pred_df: pd.DataFrame, base_preds: dict,
                        hidden: list[str], df: pd.DataFrame) -> None:
    """各待预测点平均修正强度（预测相对 D1 基线的平均 L1 变化）的空间分布。"""
    coords = df.groupby("point_id")[["coord_x", "coord_y"]].first()
    strength = {}
    for p in hidden:
        pred = pred_df[pred_df["point_id"] == p].sort_values("hour")[PRED_COLS].to_numpy()
        base = np.array([base_preds[(p, t)] for t in range(24)])
        strength[p] = float(np.abs(pred - base).sum(axis=1).mean())

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter([coords.loc[p, "coord_x"] for p in hidden],
                    [coords.loc[p, "coord_y"] for p in hidden],
                    c=[strength[p] for p in hidden], cmap="Reds",
                    s=160, edgecolors="k", linewidths=0.6, zorder=3)
    ax.scatter(coords.loc[KNOWN_POINTS, "coord_x"], coords.loc[KNOWN_POINTS, "coord_y"],
               marker="s", s=160, c="#1f77b4", label="D3已知交叉口",
               edgecolors="k", linewidths=0.6, zorder=3)
    for p in hidden + KNOWN_POINTS:
        ax.annotate(p, (coords.loc[p, "coord_x"], coords.loc[p, "coord_y"]),
                    textcoords="offset points", xytext=(8, 8), fontsize=10)
    fig.colorbar(sc, ax=ax, shrink=0.8, label="平均修正强度（L1）")
    ax.set_xlabel("相对空间坐标 x")
    ax.set_ylabel("相对空间坐标 y")
    ax.set_title("D3 预测相对 D1 基线的平均修正强度空间分布")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_修正强度分布.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 主流程
def main() -> None:
    df = load_data()
    all_points = sorted(df["point_id"].unique())
    hidden = [p for p in all_points if p not in KNOWN_POINTS]

    print("=" * 66)
    print("模型配置")
    print("=" * 66)
    print(f"  M0 基线: D1 历史同时段占比")
    print(f"  M3 最终: D1 基线 + D3已知5点变化校准 "
          f"(h={H_KERNEL}, γ={GAMMA_SIM}, λ={LAMBDA_CAL})")

    arr_d1 = share_array(df, "D1", all_points)
    base_preds = baseline_d1(arr_d1, all_points, hidden)
    preds, weights = spatiotemporal_calibration(df, all_points, hidden)

    print("\n  各待预测点的已知点权重（行和=1）:")
    print(weights.to_string(float_format=lambda x: f"{x:.3f}"))

    pred_df = build_prediction_frame(preds, hidden)
    check_predictions(pred_df, hidden)

    csv_path = OUT_DIR / "predictions_d3.csv"
    pred_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    xlsx_path = fill_submission_template(pred_df)

    base_arr = np.array([base_preds[(p, t)] for p in hidden for t in range(24)])
    pred_arr = pred_df[PRED_COLS].to_numpy()
    print("\n  预测相对 D1 基线的平均绝对调整量（按车型）:")
    for name, k in zip(CLASS_NAMES, range(8)):
        print(f"    {name:<10}: {np.abs(pred_arr[:, k] - base_arr[:, k]).mean():.4f}")
    print(f"\n  预测记录: {csv_path}")
    print(f"  提交文件: {xlsx_path}（填写后请重命名xlsx）")

    fig7_examples(pred_df, base_preds, hidden, df)
    fig8_correction_map(pred_df, base_preds, hidden, df)
    print(f"  图件目录: {FIG_DIR}")


if __name__ == "__main__":
    main()
