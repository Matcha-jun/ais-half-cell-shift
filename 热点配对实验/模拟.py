import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np


TRIALS = 2_000
WIDTH = 10
ORIGINS = np.array(list(product((0, 2.5, 5, 7.5), repeat=2)))
VOTE_LIMITS = (8, 12, 16)
ABLATION = "配对，无检查"
METHODS = ("旧方法", "固定窗口质心", ABLATION, *(f"配对，{n}票" for n in VOTE_LIMITS))
SCENARIOS = (
    ("无干扰单峰", 0),
    *product(("双峰，强弱不变", "双峰，强弱互换"), (15, 20, 30)),
)
LABELS = ("找对未报东移", "找对报东移", "错认未报东移", "错认报东移",
          "定位失败未报东移", "定位失败报东移", "不确定或未匹配")
PAIR_SUMMARIES = ("挡下错认", "增加未作答", "损失正确检出")
# 每个方案的两个值分别是两枚基期候选的去向。-1 表示未配对。
ASSIGNMENTS = ((-1, -1), (0, -1), (1, -1), (-1, 0), (-1, 1), (0, 1), (1, 0))
OUT = Path(__file__).resolve().parent


def generate(rng, scenario, spacing, displacement):
    anchors = rng.uniform(0, WIDTH, (TRIALS, 2))
    counts = (120, 0) if not spacing else (120, 80)
    later_counts = counts[::-1] if scenario == "双峰，强弱互换" else counts
    first_centers = np.stack((anchors, anchors + (spacing, 0)), axis=1)
    second_centers = first_centers.copy()
    second_centers[:, 0] += (displacement, 0)
    peak_count = 2 if spacing else 1

    def sample(centers, ship_counts):
        points = np.concatenate([
            centers[:, index, None, :] + rng.normal(0, 3.5, (TRIALS, ship_counts[index], 2))
            for index in range(peak_count)
        ], axis=1)
        return rng.permuted(points, axis=1)

    first = sample(first_centers, counts)
    second = sample(second_centers, later_counts)
    # 评分数据与方法输入分开，数组第 i 行对应轮次 i。
    return (first, second, anchors), second_centers[:, :peak_count]


def grid_peaks(points, origin):
    cells = np.floor((points - origin) / WIDTH).astype(int)
    lower = cells.min(axis=(0, 1))
    nx, ny = cells.max(axis=(0, 1)) - lower + 1
    shifted = cells - lower
    offsets = np.arange(len(points))[:, None] * nx * ny
    ids = shifted[..., 0] * ny + shifted[..., 1] + offsets
    counts = np.bincount(ids.ravel(), minlength=len(points) * nx * ny)
    counts = counts.reshape(len(points), nx * ny)

    def centers(ids):
        xy = np.column_stack((ids // ny, ids % ny)) + lower
        return origin + (xy + 0.5) * WIDTH

    index = counts.argmax(axis=1)
    hottest = centers(index)
    bins = np.arange(nx * ny)
    neighbors = (np.abs(index[:, None] // ny - bins // ny) <= 1) & (
        np.abs(index[:, None] % ny - bins % ny) <= 1
    )
    runner_counts = np.where(neighbors, 0, counts)
    runner_index = runner_counts.argmax(axis=1)
    rows = np.arange(len(points))
    candidates = np.stack((hottest, centers(runner_index)), axis=1)
    candidates[:, 0][counts[rows, index] < 10] = np.nan
    candidates[:, 1][runner_counts[rows, runner_index] < 10] = np.nan
    return hottest, candidates


def match_peaks(first, second, anchors, limit):
    rows = np.arange(len(first))
    anchor_distance = np.linalg.norm(first - anchors[:, None, :], axis=2)
    anchor_distance = np.where(np.isnan(anchor_distance), np.inf, anchor_distance)
    target = anchor_distance.argmin(axis=1)
    has_target = anchor_distance[rows, target] <= 10
    allowed = np.linalg.norm(first[:, :, None] - second[:, None, :], axis=3) <= limit

    sizes = np.full((len(first), len(ASSIGNMENTS)), -1)
    for k, assignment in enumerate(ASSIGNMENTS):
        links = [(source, dest) for source, dest in enumerate(assignment) if dest >= 0]
        valid = np.ones(len(first), dtype=bool)
        for source, dest in links:
            valid &= allowed[:, source, dest]
        sizes[valid, k] = len(links)
    best = sizes == sizes.max(axis=1, keepdims=True)
    target_destinations = np.array(ASSIGNMENTS).T[target]
    locations = np.full((len(first), 2), np.nan)
    for dest in range(2):
        agreed = has_target & np.all(~best | (target_destinations == dest), axis=1)
        locations[agreed] = second[agreed, dest]
    displacement = locations[:, 0] - first[rows, target, 0]
    return np.column_stack((locations, displacement))


def predict(first, second, anchors, limit):
    hottest = []
    matches = []
    for origin in ORIGINS:
        first_hot, first_peaks = grid_peaks(first, origin)
        second_hot, second_peaks = grid_peaks(second, origin)
        hottest.append(np.column_stack((second_hot, second_hot[:, 0] - first_hot[:, 0])))
        matches.append(match_peaks(first_peaks, second_peaks, anchors, limit))
    old = np.median(hottest, axis=0)
    matched = np.stack(matches, axis=1)
    present = np.isfinite(matched[:, :, 2]).any(axis=1)
    aggregate = np.full((len(first), 3), np.nan)
    aggregate[present] = np.nanmedian(matched[present], axis=1)
    coordinate_gaps = np.abs(matched[:, :, :2] - aggregate[:, None, :2])
    votes = (coordinate_gaps <= 5).all(axis=2).sum(axis=1)

    def centroid(points):
        inside = np.linalg.norm(points - anchors[:, None, :], axis=2) <= 20
        count = inside.sum(axis=1, keepdims=True)
        sums = np.where(inside[:, :, None], points, 0).sum(axis=1)
        return np.divide(sums, count, out=np.full_like(sums, np.nan), where=count > 0)

    first_center = centroid(first)
    second_center = centroid(second)
    local_mean = np.column_stack((second_center, second_center[:, 0] - first_center[:, 0]))
    local_mean[~np.isfinite(local_mean).all(axis=1)] = np.nan
    methods = dict(zip(METHODS, (old, local_mean, aggregate, *(
        np.where((votes >= n)[:, None], aggregate, np.nan) for n in VOTE_LIMITS
    ))))
    return methods, votes, coordinate_gaps


def score(prediction, centers, east):
    answered = np.isfinite(prediction[:, 2])
    distances = np.linalg.norm(prediction[:, None, :2] - centers, axis=2)
    nearest = distances.min(axis=1, keepdims=True)
    close = (distances == nearest) & (distances <= 7.5)
    identity = np.where(close.sum(axis=1) == 1, close.argmax(axis=1), 2)
    return np.where(answered, 2 * identity + east, 6)


def checks():
    first = np.repeat(np.array([[[1., 1.]]]), 120, axis=1)
    second = first + (5, 0)
    predictions, votes, gaps = predict(first, second, np.array([[1., 1.]]), 15)
    assert votes.tolist() == [16]
    np.testing.assert_array_equal(gaps.max(axis=(0, 1)), [3.75, 3.75])
    assert all(predictions[name][0, 2] == 5 for name in METHODS[2:])

    first = np.repeat(np.array([[[0., 0.], [20., 0.]]]), 100, axis=1)
    second = np.repeat(np.array([[[-10., 0.], [10., 0.]]]), 100, axis=1)
    anchor = np.array([[0., 0.]])
    histories = {"非交叉": np.array([[[-10., 0.], [10., 0.]]]),
                 "交叉": np.array([[[10., 0.], [-10., 0.]]])}
    results = {}
    for history, truth in histories.items():
        # 两次方法调用拿到的是同一组坐标和同一个匹配上限。
        predictions, votes, _ = predict(first, second, anchor, 30)
        assert votes.tolist() == [0]
        assert all(np.isnan(predictions[name]).all() for name in METHODS[2:])
        results[history] = (predictions, truth)

    lines = ["单点：16/16 票，两个方向的最大偏差均为 3.75 km，配对位移为 5 km。",
             "交叉与非交叉：配对组在两套身份历史下均无匹配，所有方法的估计逐一相同。"]
    lines.extend(["", "| 方法 | 估计位置（km） | 东西位移（km） | 非交叉身份与判定 | 交叉身份与判定 |",
                  "| --- | ---: | ---: | --- | --- |"])
    for name in METHODS:
        reference = results["非交叉"][0][name]
        grades = []
        for predictions, truth in results.values():
            prediction = predictions[name]
            np.testing.assert_array_equal(prediction, reference)
            grades.append(LABELS[score(prediction, truth, prediction[:, 2] > 2.5)[0]])
        x, y, dx = reference[0]
        location = f"({x:g}, {y:g})" if np.isfinite(dx) else "—"
        movement = f"{dx:g}" if np.isfinite(dx) else "—"
        lines.append(f"| {name} | {location} | {movement} | {grades[0]} | {grades[1]} |")
    return "\n".join(lines)


def calibrate():
    rng = np.random.default_rng(20261119)
    null_scores = {name: [] for name in METHODS}
    for scenario, spacing in SCENARIOS:
        inputs, _ = generate(rng, scenario, spacing, 0)
        predictions, _, _ = predict(*inputs, 15)
        for name, values in predictions.items():
            null_scores[name].append(values[:, 2])

    rules = {}
    target_count = len(SCENARIOS) * TRIALS * 0.05
    for name, batches in null_scores.items():
        scores = np.concatenate(batches)
        valid = np.sort(scores[np.isfinite(scores)])[::-1]
        if len(valid) < target_count:
            rules[name] = None
            continue
        threshold = valid[int(np.ceil(target_count)) - 1]
        ahead = (valid > threshold).sum()
        tied = (valid == threshold).sum()
        rules[name] = (threshold, (target_count - ahead) / tied)
    return rules


def write_csv(filename, headers, rows):
    with (OUT / filename).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def rate(condition):
    return round(100 * np.mean(condition), 6)


def paired_change(new, old):
    diff = old.astype(float) - new
    mean = diff.mean()
    margin = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
    return [int(diff.sum()), *[round(100 * value, 6) for value in (mean, mean - margin, mean + margin)]]


def run():
    rules = calibrate()
    rng = np.random.default_rng(20261120)
    tie_rng = np.random.default_rng(20261121)
    grades = []
    comparisons = []
    supplemental = []
    for scenario, spacing in SCENARIOS:
        for truth in (0, 5):
            # 没有评分真值传入预测步骤，各门槛共用同一份估计。
            inputs, centers = generate(rng, scenario, spacing, truth)
            predictions, _, _ = predict(*inputs, 15)
            tie_draws = tie_rng.random(TRIALS)
            key = [scenario, spacing if spacing else "", truth]
            categories = {}
            for name, prediction in predictions.items():
                dx = prediction[:, 2]
                east = dx > 2.5
                grade = score(prediction, centers, east)
                categories[name] = grade
                counts = np.bincount(grade, minlength=len(LABELS))
                right = grade < 2
                error = np.abs(dx[right] - truth).mean() if right.any() else ""
                grades.append([*key, name, TRIALS, *counts,
                               *[round(100 * count / TRIALS, 6) for count in counts],
                               rate(east) if truth == 0 else "",
                               rate(grade == 1) if truth else "",
                               round(float(error), 6) if right.any() else "", int(right.sum())])

                rule = rules[name]
                if rule is None:
                    supplemental.append([*key, name, TRIALS, "不可达", "", "", "", ""])
                    continue
                threshold, probability = rule
                east = (dx > threshold) | ((dx == threshold) & (tie_draws < probability))
                supplemental.append([*key, name, TRIALS, "可达", threshold, probability,
                                     rate(east) if truth == 0 else "",
                                     rate(right & east) if truth else ""])

            old = categories[ABLATION]
            for n in VOTE_LIMITS:
                name = f"配对，{n}票"
                new = categories[name]
                assert np.all((new == 6) | (new == old))
                wrong = paired_change((new == 2) | (new == 3), (old == 2) | (old == 3))
                unanswered = paired_change(old == 6, new == 6)
                lost = paired_change(new == 1, old == 1) if truth else [""] * 4
                comparisons.append([*key, name, TRIALS, *wrong, *unanswered, *lost])

    headings = ["场景", "双峰基期中心间距（km）", "原热点真实东西位移（km）", "方法", "轮次"]
    write_csv("成绩单.csv",
              [*headings, *LABELS, *(f"{label}（%）" for label in LABELS),
               "误报率（%）", "正确检出率（%）", "找对轮次位移平均绝对误差（km）", "误差分母"],
              grades)
    write_csv("消融比较.csv", [
        *headings,
        *(f"{metric}{stat}" for metric in PAIR_SUMMARIES
          for stat in ("（轮）", "（百分点）", "区间下限（百分点）", "区间上限（百分点）")),
    ], comparisons)
    for truth, metric_column in ((0, -2), (5, -1)):
        for name, rule in rules.items():
            key = ["主场景等权汇总", "", truth, name, TRIALS * len(SCENARIOS)]
            if rule is None:
                supplemental.append([*key, "不可达", "", "", "", ""])
                continue
            values = [row[metric_column] for row in supplemental
                      if row[0] != "主场景等权汇总" and row[2] == truth and row[3] == name]
            mean = round(float(np.mean(values)), 6)
            metrics = ["", ""]
            metrics[metric_column] = mean
            supplemental.append([*key, "可达", *rule, *metrics])
    write_csv("校准补充.csv", [
        *headings, "校准5%是否可达", "位移阈值（km）", "并列时报告概率",
        "实测误报率（%）", "实测正确检出率（%）",
    ], supplemental)


def main():
    parser = argparse.ArgumentParser(description="复现热点配对预注册实验")
    parser.add_argument("--check-only", action="store_true", help="只验证两个确定性算例")
    args = parser.parse_args()
    diagnostics = checks()
    print(diagnostics, flush=True)
    if not args.check_only:
        run()


if __name__ == "__main__":
    main()
