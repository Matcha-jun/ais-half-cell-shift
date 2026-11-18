import csv
from pathlib import Path

import numpy as np


GRID_WIDTH = 10
HALF_CELL = GRID_WIDTH / 2
TRIALS = 2_000
SEED = 20261118
ORIGINS = np.array(
    [(x, y) for x in (0, 2.5, 5, 7.5) for y in (0, 2.5, 5, 7.5)]
)
SCENARIOS = (
    ("主场景", 200, 3.5),
    ("少船", 50, 3.5),
    ("多船", 800, 3.5),
    ("热点集中", 200, 1.5),
    ("热点分散", 200, 6.5),
)


def hottest_x(points, origin):
    """以各轮为单位计数。按东西、南北坐标升序编号，保证并列时取首格。"""
    cells = np.floor((points - origin) / GRID_WIDTH).astype(np.int64)
    lower_x, lower_y = cells.min(axis=(0, 1))
    upper_x, upper_y = cells.max(axis=(0, 1))
    x_count = upper_x - lower_x + 1
    y_count = upper_y - lower_y + 1
    cell_count = x_count * y_count
    cell_ids = (cells[..., 0] - lower_x) * y_count + cells[..., 1] - lower_y
    trial_offsets = np.arange(len(points))[:, None] * cell_count
    counts = np.bincount(
        (cell_ids + trial_offsets).ravel(), minlength=len(points) * cell_count
    ).reshape(len(points), cell_count)
    x_index = counts.argmax(axis=1) // y_count + lower_x
    return origin[0] + (x_index + 0.5) * GRID_WIDTH


def hottest_x_by_origin(points):
    return np.column_stack([hottest_x(points, origin) for origin in ORIGINS])


def summary_row(scenario, ship_count, sigma, metric, baseline, multiple, scale):
    delta = (multiple - baseline) * scale
    margin = 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))
    return (
        scenario, ship_count, sigma, metric,
        round(baseline.mean() * scale, 6), round(multiple.mean() * scale, 6),
        round(delta.mean(), 6), round(delta.mean() - margin, 6),
        round(delta.mean() + margin, 6),
    )


def main():
    point = np.array([[[1.0, 1.0]]])
    point_x = hottest_x_by_origin(point)
    assert np.all(hottest_x_by_origin(point.copy()) - point_x == 0)
    shifted_point = point + (HALF_CELL, 0)
    point_displacement = hottest_x_by_origin(shifted_point) - point_x
    assert np.count_nonzero(point_displacement == 0) == 8
    assert np.count_nonzero(point_displacement == GRID_WIDTH) == 8
    assert np.median(point_displacement) == HALF_CELL

    rng = np.random.default_rng(SEED)
    rows = []
    for scenario, ship_count, sigma in SCENARIOS:
        center = rng.uniform(0, GRID_WIDTH, size=(TRIALS, 1, 2))
        first = center + rng.normal(0, sigma, size=(TRIALS, ship_count, 2))
        second = center + rng.normal(0, sigma, size=(TRIALS, ship_count, 2))
        first_x = hottest_x_by_origin(first)

        for truth, rate_label in ((0, "误报率（%）"), (HALF_CELL, "检出率（%）")):
            displacement = hottest_x_by_origin(second + (truth, 0)) - first_x
            median = np.median(displacement, axis=1)
            rows.append(summary_row(
                scenario, ship_count, sigma, rate_label,
                (displacement >= HALF_CELL).mean(axis=1), median >= HALF_CELL, 100,
            ))
            rows.append(summary_row(
                scenario, ship_count, sigma, f"真值{truth:g} km的平均绝对误差（km）",
                np.abs(displacement - truth).mean(axis=1), np.abs(median - truth), 1,
            ))

        if scenario == "主场景":
            same_origin = (hottest_x_by_origin(first.copy()) - first_x >= HALF_CELL)
            shifted_x = np.column_stack([
                hottest_x(first, origin + (HALF_CELL, 0)) for origin in ORIGINS
            ])
            changed_origin = shifted_x - first_x >= HALF_CELL
            assert not same_origin.any()
            print(f"同一坐标复用、同一网格的误报率：{same_origin.mean():.2%}")
            print(f"同一坐标复用、第二期网格东移半格的误报率：{changed_origin.mean():.2%}")

    output = Path(__file__).with_name("汇总数据.csv")
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "场景", "每期船数", "空间标准差（km）", "指标", "单网格", "多起点",
            "配对差", "95%区间下限", "95%区间上限",
        ])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
