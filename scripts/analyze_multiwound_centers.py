#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

STEP_RE = re.compile(r'_(\d+)\.vtk$')
DEFAULT_CASE_REGEX = r'^(?:test_multi_wound|cyl_dura_only_double_wound)_(\d+)t_(\d+)_week$'

T_DURA = 0.4
R_CORD = 5.0
X_CENTER = (-(R_CORD + T_DURA) + (-R_CORD + 0.01)) / 2.0
Y_CENTER = 0.0
Z1_CENTER = 5.0
DT = 0.2
W2_SAMPLE_HOURS = 24.0


def parse_case(folder: Path, case_re):
    m = case_re.match(folder.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def list_case_dirs(root: Path, case_re):
    out = []
    for d in root.iterdir():
        if d.is_dir() and parse_case(d, case_re) is not None:
            out.append(d)
    return sorted(out)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze wound-center trajectories from multi-wound VTK case folders. "
            "Supports both legacy test_multi_wound_* and cyl_dura_only_double_wound_* layouts."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory containing case folders (default: current directory).",
    )
    parser.add_argument(
        "--case-regex",
        default=DEFAULT_CASE_REGEX,
        help=(
            "Regex used to identify case folders; must capture X_t and Y_week as the first two groups. "
            f"default: {DEFAULT_CASE_REGEX}"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Directory where CSV/SVG outputs are written (default: analysis_outputs).",
    )
    return parser.parse_args()


def sorted_wound_vtks(case_dir: Path, wound_idx: int):
    files = []
    for p in case_dir.glob(f'wound_{wound_idx}_*.vtk'):
        if '_second_' in p.name:
            continue
        m = STEP_RE.search(p.name)
        if not m:
            continue
        files.append((int(m.group(1)), p))
    files.sort(key=lambda x: x[0])
    return files


def read_points_from_vtk(vtk_path: Path):
    points = []
    n_pts = None
    with vtk_path.open('r') as f:
        for line in f:
            if line.startswith('POINTS '):
                n_pts = int(line.split()[1])
                break
        if n_pts is None:
            raise RuntimeError(f'POINTS not found in {vtk_path}')

        vals = []
        need = 3 * n_pts
        while len(vals) < need:
            line = f.readline()
            if not line:
                break
            parts = line.split()
            if parts:
                vals.extend(float(x) for x in parts)

    if len(vals) < need:
        raise RuntimeError(f'Could not read all points in {vtk_path}')

    for i in range(n_pts):
        points.append((vals[3 * i], vals[3 * i + 1], vals[3 * i + 2]))
    return points


def nearest_node(points, tx, ty, tz):
    best_i = -1
    best_d2 = 1e300
    for i, (x, y, z) in enumerate(points):
        d2 = (x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i, math.sqrt(best_d2)


def read_rho_c_phi_kappa_at_node(vtk_path: Path, node_idx: int):
    n_pts = None
    with vtk_path.open('r') as f:
        for line in f:
            if line.startswith('POINT_DATA '):
                n_pts = int(line.split()[1])
                break
        if n_pts is None:
            raise RuntimeError(f'POINT_DATA not found in {vtk_path}')

        found = False
        for line in f:
            if line.startswith('SCALARS rho_c_phif_kappa'):
                found = True
                break
        if not found:
            raise RuntimeError(f'rho_c_phif_kappa not found in {vtk_path}')

        # LOOKUP_TABLE default
        _ = f.readline()

        if node_idx < 0 or node_idx >= n_pts:
            raise RuntimeError(f'Invalid node index {node_idx} for {vtk_path}')

        for i in range(n_pts):
            line = f.readline()
            if not line:
                raise RuntimeError(f'Unexpected EOF in {vtk_path}')
            if i == node_idx:
                vals = line.split()
                if len(vals) != 4:
                    raise RuntimeError(f'Bad data row for node {node_idx} in {vtk_path}')
                return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])

    raise RuntimeError(f'Node {node_idx} not found in {vtk_path}')


def read_rho_c_phi_kappa_at_nodes(vtk_path: Path, node_indices):
    wanted = sorted(set(node_indices))
    if not wanted:
        return {}
    n_pts = None
    with vtk_path.open('r') as f:
        for line in f:
            if line.startswith('POINT_DATA '):
                n_pts = int(line.split()[1])
                break
        if n_pts is None:
            raise RuntimeError(f'POINT_DATA not found in {vtk_path}')

        for idx in wanted:
            if idx < 0 or idx >= n_pts:
                raise RuntimeError(f'Invalid node index {idx} for {vtk_path}')

        found = False
        for line in f:
            if line.startswith('SCALARS rho_c_phif_kappa'):
                found = True
                break
        if not found:
            raise RuntimeError(f'rho_c_phif_kappa not found in {vtk_path}')

        _ = f.readline()  # LOOKUP_TABLE default

        out = {}
        wanted_set = set(wanted)
        for i in range(n_pts):
            line = f.readline()
            if not line:
                raise RuntimeError(f'Unexpected EOF in {vtk_path}')
            if i in wanted_set:
                vals = line.split()
                if len(vals) != 4:
                    raise RuntimeError(f'Bad data row for node {i} in {vtk_path}')
                out[i] = (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))
                if len(out) == len(wanted):
                    break

    if len(out) != len(wanted):
        missing = sorted(set(wanted) - set(out.keys()))
        raise RuntimeError(f'Missing node values {missing} in {vtk_path}')
    return out


def write_csv(path: Path, rows, fieldnames):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def rgb_tuple_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


def lerp(a, b, t):
    return a + (b - a) * t


def colormap(v, vmin, vmax):
    # simple blue->cyan->yellow->red
    if vmax <= vmin:
        return (180, 180, 180)
    t = (v - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    stops = [
        (0.0, (49, 54, 149)),
        (0.35, (69, 186, 255)),
        (0.7, (255, 220, 97)),
        (1.0, (213, 62, 79)),
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= t <= t1:
            local = (t - t0) / (t1 - t0)
            return (
                int(lerp(c0[0], c1[0], local)),
                int(lerp(c0[1], c1[1], local)),
                int(lerp(c0[2], c1[2], local)),
            )
    return stops[-1][1]


def text_color_for_bg(rgb):
    # WCAG-ish luminance heuristic for readable overlaid labels
    lum = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255.0
    return '#111111' if lum > 0.58 else '#ffffff'


def write_heatmap_svg(path: Path, title, xs, ys, grid, legend_label):
    # grid: dict[(y,x)] -> value
    vals = [v for v in grid.values() if v is not None]
    vmin = min(vals) if vals else 0.0
    vmax = max(vals) if vals else 1.0

    cell_w, cell_h = 150, 110
    left, top = 220, 160
    width = left + cell_w * len(xs) + 360
    height = top + cell_h * len(ys) + 220

    out = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    out.append('<style>text{font-family:Helvetica,Arial,sans-serif;fill:#111} .tiny{font-size:20px} .small{font-size:24px} .med{font-size:30px} .title{font-size:42px;font-weight:700}</style>')
    out.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    out.append(f'<text class="title" x="{left}" y="74">{title}</text>')

    # cells
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            v = grid.get((y, x), None)
            rgb = (220, 220, 220) if v is None else colormap(v, vmin, vmax)
            fill = rgb_tuple_to_hex(rgb)
            x0 = left + ix * cell_w
            y0 = top + (len(ys) - 1 - iy) * cell_h
            out.append(f'<rect x="{x0}" y="{y0}" width="{cell_w}" height="{cell_h}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')
            if v is not None:
                out.append(f'<text class="small" x="{x0 + cell_w/2}" y="{y0 + cell_h/2 + 8}" text-anchor="middle" fill="{text_color_for_bg(rgb)}">{v:.4g}</text>')

    # x labels
    out.append(f'<text class="med" x="{left + cell_w*len(xs)/2}" y="{top + cell_h*len(ys) + 108}" text-anchor="middle">X (wound spacing in t)</text>')
    for ix, x in enumerate(xs):
        x0 = left + ix * cell_w + cell_w / 2
        out.append(f'<text class="small" x="{x0}" y="{top + cell_h*len(ys) + 56}" text-anchor="middle">{x}</text>')

    # y labels
    out.append(f'<text class="med" transform="translate(58,{top + cell_h*len(ys)/2}) rotate(-90)" text-anchor="middle">Y (weeks healed before 2nd wound)</text>')
    for iy, y in enumerate(ys):
        y0 = top + (len(ys) - 1 - iy) * cell_h + cell_h / 2 + 8
        out.append(f'<text class="small" x="{left - 26}" y="{y0}" text-anchor="end">{y}</text>')

    # color bar
    cb_x = left + cell_w * len(xs) + 66
    cb_y = top
    cb_h = cell_h * len(ys)
    cb_w = 38
    n = 100
    for i in range(n):
        t = i / (n - 1)
        v = vmin + t * (vmax - vmin)
        fill = rgb_tuple_to_hex(colormap(v, vmin, vmax))
        yy = cb_y + (1 - t) * cb_h
        out.append(f'<rect x="{cb_x}" y="{yy}" width="{cb_w}" height="{cb_h/n + 1}" fill="{fill}" stroke="none"/>')
    out.append(f'<rect x="{cb_x}" y="{cb_y}" width="{cb_w}" height="{cb_h}" fill="none" stroke="#222" stroke-width="1.8"/>')
    out.append(f'<text class="small" x="{cb_x + cb_w + 16}" y="{cb_y + 24}">{vmax:.4g}</text>')
    out.append(f'<text class="small" x="{cb_x + cb_w + 16}" y="{cb_y + cb_h - 4}">{vmin:.4g}</text>')
    out.append(f'<text class="small" transform="translate({cb_x + cb_w + 58},{cb_y + cb_h/2}) rotate(-90)" text-anchor="middle">{legend_label}</text>')

    out.append('</svg>')
    path.write_text('\n'.join(out))


def line_colors(n):
    base = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
    ]
    return [base[i % len(base)] for i in range(n)]


def write_facets_svg(path: Path, title, ys, xs, series_map, y_label):
    # series_map[(y,x)] -> list[(t,v)]
    width, height = 2200, 1700
    margin = 100
    panel_w = 930
    panel_h = 620
    gap_x = 130
    gap_y = 140

    # global ranges
    all_t, all_v = [], []
    for key, pts in series_map.items():
        for t, v in pts:
            all_t.append(t)
            all_v.append(v)
    tmin, tmax = (min(all_t), max(all_t)) if all_t else (0, 4)
    vmin, vmax = (min(all_v), max(all_v)) if all_v else (0, 1)
    if vmax <= vmin:
        vmax = vmin + 1.0

    cols = line_colors(len(xs))
    col_by_x = {x: cols[i] for i, x in enumerate(xs)}

    def sx(t, x0):
        return x0 + 50 + (t - tmin) / (tmax - tmin + 1e-12) * (panel_w - 80)

    def sy(v, y0):
        return y0 + 30 + (1.0 - (v - vmin) / (vmax - vmin + 1e-12)) * (panel_h - 60)

    out = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    out.append('<style>text{font-family:Helvetica,Arial,sans-serif;fill:#111} .small{font-size:26px} .med{font-size:34px} .title{font-size:52px;font-weight:700}</style>')
    out.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    out.append(f'<text class="title" x="{margin}" y="74">{title}</text>')

    for i, y in enumerate(sorted(ys)):
        row = i // 2
        col = i % 2
        x0 = margin + col * (panel_w + gap_x)
        y0 = 130 + row * (panel_h + gap_y)

        # frame
        out.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="white" stroke="#222" stroke-width="2"/>')
        out.append(f'<text class="med" x="{x0 + 20}" y="{y0 + 44}">Y = {y} week</text>')

        # axes
        ax_l = x0 + 96
        ax_r = x0 + panel_w - 52
        ax_t = y0 + 56
        ax_b = y0 + panel_h - 76
        out.append(f'<line x1="{ax_l}" y1="{ax_b}" x2="{ax_r}" y2="{ax_b}" stroke="#111" stroke-width="2.5"/>')
        out.append(f'<line x1="{ax_l}" y1="{ax_b}" x2="{ax_l}" y2="{ax_t}" stroke="#111" stroke-width="2.5"/>')

        # ticks
        for k in range(5):
            tt = tmin + (tmax - tmin) * k / 4.0
            xk = sx(tt, x0)
            out.append(f'<line x1="{xk}" y1="{ax_b}" x2="{xk}" y2="{ax_b+10}" stroke="#111" stroke-width="1.8"/>')
            out.append(f'<text class="small" x="{xk}" y="{ax_b+36}" text-anchor="middle">{tt:.1f}</text>')
        for k in range(5):
            vv = vmin + (vmax - vmin) * k / 4.0
            yk = sy(vv, y0)
            out.append(f'<line x1="{ax_l-10}" y1="{yk}" x2="{ax_l}" y2="{yk}" stroke="#111" stroke-width="1.8"/>')
            out.append(f'<text class="small" x="{ax_l-16}" y="{yk+8}" text-anchor="end">{vv:.3g}</text>')
            out.append(f'<line x1="{ax_l}" y1="{yk}" x2="{ax_r}" y2="{yk}" stroke="#c7c7c7" stroke-width="1" />')

        out.append(f'<text class="small" x="{(ax_l+ax_r)/2}" y="{ax_b+66}" text-anchor="middle">Weeks after second wound</text>')
        out.append(f'<text class="small" transform="translate({ax_l-72},{(ax_t+ax_b)/2}) rotate(-90)" text-anchor="middle">{y_label}</text>')

        for x in xs:
            pts = series_map.get((y, x), [])
            if len(pts) < 2:
                continue
            c = rgb_tuple_to_hex(col_by_x[x])
            d = []
            for j, (tt, vv) in enumerate(pts):
                px = sx(tt, x0)
                py = sy(vv, y0)
                d.append(('M' if j == 0 else 'L') + f'{px:.2f},{py:.2f}')
            out.append(f'<path d="{" ".join(d)}" fill="none" stroke="{c}" stroke-width="4"/>')

    # legend
    lx, ly = width - 410, 130
    out.append(f'<rect x="{lx}" y="{ly}" width="290" height="{54 + 46*len(xs)}" fill="white" stroke="#333" stroke-width="2"/>')
    out.append(f'<text class="med" x="{lx+16}" y="{ly+38}">X spacing</text>')
    for i, x in enumerate(xs):
        yy = ly + 72 + i * 44
        c = rgb_tuple_to_hex(col_by_x[x])
        out.append(f'<line x1="{lx+16}" y1="{yy}" x2="{lx+86}" y2="{yy}" stroke="{c}" stroke-width="6"/>')
        out.append(f'<text class="small" x="{lx+100}" y="{yy+9}">{x}t</text>')

    out.append('</svg>')
    path.write_text('\n'.join(out))


def main():
    args = parse_args()
    root = args.root.resolve()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    case_re = re.compile(args.case_regex)

    rows = []
    rows_w1_during_w2 = []
    center_info = []

    case_dirs = list_case_dirs(root, case_re)
    if not case_dirs:
        raise RuntimeError(
            f"No case folders found under {root} matching regex: {args.case_regex}"
        )
    print(f'Found {len(case_dirs)} matching case folders.', flush=True)

    for case_dir in case_dirs:
        x_t, y_w = parse_case(case_dir, case_re)
        print(f'Processing {case_dir.name} ...', flush=True)

        w1_files = sorted_wound_vtks(case_dir, 1)
        w2_files = sorted_wound_vtks(case_dir, 2)
        if not w1_files or not w2_files:
            continue

        points = read_points_from_vtk(w1_files[0][1])
        node1, d1 = nearest_node(points, X_CENTER, Y_CENTER, Z1_CENTER)
        z2 = Z1_CENTER + x_t * T_DURA
        node2, d2 = nearest_node(points, X_CENTER, Y_CENTER, z2)

        center_info.append({
            'case_folder': case_dir.name,
            'X_t': x_t,
            'Y_week': y_w,
            'wound1_node': node1,
            'wound1_center_distance_mm': d1,
            'wound2_node': node2,
            'wound2_center_distance_mm': d2,
            'target_x_mm': X_CENTER,
            'target_y_mm': Y_CENTER,
            'target_z1_mm': Z1_CENTER,
            'target_z2_mm': z2,
        })

        # wound 1: final only (for X/Y effect summary)
        step1, vtk1 = w1_files[-1]
        rho, c, phi, kappa = read_rho_c_phi_kappa_at_node(vtk1, node1)
        t = step1 * DT
        rows.append({
            'case_folder': case_dir.name,
            'X_t': x_t,
            'Y_week': y_w,
            'wound': 1,
            'step_index': step1,
            'time_hours': t,
            'time_weeks_after_this_wound': t / (24.0 * 7.0),
            'rho_fibroblast': rho,
            'c_chemical': c,
            'phi_collagen': phi,
            'kappa': kappa,
            'sample_type': 'final_only',
        })

        # wound 2: full time series (requested plots)
        sampled_w2_files = []
        for step2, vtk2 in w2_files:
            hours = step2 * DT
            shifted_hours = hours - DT  # first saved frame is at t=DT
            # keep first/last and approximately daily samples for 4-week trends
            if step2 == w2_files[0][0] or step2 == w2_files[-1][0] or abs((shifted_hours / W2_SAMPLE_HOURS) - round(shifted_hours / W2_SAMPLE_HOURS)) < 1e-9:
                sampled_w2_files.append((step2, vtk2))

        for step2, vtk2 in sampled_w2_files:
            vals_by_node = read_rho_c_phi_kappa_at_nodes(vtk2, [node1, node2])
            rho, c, phi, kappa = vals_by_node[node2]
            t2 = step2 * DT
            rows.append({
                'case_folder': case_dir.name,
                'X_t': x_t,
                'Y_week': y_w,
                'wound': 2,
                'step_index': step2,
                'time_hours': t2,
                'time_weeks_after_this_wound': t2 / (24.0 * 7.0),
                'rho_fibroblast': rho,
                'c_chemical': c,
                'phi_collagen': phi,
                'kappa': kappa,
                'sample_type': 'daily_samples',
            })
            rho1, c1, phi1, kappa1 = vals_by_node[node1]
            rows_w1_during_w2.append({
                'case_folder': case_dir.name,
                'X_t': x_t,
                'Y_week': y_w,
                'step_index': step2,
                'time_hours': t2,
                'time_weeks_after_second_wound': t2 / (24.0 * 7.0),
                'rho_fibroblast': rho1,
                'c_chemical': c1,
                'phi_collagen': phi1,
                'kappa': kappa1,
                'sample_type': 'daily_samples',
            })
        print(f'  wound_2 sampled points: {len(sampled_w2_files)}', flush=True)

    rows.sort(key=lambda r: (r['X_t'], r['Y_week'], r['wound'], r['step_index']))
    if not rows:
        raise RuntimeError(
            "No samples collected. Check case folders and wound_1_/wound_2_*.vtk naming."
        )
    write_csv(outdir / 'center_samples_all_cases.csv', rows, list(rows[0].keys()))
    rows_w1_during_w2.sort(key=lambda r: (r['X_t'], r['Y_week'], r['step_index']))
    write_csv(outdir / 'wound1_center_during_wound2_samples.csv', rows_w1_during_w2, list(rows_w1_during_w2[0].keys()))

    center_info.sort(key=lambda r: (r['X_t'], r['Y_week']))
    write_csv(outdir / 'center_node_selection.csv', center_info, list(center_info[0].keys()))

    # final values per case/wound
    final_rows = []
    by_key = {}
    for r in rows:
        k = (r['X_t'], r['Y_week'], r['wound'])
        if k not in by_key or r['step_index'] > by_key[k]['step_index']:
            by_key[k] = r
    final_rows = sorted(by_key.values(), key=lambda r: (r['X_t'], r['Y_week'], r['wound']))
    write_csv(outdir / 'center_final_values_by_case.csv', final_rows, list(final_rows[0].keys()))

    xs = sorted({r['X_t'] for r in final_rows})
    ys = sorted({r['Y_week'] for r in final_rows})

    def make_grid(wound, field):
        g = {}
        for r in final_rows:
            if r['wound'] == wound:
                g[(r['Y_week'], r['X_t'])] = r[field]
        return g

    write_heatmap_svg(
        outdir / 'heatmap_final_phi_wound1.svg',
        'Final Collagen at Wound 1 Center',
        xs, ys, make_grid(1, 'phi_collagen'), 'phi (collagen)'
    )
    write_heatmap_svg(
        outdir / 'heatmap_final_rho_wound1.svg',
        'Final Fibroblast at Wound 1 Center',
        xs, ys, make_grid(1, 'rho_fibroblast'), 'rho (fibroblast)'
    )
    write_heatmap_svg(
        outdir / 'heatmap_final_phi_wound2.svg',
        'Final Collagen at Wound 2 Center',
        xs, ys, make_grid(2, 'phi_collagen'), 'phi (collagen)'
    )
    write_heatmap_svg(
        outdir / 'heatmap_final_rho_wound2.svg',
        'Final Fibroblast at Wound 2 Center',
        xs, ys, make_grid(2, 'rho_fibroblast'), 'rho (fibroblast)'
    )

    # second wound requested time plots
    series_phi = defaultdict(list)
    series_rho = defaultdict(list)
    for r in rows:
        if r['wound'] != 2:
            continue
        k = (r['Y_week'], r['X_t'])
        t = r['time_weeks_after_this_wound']
        series_phi[k].append((t, r['phi_collagen']))
        series_rho[k].append((t, r['rho_fibroblast']))
    for k in series_phi:
        series_phi[k].sort(key=lambda x: x[0])
    for k in series_rho:
        series_rho[k].sort(key=lambda x: x[0])

    write_facets_svg(
        outdir / 'wound2_center_phi_vs_time_by_XY.svg',
        'Second Wound Center: Collagen vs Time',
        ys, xs, series_phi, 'phi (collagen)'
    )
    write_facets_svg(
        outdir / 'wound2_center_rho_vs_time_by_XY.svg',
        'Second Wound Center: Fibroblast vs Time',
        ys, xs, series_rho, 'rho (fibroblast)'
    )

    # first wound center during second wound healing (effect of wound 2 on wound 1)
    series_phi_w1_during_w2 = defaultdict(list)
    series_rho_w1_during_w2 = defaultdict(list)
    for r in rows_w1_during_w2:
        k = (r['Y_week'], r['X_t'])
        t = r['time_weeks_after_second_wound']
        series_phi_w1_during_w2[k].append((t, r['phi_collagen']))
        series_rho_w1_during_w2[k].append((t, r['rho_fibroblast']))
    for k in series_phi_w1_during_w2:
        series_phi_w1_during_w2[k].sort(key=lambda x: x[0])
    for k in series_rho_w1_during_w2:
        series_rho_w1_during_w2[k].sort(key=lambda x: x[0])

    write_facets_svg(
        outdir / 'wound1_center_during_wound2_phi_vs_time_by_XY.svg',
        'First Wound Center During Second Wound Healing: Collagen vs Time',
        ys, xs, series_phi_w1_during_w2, 'phi (collagen)'
    )
    write_facets_svg(
        outdir / 'wound1_center_during_wound2_rho_vs_time_by_XY.svg',
        'First Wound Center During Second Wound Healing: Fibroblast vs Time',
        ys, xs, series_rho_w1_during_w2, 'rho (fibroblast)'
    )

    with (outdir / 'summary.txt').open('w') as f:
        f.write('Assumptions:\n')
        f.write('- Fibroblast concentration interpreted as rho (first scalar in rho_c_phif_kappa).\n')
        f.write('- c is chemical concentration (second scalar).\n')
        f.write('- wound_2 series treated as 4-week post-second-wound trajectory.\n')
        f.write('- wound_1 sampled at final time only for X/Y effect summary.\n')
        f.write('- Additional wound_1 center trajectory during wound_2 uses wound_2 output files.\n')
        f.write(f'- Case regex used: {args.case_regex}\n')
        f.write(f'- Root directory used: {root}\n')
        f.write('\nGenerated files:\n')
        for p in sorted(outdir.iterdir()):
            f.write(f'- {p.name}\n')

    print(f'Wrote outputs to {outdir.resolve()}')


if __name__ == '__main__':
    main()
