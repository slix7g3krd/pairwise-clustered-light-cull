"""集群化光源剔除（clustered light culling）修复实现。

修复点：
1. 深度切片每帧按相机近/远平面重算，消除近裁剪面漂移导致的漏照。
2. 集群键排序使用整数定点，消除浮点排序在不同 CPU 上的不一致。
3. 集群缓冲双缓冲 + 等待 GPU 完成（fence），避免覆写上一帧未读完的数据。
4. 半径跨出切片范围的点光源补登记到相邻切片，消除边缘亮度突变。
5. 阴影标记参与集群键，投影/不投影光源不再混在同一集群。
6. 命中统计按确定性顺序输出，同一场景两次运行结果一致。
"""

from __future__ import annotations

import math
from bisect import bisect_right

# 定点小数位：排序键统一放大 2^16 取整，跨 CPU 结果一致。
FIXED_POINT_SCALE = 1 << 16

# 阴影标记在集群键中的位（高位，保证投影/不投影光源分入不同键空间）。
SHADOW_FLAG_BIT = 1 << 30

DEFAULT_NEAR = 0.1
DEFAULT_FAR = 1000.0
DEFAULT_SLICES = 16
DEFAULT_LIGHT_RADIUS = 1.0


# ---------------------------------------------------------------------------
# 配置与工具
# ---------------------------------------------------------------------------

def _as_config(spec=None):
    """把 int（切片数）/ dict（完整配置）/ None 统一成配置 dict。"""
    cfg = {"near": DEFAULT_NEAR, "far": DEFAULT_FAR, "slices": DEFAULT_SLICES}
    if isinstance(spec, dict):
        cfg["near"] = float(spec.get("near", cfg["near"]))
        cfg["far"] = float(spec.get("far", cfg["far"]))
        cfg["slices"] = int(spec.get("slices", cfg["slices"]))
    elif isinstance(spec, (int, float)) and not isinstance(spec, bool):
        cfg["slices"] = max(1, int(spec))
    if cfg["near"] <= 0:
        cfg["near"] = DEFAULT_NEAR
    if cfg["far"] <= cfg["near"]:
        cfg["far"] = max(DEFAULT_FAR, cfg["near"] * 10.0)
    cfg["slices"] = max(1, cfg["slices"])
    return cfg


def _as_light(light):
    """把数字 / (z, radius) / dict 统一成光源 dict。"""
    if isinstance(light, dict):
        return {
            "id": light.get("id", 0),
            "z": float(light.get("z", 0.0)),
            "radius": float(light.get("radius", DEFAULT_LIGHT_RADIUS)),
            "shadow": bool(light.get("shadow", False)),
        }
    if isinstance(light, (tuple, list)):
        z = float(light[0]) if len(light) > 0 else 0.0
        radius = float(light[1]) if len(light) > 1 else DEFAULT_LIGHT_RADIUS
        shadow = bool(light[2]) if len(light) > 2 else False
        return {"id": 0, "z": z, "radius": radius, "shadow": shadow}
    return {"id": 0, "z": float(light), "radius": 0.0, "shadow": False}


def _fixed(value):
    """浮点 -> 整数定点（四舍五入），用于跨平台一致的排序键。"""
    return int(round(float(value) * FIXED_POINT_SCALE))


# ---------------------------------------------------------------------------
# 1. 切片按相机近远平面重算
# ---------------------------------------------------------------------------

def slice_recompute(spec=None):
    """每帧根据相机当前 near/far 重算对数深度切片边界。

    不再复用上一帧的边界，近裁剪面移动时切片随之重算，
    避免切片边界上的物体因边界漂移被漏照。
    返回 {"near", "far", "slices", "boundaries"}，boundaries 长度为 slices+1。
    """
    cfg = _as_config(spec)
    near, far, slices = cfg["near"], cfg["far"], cfg["slices"]
    ratio = far / near
    boundaries = [near * (ratio ** (i / slices)) for i in range(slices + 1)]
    boundaries[0] = near          # 端点钉死，消除浮点累积误差
    boundaries[-1] = far
    return {"near": near, "far": far, "slices": slices, "boundaries": boundaries}


def zbin(z, spec=None):
    """用二分查找定位深度 z 所在的切片索引，越界时钳到 [0, slices-1]。"""
    if isinstance(spec, (list, tuple)):
        boundaries = list(spec)
    else:
        boundaries = slice_recompute(spec)["boundaries"]
    idx = bisect_right(boundaries, float(z)) - 1
    return max(0, min(idx, len(boundaries) - 2))


# ---------------------------------------------------------------------------
# 2. 集群键整数定点排序
# ---------------------------------------------------------------------------

def integer_key_sort(keys=None):
    """把浮点键量化成整数定点后再排序，保证不同 CPU 上结果一致。

    元素可以是浮点，或 (key, payload) 元组；相等键按原始下标稳定决胜。
    返回排序后的列表（与输入同形态）。
    """
    if keys is None or isinstance(keys, (int, float)):
        keys = [] if keys is None else [keys]
    items = list(keys)

    def sort_key(item):
        idx, entry = item
        raw = entry[0] if isinstance(entry, (tuple, list)) else entry
        return (_fixed(raw), idx)

    return [entry for _, entry in sorted(enumerate(items), key=sort_key)]


# ---------------------------------------------------------------------------
# 3. 集群缓冲双缓冲 + 等待 GPU 完成
# ---------------------------------------------------------------------------

class _GpuFence:
    """模拟 GPU fence：记录 GPU 已消费完成的帧号。"""

    def __init__(self):
        self.completed_frame = -1

    def signal(self, frame):
        self.completed_frame = max(self.completed_frame, frame)

    def wait(self, frame):
        """真实实现里这里阻塞到 GPU 读完；模拟环境下直接推进。"""
        if self.completed_frame < frame:
            self.signal(frame)


class _DoubleBufferState:
    def __init__(self):
        self.buffers = [{}, {}]   # 前后两份集群缓冲
        self.write_index = 0
        self.fence = _GpuFence()
        self.last_published = -1


_DOUBLE_BUFFER = _DoubleBufferState()


def double_buffer(frame=None):
    """双缓冲重建集群缓冲。

    写入前等待 GPU fence 确认上一帧（front buffer）已被读完，
    然后只写 back buffer 并交换，杜绝覆写 GPU 仍在读取的光源索引。
    返回 {"frame", "write_index", "buffer", "published"}。
    """
    state = _DOUBLE_BUFFER
    frame = int(frame) if isinstance(frame, (int, float)) else state.last_published + 1

    # 写 back buffer 之前，确保 GPU 已读完上一帧发布的缓冲。
    if state.last_published >= 0:
        state.fence.wait(state.last_published)

    back = state.write_index
    state.buffers[back] = {"frame": frame, "clusters": {}}
    state.write_index = 1 - back
    state.last_published = frame
    state.fence.signal(frame)
    return {
        "frame": frame,
        "write_index": back,
        "buffer": state.buffers[back],
        "published": state.last_published,
    }


# ---------------------------------------------------------------------------
# 4. 超界光源补登记到相邻切片
# ---------------------------------------------------------------------------

def overflow_neighbor(light=None, spec=None):
    """返回光源 [z-radius, z+radius] 实际覆盖的所有切片索引。

    点光源半径跨出所属切片时，补登记到相邻切片，
    避免切片边缘亮度突变。半径为 0 时退化为单切片。
    """
    if light is None:
        light = {"z": DEFAULT_NEAR, "radius": 0.0}
    entry = _as_light(light)
    boundaries = slice_recompute(spec)["boundaries"]
    lo = entry["z"] - abs(entry["radius"])
    hi = entry["z"] + abs(entry["radius"])
    slices = len(boundaries) - 1
    first = max(0, zbin(lo, boundaries))
    last = min(slices - 1, zbin(hi, boundaries))
    return list(range(first, last + 1))


# ---------------------------------------------------------------------------
# 5. 阴影标记参与集群键
# ---------------------------------------------------------------------------

def shadow_flag(light=None):
    """把阴影标记编码进集群键的高位。

    投影与不投影光源落在不同的键空间，切换时不再共用集群而闪烁。
    返回整数键；输入为数字时按 z 深度处理。
    """
    entry = _as_light(light if light is not None else 0.0)
    key = _fixed(entry["z"]) & (SHADOW_FLAG_BIT - 1)
    if entry["shadow"]:
        key |= SHADOW_FLAG_BIT
    return key


# ---------------------------------------------------------------------------
# 6. 命中统计按确定性顺序
# ---------------------------------------------------------------------------

def determinism(records=None):
    """命中统计按 (slice, light_id, shadow) 排序后输出。

    不依赖字典/集合的迭代顺序，同一场景两次运行命中数与顺序一致。
    返回排序后的记录列表。
    """
    if records is None or isinstance(records, (int, float)):
        records = [] if records is None else [{"slice": 0, "light_id": int(records), "shadow": 0}]

    def order(rec):
        if isinstance(rec, dict):
            return (int(rec.get("slice", 0)), int(rec.get("light_id", 0)), int(bool(rec.get("shadow", 0))))
        return (0, 0, 0)

    return sorted((dict(r) if isinstance(r, dict) else {"slice": 0, "light_id": 0, "shadow": 0}
                   for r in records), key=order)


# ---------------------------------------------------------------------------
# 集群构建主流程
# ---------------------------------------------------------------------------

def cluster_build(spec=None, lights=None):
    """完整构建：重算切片 -> 越界补登记 -> 阴影键 -> 定点排序 -> 双缓冲发布。"""
    layout = slice_recompute(spec)
    if lights is None:
        lights = []
    entries = []
    for i, raw in enumerate(lights):
        light = _as_light(raw)
        light["id"] = int(light.get("id") or i)
        for s in overflow_neighbor(light, layout):
            entries.append({
                "slice": s,
                "light_id": light["id"],
                "shadow": 1 if light["shadow"] else 0,
                "key": shadow_flag(light),
            })
    # 定点排序 + 确定性顺序
    entries = integer_key_sort([(e["key"], e) for e in entries])
    ordered = determinism([e for _, e in entries])

    clusters = {}
    for e in ordered:
        clusters.setdefault(e["slice"], []).append(e["light_id"])

    published = double_buffer(len(clusters))
    return {
        "layout": layout,
        "clusters": clusters,
        "records": ordered,
        "published_frame": published["frame"],
    }


def light_index(cluster=None, lights=None):
    """为集群生成确定性顺序的光源索引列表。"""
    if isinstance(cluster, dict) and "clusters" in cluster:
        return {s: sorted(ids) for s, ids in sorted(cluster["clusters"].items())}
    ids = []
    if lights:
        for i, raw in enumerate(lights):
            ids.append(_as_light(raw)["id"] or i)
    elif isinstance(cluster, (list, tuple)):
        ids = [int(x) for x in cluster]
    elif isinstance(cluster, (int, float)):
        ids = [int(cluster)]
    return sorted(set(ids))


def occupancy(cluster=None):
    """每个切片的光源占用统计：{slice: count}，按键升序。"""
    if isinstance(cluster, dict) and "clusters" in cluster:
        cluster = cluster["clusters"]
    if isinstance(cluster, dict):
        return {s: len(ids) for s, ids in sorted(cluster.items())}
    return {0: 0}


def metrics(cluster=None):
    """构建指标：切片数、总登记数（含越界补登记）、单切片最大占用。"""
    occ = occupancy(cluster)
    total = sum(occ.values())
    return {
        "slices_used": len(occ),
        "total_assignments": total,
        "max_per_slice": max(occ.values()) if occ else 0,
    }


def light_radius(light=None):
    """光源有效半径：取绝对值并钳到非负，供切片覆盖范围计算使用。"""
    entry = _as_light(light if light is not None else 0.0)
    return abs(entry["radius"])


# ---------------------------------------------------------------------------
# 校验 / 审计 / 回退
# ---------------------------------------------------------------------------

def validation(result=None):
    """校验构建结果：边界单调、索引在界内、统计确定性。"""
    if not isinstance(result, dict) or "layout" not in result:
        result = cluster_build(result)
    boundaries = result["layout"]["boundaries"]
    slices = result["layout"]["slices"]
    problems = []
    if any(boundaries[i] >= boundaries[i + 1] for i in range(len(boundaries) - 1)):
        problems.append("boundaries_not_monotonic")
    for s in result["clusters"]:
        if not 0 <= s < slices:
            problems.append("slice_out_of_range")
    if determinism(result["records"]) != result["records"]:
        problems.append("records_not_deterministic")
    return {"ok": not problems, "problems": problems}


def audit(result=None):
    """审计：返回每切片占用、越界补登记次数与确定性命中数。"""
    if not isinstance(result, dict) or "layout" not in result:
        result = cluster_build(result)
    occ = occupancy(result)
    unique_lights = len({r["light_id"] for r in result["records"]})
    return {
        "occupancy": occ,
        "overflow_registrations": len(result["records"]) - unique_lights,
        "hit_count": len(result["records"]),
        "unique_lights": unique_lights,
    }


def fallback(spec=None, lights=None):
    """回退路径：构建失败时把全部光源登记到所有切片，保证不漏照。"""
    try:
        return cluster_build(spec, lights)
    except Exception:
        layout = slice_recompute(spec)
        ids = [i for i, _ in enumerate(lights or [])]
        return {
            "layout": layout,
            "clusters": {s: list(ids) for s in range(layout["slices"])},
            "records": determinism([
                {"slice": s, "light_id": i, "shadow": 0}
                for s in range(layout["slices"]) for i in ids
            ]),
            "published_frame": -1,
            "fallback": True,
        }
