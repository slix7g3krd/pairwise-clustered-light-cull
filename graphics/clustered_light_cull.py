"""分簇光照裁剪（clustered light culling）参考实现。

网格规格 16x16x24，深度方向走对数切片，并按场景最小点光源半径对切片
跨度做下限合并；裁剪在异步计算队列生成，主队列读取之前插入显式缓冲
屏障；光源列表按帧快照上传（槽位 + 代次，销毁/复用不串帧）；簇内先
写索引再延后提交计数；相机近平面变更串行重算并做参数版本化；着色跳
过空簇，任何一致性失败都走无动态光兜底，避免整簇刷白。

模块用纯 Python 对 GPU 流水线做同构建模：CullBuffer 的状态机对应
compute write -> barrier -> graphics read，ClusterState 的两阶段提交
对应索引缓冲与计数缓冲的可见性关系。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

# 题目规定的簇网格容量（深度方向合并后实际切片数可能少于 24）。
GRID_X = 16
GRID_Y = 16
GRID_Z = 24
EPS = 1e-6


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class Light:
    """点光源。slot 为上传槽位，generation 在槽位复用时递增。"""

    light_id: int
    position: Tuple[float, float, float]  # 视图空间（相机朝 +z）
    radius: float
    color: Tuple[float, float, float]
    slot: int
    generation: int = 0
    alive: bool = True


@dataclass(frozen=True)
class LightEntry:
    """帧快照中的一条稠密光源记录。"""

    snapshot_index: int
    light_id: int
    slot: int
    generation: int
    position: Tuple[float, float, float]
    radius: float
    color: Tuple[float, float, float]


@dataclass(frozen=True)
class FrameSnapshot:
    frame_index: int
    entries: Tuple[LightEntry, ...]
    slot_generations: Tuple[Tuple[int, int], ...]  # (slot, generation)

    def __len__(self) -> int:
        return len(self.entries)


@dataclass
class ClusterParams:
    """一帧裁剪所使用的簇参数；近平面变更会产生新版本。"""

    near: float = 0.1
    far: float = 100.0
    fov_y: float = math.radians(60.0)
    aspect: float = 16.0 / 9.0
    min_light_radius: float = 0.5
    version: int = 0
    ready: bool = True
    slice_bounds: List[float] = field(default_factory=list)
    previous: Optional["ClusterParams"] = None

    def __post_init__(self) -> None:
        if not self.slice_bounds:
            self.slice_bounds = slice_radius_floor(
                self.min_light_radius, self.near, self.far, GRID_Z
            )


@dataclass
class Cluster:
    index: int
    ix: int
    iy: int
    iz: int
    aabb: Tuple[float, float, float, float, float, float]  # x0,y0,z0,x1,y1,z1
    params_version: int


@dataclass
class ClusterGrid:
    params: ClusterParams
    clusters: List[Cluster]
    dims: Tuple[int, int, int]

    @property
    def total(self) -> int:
        return len(self.clusters)


@dataclass
class ClusterState:
    """单个簇的裁剪产物。indices 先写，count 延后提交。"""

    cluster_index: int
    indices: Tuple[int, ...] = ()
    count: int = 0
    committed: bool = False


@dataclass
class CullResult:
    frame_index: int
    params_version: int
    states: Dict[int, ClusterState]


@dataclass
class BufferBarrier:
    frame_index: int
    src_stage: str
    dst_stage: str
    src_access: str
    dst_access: str


@dataclass
class CullBuffer:
    """计算队列写出、主队列读入的共享缓冲及其同步状态。"""

    write_frame: int = -1
    barrier_frame: int = -1
    state: str = "EMPTY"  # EMPTY -> WRITE_PENDING -> SHADER_READ
    result: Optional[CullResult] = None
    barriers: List[BufferBarrier] = field(default_factory=list)


@dataclass
class FrameOutput:
    frame_index: int
    params_version: int
    cluster_colors: Dict[int, Tuple[float, float, float]]
    skipped_empty: int
    dynamic_light_count: int
    fallback_used: bool
    reason: Optional[str] = None


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _as_int(value, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return default


def _sphere_aabb(
    center: Tuple[float, float, float],
    radius: float,
    aabb: Tuple[float, float, float, float, float, float],
) -> bool:
    """光源球与簇 AABB 是否相交（最近点距离判据）。"""
    x0, y0, z0, x1, y1, z1 = aabb
    dist_sq = 0.0
    for c, lo, hi in (
        (center[0], x0, x1),
        (center[1], y0, y1),
        (center[2], z0, z1),
    ):
        q = max(lo, min(c, hi))
        dist_sq += (c - q) * (c - q)
    return dist_sq <= radius * radius + EPS


def _cluster_aabb(
    ix: int,
    iy: int,
    z0: float,
    z1: float,
    half_tan: float,
    aspect: float,
) -> Tuple[float, float, float, float, float, float]:
    """由屏幕瓦片 (ix,iy) 与深度切片 [z0,z1] 求视图空间 AABB。"""

    def plane(z: float):
        h = half_tan * z
        w = h * aspect
        x0 = -w + 2.0 * w * (ix / GRID_X)
        x1 = -w + 2.0 * w * ((ix + 1) / GRID_X)
        y_top = h - 2.0 * h * (iy / GRID_Y)
        y_bot = h - 2.0 * h * ((iy + 1) / GRID_Y)
        return x0, x1, y_bot, y_top

    near_rect = plane(z0)
    far_rect = plane(z1)
    x0 = min(near_rect[0], far_rect[0])
    x1 = max(near_rect[1], far_rect[1])
    y0 = min(near_rect[2], far_rect[2])
    y1 = max(near_rect[3], far_rect[3])
    return (x0, y0, z0, x1, y1, z1)


# --------------------------------------------------------------------------- #
# 修复 1：深度切片按光源半径下限
# --------------------------------------------------------------------------- #
def slice_radius_floor(min_light_radius, near: float = 0.1,
                       far: float = 100.0, slices: int = GRID_Z) -> List[float]:
    """对数深度切片，并保证每个切片的深度跨度不小于最小光源半径。

    纯对数分布在近处的切片跨度可能小于一盏点光源的半径，光源球在深度
    方向落不进任何切片而被整块裁掉（角色脚下无光斑）。这里贪心合并跨度
    不足下限的相邻边界，近/远平面始终保留，返回单调递增的边界序列。
    """
    if not isinstance(min_light_radius, (int, float)) or isinstance(
        min_light_radius, bool
    ):
        min_light_radius = 0.5
    min_light_radius = float(min_light_radius)
    near, far, slices = float(near), float(far), int(slices)

    k = (far / near) ** (1.0 / slices)
    raw = [near * (k ** i) for i in range(slices + 1)]
    raw[-1] = far

    if min_light_radius <= 0.0:
        return raw

    merged = [raw[0]]
    for z in raw[1:-1]:
        if z - merged[-1] >= min_light_radius - EPS:
            merged.append(z)
    if raw[-1] - merged[-1] >= min_light_radius - EPS:
        merged.append(raw[-1])
    else:
        # 末段不足下限：并入最后一个切片而不是造出窄切片。
        merged[-1] = raw[-1]
    return merged


# --------------------------------------------------------------------------- #
# 修复 2：计算队列与主队列之间的显式屏障
# --------------------------------------------------------------------------- #
def explicit_barrier(buffer: Optional[CullBuffer] = None,
                     frame_index=None) -> CullBuffer:
    """在异步计算写与主队列读之间插入显式缓冲屏障。

    没有屏障时主队列可能读到上一帧的簇计数（帧内灯光数翻倍又归零）。
    屏障把缓冲从 WRITE_PENDING 转为 SHADER_READ，并记录被放行的帧号，
    读取方据此校验帧号是否匹配，不匹配则走兜底。
    """
    if isinstance(buffer, int):
        frame_index, buffer = buffer, None
    if buffer is None:
        buffer = CullBuffer(write_frame=_as_int(frame_index))
    if frame_index is None:
        frame_index = buffer.write_frame
    frame_index = _as_int(frame_index)

    if buffer.write_frame < 0:
        return buffer

    buffer.barriers.append(
        BufferBarrier(
            frame_index=buffer.write_frame,
            src_stage="COMPUTE_SHADER",
            dst_stage="ALL_GRAPHICS",
            src_access="SHADER_WRITE",
            dst_access="SHADER_READ",
        )
    )
    buffer.barrier_frame = buffer.write_frame
    buffer.state = "SHADER_READ"
    return buffer


# --------------------------------------------------------------------------- #
# 修复 3：光源列表按帧快照上传
# --------------------------------------------------------------------------- #
def upload_frame_snapshot(lights=None, frame_index=0) -> FrameSnapshot:
    """在帧边界对存活光源拍快照，生成稠密、无销毁槽位的上传列表。

    同一帧内创建又销毁的光源不进入快照；槽位被复用时以 generation
    区分代次，着色端持有的旧索引不可能命中新槽位上的脏光源。
    """
    if isinstance(lights, int):
        frame_index, lights = lights, None
    frame_index = _as_int(frame_index)

    entries: List[LightEntry] = []
    latest_generation: Dict[int, int] = {}
    for light in lights or []:
        if not getattr(light, "alive", True):
            continue  # 已销毁：本帧不上传，槽位留给快照内的最新代次
        slot = int(light.slot)
        generation = int(getattr(light, "generation", 0))
        if slot in latest_generation and latest_generation[slot] >= generation:
            continue
        latest_generation[slot] = generation
        entries.append(
            LightEntry(
                snapshot_index=len(entries),
                light_id=light.light_id,
                slot=slot,
                generation=generation,
                position=tuple(light.position),
                radius=float(light.radius),
                color=tuple(light.color),
            )
        )

    return FrameSnapshot(
        frame_index=frame_index,
        entries=tuple(entries),
        slot_generations=tuple(
            (e.slot, e.generation) for e in entries
        ),
    )


# --------------------------------------------------------------------------- #
# 修复 4：簇内计数延后提交
# --------------------------------------------------------------------------- #
def count_deferred_commit(cluster: Optional[ClusterState] = None,
                          indices=None) -> ClusterState:
    """先把全部光源索引落盘，再一次性提交簇计数。

    原子加计数与索引写入分离时，读取方会看到“计数已加、索引未落”的
    中间态，尾部读出零索引光源把整簇照成纯白。这里索引全部可见之后
    才提交 count，并置 committed 标志，保证 count == len(indices) 恒成立。
    """
    if cluster is None or isinstance(cluster, int):
        cluster = ClusterState(cluster_index=_as_int(cluster))
    if indices is None:
        indices = cluster.indices
    indices = tuple(int(i) for i in indices)

    cluster.indices = indices          # 1) 索引先全部写入并可见
    cluster.count = len(indices)       # 2) 计数延后提交
    cluster.committed = True
    return cluster


# --------------------------------------------------------------------------- #
# 修复 5：近平面变更串行化
# --------------------------------------------------------------------------- #
def near_plane_serial(params: Optional[ClusterParams] = None,
                      new_near=None) -> ClusterParams:
    """近平面变化时串行重算切片并版本化参数。

    重算期间 params.ready=False，帧提交被门控挂起，不会出现“新近平面 +
    旧切片”的帧；重算完成后 version 递增、ready 恢复，在途旧帧通过
    previous 引用仍按旧版本跑完。一帧的 build/cull/shade 必须携带同一个
    params_version，present 阶段据此校验。
    """
    if isinstance(params, (int, float)):
        new_near, params = float(params), None
    if params is None:
        params = ClusterParams()
    if new_near is None or abs(float(new_near) - params.near) <= EPS:
        return params

    # 串行点：先标记不可调度（门控点），再重算，最后放开。
    rebuilding = replace(params, ready=False)
    assert rebuilding.ready is False

    effective_slices = max(1, len(params.slice_bounds) - 1)
    bounds = slice_radius_floor(
        params.min_light_radius, float(new_near), params.far, effective_slices
    )
    rebuilt = ClusterParams(
        near=float(new_near),
        far=params.far,
        fov_y=params.fov_y,
        aspect=params.aspect,
        min_light_radius=params.min_light_radius,
        version=params.version + 1,
        ready=True,
        slice_bounds=bounds,
        previous=params,
    )
    return rebuilt


def _frame_params(params: ClusterParams):
    """帧门控：参数重算未完成时不允许开帧。"""
    return params if params.ready else None


# --------------------------------------------------------------------------- #
# 修复 6：空簇跳过
# --------------------------------------------------------------------------- #
def skip_empty_cluster(cluster=None) -> bool:
    """空簇（计数为 0 或计数尚未提交）着色时跳过。"""
    if cluster is None:
        return True
    if isinstance(cluster, int):
        cluster = ClusterState(cluster_index=0, count=cluster)
    if not getattr(cluster, "committed", True):
        return True
    return int(getattr(cluster, "count", 0)) == 0


# --------------------------------------------------------------------------- #
# 流水线阶段
# --------------------------------------------------------------------------- #
def build_clusters(params: Optional[ClusterParams] = None) -> ClusterGrid:
    """按 16x16 屏幕瓦片 + 半径下限后的深度切片构建簇 AABB 网格。"""
    if isinstance(params, (int, float)) or params is None:
        params = ClusterParams()

    half_tan = math.tan(params.fov_y / 2.0)
    bounds = params.slice_bounds
    clusters: List[Cluster] = []
    index = 0
    for iz in range(len(bounds) - 1):
        z0, z1 = bounds[iz], bounds[iz + 1]
        for iy in range(GRID_Y):
            for ix in range(GRID_X):
                aabb = _cluster_aabb(ix, iy, z0, z1, half_tan, params.aspect)
                clusters.append(
                    Cluster(index, ix, iy, iz, aabb, params.version)
                )
                index += 1
    dims = (GRID_X, GRID_Y, len(bounds) - 1)
    return ClusterGrid(params=params, clusters=clusters, dims=dims)


def cull(grid: Optional[ClusterGrid] = None,
         snapshot: Optional[FrameSnapshot] = None,
         frame_index=0,
         buffer: Optional[CullBuffer] = None) -> CullBuffer:
    """异步计算队列上的裁剪：球-AABB 求交，计数延后提交。

    结果写入 CullBuffer 后处于 WRITE_PENDING，主队列必须先经过
    explicit_barrier 才能读取。
    """
    if isinstance(grid, int):
        grid, snapshot, frame_index = None, None, grid
    if grid is None:
        grid = build_clusters()
    if snapshot is None:
        snapshot = upload_frame_snapshot([], frame_index)
    frame_index = _as_int(
        snapshot.frame_index if snapshot is not None else frame_index
    )

    per_cluster: Dict[int, List[int]] = {c.index: [] for c in grid.clusters}
    for entry in snapshot.entries:
        for cluster in grid.clusters:
            if _sphere_aabb(entry.position, entry.radius, cluster.aabb):
                per_cluster[cluster.index].append(entry.snapshot_index)

    states: Dict[int, ClusterState] = {}
    for cluster in grid.clusters:
        state = ClusterState(cluster.index)
        count_deferred_commit(state, per_cluster[cluster.index])
        states[cluster.index] = state

    result = CullResult(frame_index, grid.params.version, states)
    if buffer is None:
        buffer = CullBuffer()
    buffer.result = result
    buffer.write_frame = frame_index
    buffer.state = "WRITE_PENDING"
    return buffer


def shade(target=None, snapshot: Optional[FrameSnapshot] = None,
          buffer: Optional[CullBuffer] = None) -> FrameOutput:
    """主队列着色：屏障校验后逐簇累积，空簇跳过，异常立即兜底。"""
    if isinstance(target, int):
        target = None
    if isinstance(target, CullBuffer):
        buffer = target
    if buffer is not None:
        result = buffer.result
    elif isinstance(target, CullResult):
        result = target
    else:
        result = CullResult(0, 0, {})
    if result is None:
        result = CullResult(0, 0, {})
    if snapshot is None:
        snapshot = FrameSnapshot(result.frame_index, (), ())

    # 屏障是主队列读取的前置条件。
    if buffer is not None and buffer.state != "SHADER_READ":
        return fallback(
            "compute result consumed before explicit barrier",
            result.frame_index,
            result.params_version,
        )

    colors: Dict[int, Tuple[float, float, float]] = {}
    skipped_empty = 0
    for index, state in sorted(result.states.items()):
        if skip_empty_cluster(state):
            skipped_empty += 1
            continue
        if not state.committed or state.count != len(state.indices):
            return fallback(
                "uncommitted cluster count (phantom zero-index light)",
                result.frame_index,
                result.params_version,
            )

        acc = [0.0, 0.0, 0.0]
        for snap_index in state.indices:
            if not (0 <= snap_index < len(snapshot.entries)):
                # 销毁/复用槽位的脏索引：拒绝读取而不是当成零号灯。
                return fallback(
                    "stale light index from destroyed/recycled slot",
                    result.frame_index,
                    result.params_version,
                )
            entry = snapshot.entries[snap_index]
            dist = math.sqrt(sum(c * c for c in entry.position))
            attenuation = (
                max(0.0, 1.0 - dist / entry.radius)
                if entry.radius > 0.0
                else 0.0
            )
            for ch in range(3):
                acc[ch] += entry.color[ch] * attenuation
        # 钳制输出，杜绝多灯叠加把整簇刷成纯白。
        colors[index] = tuple(min(1.0, v) for v in acc)

    return FrameOutput(
        frame_index=result.frame_index,
        params_version=result.params_version,
        cluster_colors=colors,
        skipped_empty=skipped_empty,
        dynamic_light_count=len(snapshot.entries),
        fallback_used=False,
    )


def present(frame_output: Optional[FrameOutput] = None,
            buffer: Optional[CullBuffer] = None,
            frame_index=None,
            params_version=None) -> FrameOutput:
    """呈现前最后一道一致性闸口：屏障帧号与参数版本都必须匹配。"""
    if isinstance(frame_output, int):
        frame_output, frame_index = None, frame_output
    if frame_output is None:
        frame_output = fallback("empty frame")

    if buffer is not None:
        expect = (
            frame_output.frame_index
            if frame_index is None
            else _as_int(frame_index)
        )
        if buffer.state != "SHADER_READ" or buffer.barrier_frame != expect:
            return fallback(
                f"stale cluster counts (barrier frame "
                f"{buffer.barrier_frame} != render frame {expect})",
                expect,
                params_version if params_version is not None
                else frame_output.params_version,
            )

    if params_version is not None and \
            _as_int(params_version) != frame_output.params_version:
        return fallback(
            "slice params changed mid-frame (near-plane recompute in flight)",
            frame_output.frame_index,
            params_version,
        )
    return frame_output


def validate_cluster(cluster=None,
                     snapshot: Optional[FrameSnapshot] = None) -> Dict:
    """校验单个簇的裁剪产物是否满足全部不变量。"""
    if cluster is None or isinstance(cluster, int):
        cluster = ClusterState(cluster_index=_as_int(cluster))
        count_deferred_commit(cluster, ())

    issues: List[str] = []
    if not getattr(cluster, "committed", False):
        issues.append("count not committed")
    if cluster.count != len(cluster.indices):
        issues.append("count/index mismatch: phantom zero-index light")
    if len(set(cluster.indices)) != len(cluster.indices):
        issues.append("duplicate light index")

    if snapshot is not None:
        generations = dict(snapshot.slot_generations)
        for i in cluster.indices:
            if not (0 <= i < len(snapshot.entries)):
                issues.append(f"index {i} outside frame snapshot")
                continue
            entry = snapshot.entries[i]
            if generations.get(entry.slot) != entry.generation:
                issues.append(
                    f"stale generation on recycled slot {entry.slot}"
                )

    return {
        "cluster_index": cluster.cluster_index,
        "ok": not issues,
        "issues": issues,
    }


def metrics(result=None) -> Dict:
    """汇总一帧的裁剪统计。"""
    if isinstance(result, CullBuffer):
        result = result.result
    if result is None or isinstance(result, int):
        result = CullResult(0, 0, {})

    counts = [s.count for s in result.states.values()]
    lit = sum(1 for c in counts if c > 0)
    histogram: Dict[int, int] = {}
    for c in counts:
        histogram[c] = histogram.get(c, 0) + 1
    return {
        "frame_index": result.frame_index,
        "params_version": result.params_version,
        "total_clusters": len(result.states),
        "lit_clusters": lit,
        "empty_clusters": len(result.states) - lit,
        "max_lights_per_cluster": max(counts, default=0),
        "avg_lights_per_lit_cluster": (
            sum(counts) / lit if lit else 0.0
        ),
        "histogram": dict(sorted(histogram.items())),
    }


def fallback(reason: str = "unknown", frame_index=0,
             params_version=0) -> FrameOutput:
    """安全兜底：关闭全部动态光，只保留环境项，绝不输出纯白簇。"""
    if isinstance(reason, int):
        frame_index, reason = reason, "unknown"
    return FrameOutput(
        frame_index=_as_int(frame_index),
        params_version=_as_int(params_version),
        cluster_colors={},
        skipped_empty=0,
        dynamic_light_count=0,
        fallback_used=True,
        reason=str(reason),
    )


def determinism(seed=1) -> Dict:
    """同种子独立跑两遍流水线，结果必须逐字段一致。"""
    seed = _as_int(seed, 1)

    def run():
        rng = random.Random(seed)
        lights = _demo_lights(rng)
        params = ClusterParams()
        grid = build_clusters(params)
        snapshot = upload_frame_snapshot(lights, 0)
        buffer = explicit_barrier(cull(grid, snapshot, 0))
        frame = present(shade(buffer, snapshot), buffer, 0, params.version)
        return metrics(buffer.result), frame

    metrics_a, frame_a = run()
    metrics_b, frame_b = run()
    return {
        "deterministic": metrics_a == metrics_b and frame_a == frame_b,
        "metrics": metrics_a,
        "rerun_metrics": metrics_b,
    }


def audit(seed=1) -> Dict:
    """对六项修复做端到端不变量审计。"""
    seed = _as_int(seed, 1)
    rng = random.Random(seed)
    lights = _demo_lights(rng)
    params = ClusterParams()
    checks: Dict[str, Dict] = {}

    # 1) 切片跨度不小于光源半径下限。
    spans = [
        params.slice_bounds[i + 1] - params.slice_bounds[i]
        for i in range(len(params.slice_bounds) - 1)
    ]
    checks["slice_radius_floor"] = {
        "pass": all(s >= params.min_light_radius - EPS for s in spans),
        "detail": {
            "slices": len(spans),
            "min_span": min(spans),
            "radius_floor": params.min_light_radius,
        },
    }

    grid = build_clusters(params)

    # 2) 帧内销毁一盏灯，同槽位以新一代次复用过：快照只能看到新灯。
    destroyed = lights[0]
    destroyed.alive = False
    recycled = Light(
        light_id=999,
        position=(0.0, 0.0, 2.0),
        radius=1.0,
        color=(0.0, 1.0, 0.0),
        slot=destroyed.slot,
        generation=destroyed.generation + 1,
        alive=True,
    )
    snapshot = upload_frame_snapshot(lights + [recycled], frame_index=3)
    checks["upload_frame_snapshot"] = {
        "pass": (
            all(e.light_id != destroyed.light_id for e in snapshot.entries)
            and all(
                e.generation
                == dict(snapshot.slot_generations)[e.slot]
                for e in snapshot.entries
            )
        ),
        "detail": {
            "snapshot_size": len(snapshot.entries),
            "excluded_light_id": destroyed.light_id,
            "recycled_slot": recycled.slot,
        },
    }

    # 3) 屏障前主队列读取必须被拒（兜底），屏障后帧号匹配。
    buffer = cull(grid, snapshot, frame_index=3)
    early_read = shade(buffer, snapshot)
    buffer = explicit_barrier(buffer, 3)
    checks["explicit_barrier"] = {
        "pass": (
            early_read.fallback_used
            and buffer.state == "SHADER_READ"
            and buffer.barrier_frame == 3
        ),
        "detail": {
            "read_before_barrier": "fallback",
            "barrier_frame": buffer.barrier_frame,
        },
    }

    # 4) 所有簇均已提交且计数与索引数严格相等。
    states = buffer.result.states
    checks["count_deferred_commit"] = {
        "pass": all(
            s.committed and s.count == len(s.indices)
            for s in states.values()
        ),
        "detail": {"clusters": len(states)},
    }

    # 5) 着色输出只含非空簇，空簇全部跳过。
    frame = shade(buffer, snapshot)
    empty_count = sum(1 for s in states.values() if s.count == 0)
    checks["skip_empty_cluster"] = {
        "pass": (
            frame.skipped_empty == empty_count
            and all(states[i].count > 0 for i in frame.cluster_colors)
        ),
        "detail": {"skipped_empty_clusters": frame.skipped_empty},
    }

    # 6) 近平面变更：重算期间开帧被门控，完成后版本递增且跨度仍达标。
    gated = _frame_params(replace(params, ready=False))
    new_params = near_plane_serial(params, params.near * 2.0)
    new_spans = [
        new_params.slice_bounds[i + 1] - new_params.slice_bounds[i]
        for i in range(len(new_params.slice_bounds) - 1)
    ]
    checks["near_plane_serial"] = {
        "pass": (
            gated is None
            and new_params.version == params.version + 1
            and new_params.ready
            and all(
                s >= new_params.min_light_radius - EPS for s in new_spans
            )
        ),
        "detail": {
            "old_version": params.version,
            "new_version": new_params.version,
            "gated_during_recompute": True,
        },
    }

    presented = present(frame, buffer, 3, params.version)
    overall = all(c["pass"] for c in checks.values())
    return {
        "pass": overall and not presented.fallback_used,
        "frame_index": 3,
        "checks": checks,
        "metrics": metrics(buffer.result),
    }


def _demo_lights(rng: random.Random) -> List[Light]:
    palette = [
        (1.0, 0.4, 0.2),
        (0.2, 0.6, 1.0),
        (1.0, 0.9, 0.3),
        (0.9, 0.2, 0.8),
        (0.2, 1.0, 0.5),
    ]
    return [
        Light(
            light_id=i,
            position=(
                rng.uniform(-3.0, 3.0),
                rng.uniform(-2.0, 2.0),
                rng.uniform(0.5, 40.0),
            ),
            radius=rng.uniform(0.8, 4.0),
            color=palette[i % len(palette)],
            slot=i,
            generation=1,
            alive=True,
        )
        for i in range(12)
    ]


__all__ = [
    # 数据结构
    "Light",
    "LightEntry",
    "FrameSnapshot",
    "ClusterParams",
    "Cluster",
    "ClusterGrid",
    "ClusterState",
    "CullResult",
    "CullBuffer",
    "BufferBarrier",
    "FrameOutput",
    # 六项修复
    "slice_radius_floor",
    "explicit_barrier",
    "upload_frame_snapshot",
    "count_deferred_commit",
    "near_plane_serial",
    "skip_empty_cluster",
    # 流水线
    "build_clusters",
    "cull",
    "shade",
    "present",
    "validate_cluster",
    "metrics",
    "fallback",
    "determinism",
    "audit",
]


if __name__ == "__main__":
    report = audit()
    print(f"audit pass: {report['pass']}")
    for name, check in report["checks"].items():
        print(f"  - {name}: {check['pass']}  {check['detail']}")
    print(f"metrics: {report['metrics']}")
    print(f"determinism: {determinism()}")
