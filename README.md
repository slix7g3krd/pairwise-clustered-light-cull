# pairwise-clustered-light-cull

## 领域

游戏、图形与多媒体

## 主题

分簇光照裁剪

## 说明

切片下限、显式屏障与帧快照上传。

初始状态下 `graphics/clustered_light_cull.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
