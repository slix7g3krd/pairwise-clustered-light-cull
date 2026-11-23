# pairwise-clustered-light-cull

## 领域

游戏、图形与多媒体

## 主题

光源集群化剔除

## 说明

深度切片漂移、浮点排序与集群缓冲覆写。

初始状态下 `graphics/clustered_light_cull.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
