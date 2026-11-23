import pytest
import graphics.clustered_light_cull as mod

CASES = [
    ("slice_recompute", (1,)),
    ("integer_key_sort", (1,)),
    ("double_buffer", (1,)),
    ("overflow_neighbor", (1,)),
    ("shadow_flag", (1,)),
    ("determinism", (1,)),
    ("cluster_build", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("light_radius", (1,)),
    ("metrics", (1,)),
    ("zbin", (1,)),
    ("light_index", (1,)),
    ("occupancy", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
