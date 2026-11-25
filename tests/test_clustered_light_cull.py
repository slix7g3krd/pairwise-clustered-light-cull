import pytest
import graphics.clustered_light_cull as mod

CASES = [
    ("slice_radius_floor", (1,)),
    ("explicit_barrier", (1,)),
    ("upload_frame_snapshot", (1,)),
    ("count_deferred_commit", (1,)),
    ("near_plane_serial", (1,)),
    ("skip_empty_cluster", (1,)),
    ("build_clusters", (1,)),
    ("cull", (1,)),
    ("shade", (1,)),
    ("present", (1,)),
    ("validate_cluster", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
