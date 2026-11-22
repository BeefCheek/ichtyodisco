"""The Depth Pro 2 adapter, against a fake ``depth_pro_2`` module (no weights needed)."""

import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")


class _FakeStream:
    def __init__(self, f_px):
        self.f_px = f_px
        self.frames = 0

    def step(self, frame):
        assert frame.ndim == 3 and frame.shape[0] == 3
        self.frames += 1
        h, w = frame.shape[1:]
        inv = torch.linspace(2.0, 0.5, w).repeat(h, 1)  # left = near, right = far
        inv[0, 0] = 0.0  # one "infinitely far" pixel
        depth = None if self.frames < 2 else 500.0 / (w * inv)  # no focal length on frame 1
        return [{"inverse_depth": inv, "depth": depth, "focallength_px": None}]


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.streams = []

    def stream(self, window, overlap=0, f_px=None, *, mode):
        assert mode == "unbuffered" and window == 3
        s = _FakeStream(f_px)
        self.streams.append(s)
        return s


@pytest.fixture
def fake_depth_pro_2(monkeypatch):
    mod = types.ModuleType("depth_pro_2")
    created = {}

    def create_model_and_transforms(preset, device=None):
        created["preset"] = preset
        model = _FakeModel()
        created["model"] = model
        return model, lambda rgb: torch.from_numpy(rgb).permute(2, 0, 1).float() / 255

    mod.create_model_and_transforms = create_model_and_transforms
    monkeypatch.setitem(sys.modules, "depth_pro_2", mod)
    return created


def test_depthpro2_streams_frames_and_returns_far_is_larger(fake_depth_pro_2):
    from aquarium.depth.depthpro2_estimator import DepthPro2Estimator

    est = DepthPro2Estimator("depth-pro-2-b", input_width=64, window=3)
    assert fake_depth_pro_2["preset"] == "depthpro2-b" and est.name == "depth-pro-2-b"
    frame = np.zeros((90, 160, 3), np.uint8)
    for _ in range(3):
        d = est.predict(frame)
    assert d.shape == (36, 64) and d.dtype == np.float32  # resized to input_width, aspect kept
    assert np.isnan(d[0, 0])  # infinite depth is masked out
    assert d[5, -1] > d[5, 1]  # right side (far) has larger depth
    assert len(fake_depth_pro_2["model"].streams) == 1  # one live session for one camera
    est.predict(np.zeros((120, 160, 3), np.uint8))  # resolution change -> new session
    assert len(fake_depth_pro_2["model"].streams) == 2


def test_cli_routes_depth_pro_2(fake_depth_pro_2):
    from aquarium.cli import config_from_args, parse_args
    from aquarium.pipeline import build_depth

    cfg = config_from_args(parse_args(["-s", "0", "--depth", "depth-pro-2"]))
    cfg.depth.window = 3
    est = build_depth(cfg)
    assert type(est).__name__ == "DepthPro2Estimator" and est.name == "depth-pro-2-s"
