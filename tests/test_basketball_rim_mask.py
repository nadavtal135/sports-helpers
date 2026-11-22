"""The rim read from the top of SAM's hoop mask, not from its widest rows.

On a real gym drill clip (TikTok 576x1024, analysed at 406x720) SAM's hoop mask
also held pieces of a net-tension strap and a support post below the net. Those
rows (y 248-254, x 198-228, 2-3 runs) were wider than the rim (y 230-236, x 192-218,
26 px), so the old "widest rows" rule put the rim line on the net, 17 px too low.
The masks here reproduce that row profile; no network.
"""

import base64

import cv2
import numpy as np
import pytest

from basketball import pipeline, sample, settings
from basketball.src import hoop as hoop_mod, synthetic
from basketball.src.cache import Cache
from basketball.src.providers.fixture import FixtureProvider

GW, GH = 406, 720


def old_rule(blob):
    """The previous parser: rows within 3% of the widest anywhere in the mask."""
    rows = np.nonzero(blob.any(axis=1))[0]
    spans = np.array([np.ptp(np.nonzero(blob[r])[0]) + 1 for r in rows])
    rim_rows = rows[spans >= 0.97 * spans.max()]
    cols = np.nonzero(blob[rim_rows].any(axis=0))[0]
    return float(np.mean(rim_rows)), float(cols.min()), float(cols.max() + 1)


def gym_mask(*, strap=True, tilt_deg=0.0):
    """Rim ring y 230-236 / x 192-218, a net tapering to x 198-213 by y 255, and (with
    *strap*) a diagonal strap from the net down-right plus a post piece at x 223-229."""
    m = np.zeros((GH, GW), np.uint8)
    cv2.ellipse(m, (205, 233), (13, 3), tilt_deg, 0, 360, 1, -1)              # rim ring
    net = np.array([[195, 234], [216, 234], [213, 245], [198, 245]], np.int32)
    cv2.fillPoly(m, [net, np.array([[198, 245], [213, 245], [212, 255], [199, 255]], np.int32)], 1)
    m[230, 195:213] = 1                                                        # rim top row
    if strap:
        cv2.line(m, (205, 248), (228, 256), 1, 2)                              # tension strap
        m[248:262, 224:230] = 1                                                # post piece
        m[248:255, 219:222] = 1                                                # bracket bit
    return m


def reply(mask, score=0.72):
    ok, png = cv2.imencode(".png", mask)
    ys, xs = np.nonzero(mask)
    box = [xs.min() / mask.shape[1], ys.min() / mask.shape[0],
           (np.ptp(xs) + 1) / mask.shape[1], (np.ptp(ys) + 1) / mask.shape[0]]
    return {"content": {"object": "img.segment.masks",
                        "items": [{"instance_id": 1, "score": score, "bbox_xywh": box}],
                        "mask": {"format": "png",
                                 "data": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()}}}


def locate(mask):
    return hoop_mod.locate(reply(mask), frame=0, width=mask.shape[1], height=mask.shape[0])


def test_gym_mask_reproduces_the_reported_profile_and_the_old_failure():
    m = gym_mask().astype(bool)
    for y, (lo, hi) in {233: (192, 217), 245: (198, 214)}.items():
        cols = np.flatnonzero(m[y])
        assert abs(cols.min() - lo) <= 1 and abs(cols.max() - hi) <= 1
    assert len(hoop_mod._runs(m[250])) >= 2                    # strap rows are split
    cy, left, right = old_rule(m)
    assert cy > 247 and right > 226                            # the old rule lands on the strap


def test_rim_is_read_from_the_top_despite_a_strap_and_post():
    h = locate(gym_mask())
    assert 232 <= h.cy <= 235
    assert abs(h.left - 192) <= 2 and abs(h.right - 218) <= 2
    assert abs(h.cx - 205) <= 2 and h.top == 230
    assert h.net_bottom <= 258                                 # the post beside the net is not net
    assert any("below the rim" in n for n in h.notes)          # flagged as contaminated


def test_mask_without_the_strap_is_unchanged_and_clean():
    m = gym_mask(strap=False)
    h = locate(m)
    cy, left, right = old_rule(m.astype(bool))
    assert abs(h.cy - cy) <= 1 and abs(h.left - left) <= 1 and abs(h.right - right) <= 1
    assert h.notes == ()


@pytest.mark.parametrize("tilt", [-8.0, 8.0])
def test_tilted_rim_stays_within_tolerance(tilt):
    h = locate(gym_mask(tilt_deg=tilt))
    assert 231 <= h.cy <= 236 and abs(h.left - 192) <= 3 and abs(h.right - 218) <= 3


def night_court_mask():
    """Night-court hoop (768x432): rim center (370, 100), 40 px wide, net bottom 135."""
    m = np.zeros((432, 768), np.uint8)
    cv2.ellipse(m, (370, 100), (20, 6), 0, 0, 360, 1, -1)
    cv2.fillPoly(m, [np.array([[352, 100], [388, 100], [381, 135], [359, 135]], np.int32)], 1)
    return m


@pytest.mark.parametrize("make", [night_court_mask,
                                  lambda: synthetic.hoop_mask(synthetic.W, synthetic.H)])
def test_hoops_that_were_right_do_not_move(make):
    m = make()
    h = locate(m)
    cy, left, right = old_rule(m.astype(bool))
    rows = np.flatnonzero(m.any(axis=1))
    assert abs(h.cy - cy) <= 1 and abs(h.left - left) <= 1 and abs(h.right - right) <= 1
    assert h.top == rows[0] and h.net_bottom == rows[-1] + 1
    assert h.notes == ()


def test_night_court_hoop_is_where_it_was():
    h = locate(night_court_mask())
    assert abs(h.cx - 370) <= 1 and abs(h.cy - 100) <= 1 and abs(h.width - 40) <= 2


def test_hoop_notes_are_exported_as_a_list():
    assert isinstance(locate(gym_mask()).to_dict()["notes"], list)


# ── the synthetic clip end to end, and the rim check image ────────────────────

@pytest.fixture(scope="module")
def sample_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("rim")
    clip = sample.make_sample(d / "clip.mp4")
    r = pipeline.run(clip, d / "out", settings=settings.Settings(), gateway=settings.Gateway(),
                     provider=FixtureProvider(), cache=Cache(d / "cache"))
    return d / "out", r


def test_synthetic_clip_calls_are_unchanged(sample_run):
    _, r = sample_run
    st = r["stats"]
    assert (st["makes"], st["unknown"]) == (3, 1)
    assert sum(1 for s in r["shots"] if s["result"] == "miss" and "rim" in s["reason"]) >= 1


def test_rim_check_image_is_written(sample_run):
    out, r = sample_run
    assert "hoop_check.png" in r["files"]
    img = cv2.imread(str(out / "hoop_check.png"))
    assert img is not None and img.shape[1] >= 600
    assert not any("contaminated" in w for w in r["warnings"])


def test_contaminated_mask_warns(tmp_path):
    warnings = []
    seg = hoop_mod.HoopSegment(0, 100, locate(gym_mask()), [], "only hoop found")
    pipeline._hoop_warnings(hoop_mod.HoopTimeline([seg]), [], settings.Settings(), warnings)
    assert any("contaminated" in w and "hoop_check.png" in w for w in warnings)


def test_rim_estimate_moving_within_a_camera_segment_warns():
    a = locate(gym_mask(strap=False))
    b = hoop_mod.Hoop(cx=a.cx + 9, cy=a.cy + 4, left=a.left + 9, right=a.right + 9, top=a.top,
                      net_bottom=a.net_bottom, score=0.7, frame=600)
    warnings = []
    tl = hoop_mod.HoopTimeline([hoop_mod.HoopSegment(0, 600, a, [], ""),
                                hoop_mod.HoopSegment(600, 1200, b, [], "")])
    pipeline._hoop_warnings(tl, [], settings.Settings(), warnings)
    assert any("rim estimate moved" in w for w in warnings)
    warnings = []
    pipeline._hoop_warnings(tl, [600], settings.Settings(), warnings)     # a cut: expected
    assert not any("rim estimate moved" in w for w in warnings)
