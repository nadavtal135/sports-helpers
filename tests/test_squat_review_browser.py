"""Fixture-mode browser acceptance test. Install Playwright and Chromium to run it."""

import json
import shutil
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn

pw = pytest.importorskip("playwright.sync_api")


@pytest.fixture
def fixture_url(tmp_path, monkeypatch):
    from webapp import server

    monkeypatch.setenv("SPORTS_HELPERS_DATA", str(tmp_path))
    monkeypatch.setenv("SPORTS_HELPERS_MODE", "fixture")
    monkeypatch.delenv("VLMRUN_API_KEY", raising=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()
        app = uvicorn.Server(uvicorn.Config(server.app, log_level="error"))
        thread = threading.Thread(target=app.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not app.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert app.started
            yield f"http://{host}:{port}"
        finally:
            app.should_exit = True
            thread.join(timeout=10)


def paused_frame(page, selector, frame, offset=15):
    # Check the actual decoded video's local clock, independently of its caption.
    page.wait_for_function("""([selector,frame]) => {
        const v = document.querySelector(selector);
        return v.readyState >= 2 && !v.seeking && v.paused && Math.floor(v.currentTime * 30) === frame;
    }""", arg=[selector, frame - offset])


def sample_pixels(page):
    return page.locator("#result-video").evaluate("""v => {
        const c = document.createElement('canvas');
        c.width = 90; c.height = 120;
        const ctx = c.getContext('2d');
        ctx.drawImage(v, 0, 0, c.width, c.height);
        return c.toDataURL();
    }""")


def test_fixture_technique_review_to_supporting_footage(fixture_url):
    with pw.sync_playwright() as playwright:
        executable = shutil.which("chromium") or shutil.which("google-chrome") or playwright.chromium.executable_path
        if not Path(executable).is_file():
            pytest.skip("Install Chromium with python -m playwright install chromium")
        browser = playwright.chromium.launch(executable_path=executable, args=["--disable-dev-shm-usage"])
        page = browser.new_page(viewport={"width": 1100, "height": 900})
        page.set_default_timeout(20000)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(fixture_url)
        page.get_by_role("button", name="Squat Analysis").click()
        page.get_by_role("button", name="Use the synthetic sample clip").click()
        pw.expect(page.locator("#clip-info")).to_contain_text("squat_sample", timeout=30000)
        page.locator("#trim-start").fill("0.5")
        page.get_by_role("button", name="Analyze", exact=True).click()
        pw.expect(page.get_by_role("heading", name="Squat technique review")).to_be_visible(timeout=90000)
        pw.expect(page.get_by_role("heading", name="What did I do well?")).to_be_visible()
        pw.expect(page.get_by_role("heading", name="What should I look at more closely?")).to_be_visible()
        pw.expect(page.locator("#tiles")).to_be_hidden()
        pw.expect(page.locator("#step-upload")).to_be_hidden()

        strength, look = page.locator("[data-note=strength]"), page.locator("[data-note=closer_look]")
        pw.expect(strength).to_contain_text("torso lean at the bottom was similar across reps 1–3")
        strength.click()
        paused_frame(page, "#result-video", 54)
        paused_frame(page, "#reference-video", 141)
        pw.expect(page.locator("#frame-caption")).to_contain_text("33.0°")
        page.locator("#evidence-choices select").select_option("2")
        paused_frame(page, "#result-video", 222)

        pw.expect(look).to_contain_text("Rep 3 was shallower than rep 1")
        look.click()
        paused_frame(page, "#result-video", 222)
        paused_frame(page, "#reference-video", 54)
        pw.expect(page.locator("#frame-caption")).to_contain_text("Hip above knee")
        pw.expect(page.locator("#reference-caption")).to_contain_text("Hip below knee")
        annotated = sample_pixels(page)
        page.get_by_label("Show pose overlay").uncheck()
        paused_frame(page, "#result-video", 222)
        paused_frame(page, "#reference-video", 54)
        assert "squat_clean.mp4" in page.locator("#result-video").evaluate("v => v.currentSrc")
        assert sample_pixels(page) != annotated
        page.get_by_label("Show pose overlay").check()
        paused_frame(page, "#result-video", 222)
        assert sample_pixels(page) == annotated

        row = page.locator(".rep-event").nth(2)
        with page.expect_response(lambda res: res.url.endswith("squat-review?depth=above&descent=0.6&ascent=0.6&tolerance=0.05")):
            page.locator("#targets-panel summary").click()
            page.get_by_label("Hip position at the bottom").select_option("above")
            page.get_by_label("Descent (s)", exact=True).fill("0.6")
            page.get_by_label("Ascent (s)", exact=True).fill("0.6")
            page.get_by_label("Tempo tolerance").fill("0.05")
            page.get_by_role("button", name="Apply targets").click()
        pw.expect(row.locator(".check")).to_have_count(3)
        assert all("met" in s for s in row.locator(".check").all_text_contents())
        pw.expect(look).to_contain_text("Rep 1 differed from your depth target")
        look.click()
        paused_frame(page, "#result-video", 54)
        paused_frame(page, "#reference-video", 222)
        with page.expect_download() as download:
            page.get_by_role("link", name="Download this target review").click()
        saved = json.loads(Path(download.value.path()).read_text())
        assert saved["checks"]["3"][0]["status"] == "met"

        result_url = fixture_url + "/api/jobs/" + page.url.split("#job=")[1] + "/result"
        data = page.request.get(result_url).json()
        for phase in ("descent", "bottom", "ascent"):
            row.locator(".phase-strip").get_by_role("button", name=phase.capitalize()).press("Enter")
            paused_frame(page, "#result-video", data["reps"][2]["source_frame_" + phase])
            pw.expect(page.locator("#reference-frame")).to_be_hidden()
        row.get_by_role("button", name="Play Rep 3").click()
        page.wait_for_function("!document.querySelector('#result-video').paused")
        paused_frame(page, "#result-video", data["reps"][2]["source_frame_end"])
        look.click()
        paused_frame(page, "#reference-video", 222)
        note = page.locator(".rep-event").last
        pw.expect(note).to_contain_text("Depth, angles and tempo are uncertain")
        note.get_by_role("button", name="See frame").click()
        paused_frame(page, "#result-video", data["reps"][-1]["source_frame_gap"])

        page.reload()
        pw.expect(look).to_contain_text("depth target")
        page.locator("#targets-panel summary").click()
        page.get_by_role("button", name="Clear targets").click()
        pw.expect(look).to_contain_text("shallower")
        page.set_viewport_size({"width": 390, "height": 844})
        look.click()
        paused_frame(page, "#result-video", 222)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert errors == []
        browser.close()
