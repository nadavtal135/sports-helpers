"""Upload the bundled scene through the file input, then inspect its actual decoded frames."""

import shutil
from pathlib import Path

import pytest
from browser_helpers import paused_frame, sample_pixels

pw = pytest.importorskip("playwright.sync_api")


def test_uploaded_jump_to_landing_frames(fixture_url, tmp_path):
    with pw.sync_playwright() as playwright:
        executable = shutil.which("chromium") or playwright.chromium.executable_path
        if not Path(executable).is_file():
            pytest.skip("Install Chromium with python -m playwright install chromium")
        browser = playwright.chromium.launch(executable_path=executable, args=["--disable-dev-shm-usage"])
        page = browser.new_page(viewport={"width": 1140, "height": 900})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.set_default_timeout(20000)
        page.goto(fixture_url)
        page.get_by_role("button", name="Standing Jump Analysis").click()
        pw.expect(page.locator("#capture-tip")).to_contain_text("both feet visible")
        # The registered sample's exact bytes may also be selected as a normal upload.
        assert page.request.post(fixture_url + "/api/samples/standing_jump").ok
        page.locator("#file").set_input_files(tmp_path / "samples" / "standing_jump_sample.mp4")
        pw.expect(page.locator("#clip-info")).to_contain_text("standing_jump_sample")
        page.locator("#trim-start").fill("0.5")
        page.get_by_role("button", name="Analyze", exact=True).click()
        pw.expect(page.get_by_role("heading", name="Standing jump technique review")).to_be_visible(timeout=90000)
        pw.expect(page.locator("#tiles")).to_be_hidden()

        strength, look = page.locator("[data-note=strength]"), page.locator("[data-note=closer_look]")
        pw.expect(strength).to_contain_text("descent timing was similar")
        strength.click()
        paused_frame(page, "#result-video", 36)
        pw.expect(page.locator("#frame-caption")).to_contain_text("0.33 s")
        look.click()
        paused_frame(page, "#result-video", 238)
        paused_frame(page, "#reference-video", 67)
        pw.expect(page.locator("#frame-caption")).to_contain_text("Added after contact 17.6°")
        page.get_by_label("Inspect a frame").select_option("3")
        paused_frame(page, "#result-video", 231)
        pw.expect(page.locator("#frame-caption")).to_contain_text("7.70 s")
        picture = sample_pixels(page)
        page.get_by_label("Show pose overlay").uncheck()
        paused_frame(page, "#result-video", 231)
        assert sample_pixels(page) != picture
        page.get_by_role("button", name="Previous frame").click()
        paused_frame(page, "#result-video", 230)
        page.get_by_label("Show pose overlay").check()
        paused_frame(page, "#result-video", 230)
        page.get_by_role("button", name="Next frame").click()
        paused_frame(page, "#result-video", 231)
        assert sample_pixels(page) == picture

        row = page.locator("article[data-attempt='3']")
        for phase, frame in (("countermovement", 207), ("takeoff", 217), ("flight", 223),
                             ("contact", 231), ("landing_bottom", 238), ("recovery", 251)):
            row.locator(f".phase-strip button[data-phase={phase}]").press("Enter")
            paused_frame(page, "#result-video", frame)
        page.locator("article[data-attempt='2']").get_by_role("button", name="Replay recovery").click()
        page.wait_for_function("!document.querySelector('#result-video').paused")
        paused_frame(page, "#result-video", 174)
        hidden = page.locator("article[data-attempt='4']")
        pw.expect(hidden.locator(".missing-phase").filter(has_text="First contact")).to_contain_text("Unavailable")
        hidden.get_by_role("button", name="Inspect frames").first.click()
        paused_frame(page, "#result-video", 321)

        page.locator("#jump-targets summary").click()
        page.get_by_label("Measurement", exact=True).select_option("landing_bend_deg")
        page.get_by_label("Target minimum").fill("45")
        page.get_by_label("Target maximum").fill("65")
        page.get_by_role("button", name="Compare target").click()
        pw.expect(look).to_contain_text("Attempt 3 was below")
        pw.expect(hidden).to_contain_text("target · unknown")
        look.click()
        paused_frame(page, "#result-video", 231)
        paused_frame(page, "#reference-video", 238)
        page.reload()
        pw.expect(look).to_contain_text("Attempt 3 was below")
        page.set_viewport_size({"width": 390, "height": 844})
        look.click()
        paused_frame(page, "#result-video", 231)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert not errors
        browser.close()
