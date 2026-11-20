"""Check the decoded video clock and picture in fixture browser tests."""


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
