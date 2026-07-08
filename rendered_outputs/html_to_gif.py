import time
import imageio.v2 as imageio
from pathlib import Path
from playwright.sync_api import sync_playwright



HTML_FILE = "rl_agent_303_1500.html"
GIF_FILE = "gif/rl_agent_303_1500.gif"


NUM_FRAMES = 20
FPS = 3

WIDTH = 1250
HEIGHT = 555



html_path = Path(HTML_FILE).resolve()
frames_dir = Path("gif_frames")
frames_dir.mkdir(exist_ok=True)


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
    page.set_default_timeout(120000)

    page.goto(
        html_path.as_uri(),
        wait_until="domcontentloaded",
        timeout=120000
    )

    page.wait_for_timeout(8000)

    frames = []

    for i in range(NUM_FRAMES):
        frame_path = frames_dir / f"frame_{i:04d}.png"

        page.screenshot(
            path=str(frame_path),
            full_page=False,
            clip={
                "x": 97,
                "y": 95,
                "width": WIDTH,
                "height": HEIGHT,
            },
            timeout=120000,
        )

        frames.append(imageio.imread(frame_path))

        # lower FPS = fewer captures, smaller GIF
        page.wait_for_timeout(int(1000 / FPS))

    browser.close()

imageio.mimsave(GIF_FILE, frames, fps=FPS)

print("Saved GIF:", GIF_FILE)