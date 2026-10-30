import sys
import itertools
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from PIL import Image

NX, NY = 300, 200
DX = 1.0
C = 1.0
DT = 0.5 * DX / C
STEPS_PER_FRAME = 5
SPONGE = 30
VMAX = 0.3

GIF_SECONDS = 10
GIF_FPS = 20
GIF_SCALE = 1.5
PALETTE_LEVELS = 64

CMAP = plt.get_cmap("RdBu_r")
NORM = colors.Normalize(-VMAX, VMAX)
WALL_RGB = (170, 170, 170)
SOURCE_RGB = (0, 0, 0)


class Sim:
    def __init__(self):
        self.p = np.zeros((NY, NX))
        self.p_prev = np.zeros((NY, NX))
        self.phase = 0.0
        self.freq = 0.04
        self.src = [50, NY // 2]
        self.paused = False
        self.coef = (C * DT / DX) ** 2

        self.wall = np.zeros((NY, NX), dtype=bool)
        wall_x, wall_w = 150, 4
        self.wall[:, wall_x:wall_x + wall_w] = True
        for slit_center in (NY // 2 - 25, NY // 2 + 25):
            self.wall[slit_center - 6:slit_center + 6, wall_x:wall_x + wall_w] = False

        yy, xx = np.mgrid[0:NY, 0:NX]
        edge = np.minimum.reduce([xx, NX - 1 - xx, yy, NY - 1 - yy])
        self.damping = 1.0 - 0.08 * np.clip((SPONGE - edge) / SPONGE, 0, 1) ** 2

    def step(self):
        p, p_prev = self.p, self.p_prev
        laplacian = (
            np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0) +
            np.roll(p, 1, axis=1) + np.roll(p, -1, axis=1) - 4 * p
        )
        p_next = 2 * p - p_prev + self.coef * laplacian
        p_next[self.src[1], self.src[0]] += 0.5 * np.sin(self.phase)
        self.phase = (self.phase + 2 * np.pi * self.freq * DT) % (2 * np.pi)
        p_next[self.wall] = 0.0
        p_next *= self.damping
        p *= self.damping
        self.p_prev, self.p = p, p_next

    def clear(self):
        self.p[:] = 0.0
        self.p_prev[:] = 0.0


def make_palette_image():
    palette = (CMAP(np.linspace(0, 1, PALETTE_LEVELS))[:, :3] * 255).astype(np.uint8).tolist()
    palette += [list(WALL_RGB), list(SOURCE_RGB)]
    flat = [channel for color in palette for channel in color]
    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(flat)
    return pal_img


PALETTE_IMAGE = make_palette_image()


def render_frame(sim):
    rgb = (CMAP(NORM(sim.p))[:, :, :3] * 255).astype(np.uint8)
    rgb[sim.wall] = WALL_RGB
    sx, sy = sim.src
    yy, xx = np.ogrid[:NY, :NX]
    rgb[(yy - sy) ** 2 + (xx - sx) ** 2 <= 9] = SOURCE_RGB
    img = Image.fromarray(np.ascontiguousarray(rgb[::-1]))
    img = img.resize((int(NX * GIF_SCALE), int(NY * GIF_SCALE)), Image.BILINEAR)
    return img.quantize(palette=PALETTE_IMAGE, dither=0)


def save_gif(frames, path):
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / GIF_FPS),
        loop=0,
        disposal=1,
    )


def build_ui():
    matplotlib.rcParams["toolbar"] = "None"
    sim = Sim()
    rec = {"active": False, "frames": []}
    drag = {"on": False}
    total_frames = GIF_SECONDS * GIF_FPS
    idle_title = "Click and drag on the plot to move the sound source"

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(left=0.08, right=0.93, top=0.93, bottom=0.25)

    img = ax.imshow(sim.p, cmap="RdBu_r", vmin=-VMAX, vmax=VMAX,
                    origin="lower", interpolation="bilinear")
    ax.imshow(np.ma.masked_where(~sim.wall, sim.wall), cmap="gray",
              origin="lower", vmin=0, vmax=1.5)
    marker, = ax.plot([sim.src[0]], [sim.src[1]], "k*", markersize=14)
    ax.set_xlim(-0.5, NX - 0.5)
    ax.set_ylim(-0.5, NY - 0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(idle_title)
    fig.colorbar(img, ax=ax, label="Pressure (red = compression, blue = rarefaction)")

    slider_ax = fig.add_axes([0.15, 0.14, 0.55, 0.03])
    pause_ax = fig.add_axes([0.15, 0.04, 0.13, 0.06])
    clear_ax = fig.add_axes([0.31, 0.04, 0.13, 0.06])
    record_ax = fig.add_axes([0.47, 0.04, 0.23, 0.06])

    freq_slider = Slider(slider_ax, "Frequency", 0.015, 0.08, valinit=sim.freq, valfmt="%.3f")
    pause_btn = Button(pause_ax, "Pause")
    clear_btn = Button(clear_ax, "Clear waves")
    record_btn = Button(record_ax, f"Record {GIF_SECONDS} s GIF")

    def move_source(event):
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        sim.src[0] = int(np.clip(round(event.xdata), 1, NX - 2))
        sim.src[1] = int(np.clip(round(event.ydata), 1, NY - 2))
        marker.set_data([sim.src[0]], [sim.src[1]])

    def on_press(event):
        if event.button == 1 and event.inaxes is ax:
            drag["on"] = True
            move_source(event)

    def on_motion(event):
        if drag["on"]:
            move_source(event)

    def on_release(_event):
        drag["on"] = False

    def on_freq(value):
        sim.freq = value

    def on_pause(_event):
        sim.paused = not sim.paused
        pause_btn.label.set_text("Play" if sim.paused else "Pause")

    def on_clear(_event):
        sim.clear()

    def finish_recording():
        rec["active"] = False
        record_btn.label.set_text(f"Record {GIF_SECONDS} s GIF")
        ax.set_title("Saving GIF...")
        fig.canvas.draw()
        path = time.strftime("sound_waves_%Y%m%d_%H%M%S.gif")
        save_gif(rec["frames"], path)
        rec["frames"] = []
        ax.set_title(f"Saved {path}")
        print("Saved", path)

    def on_record(_event):
        if rec["active"]:
            rec["active"] = False
            rec["frames"] = []
            record_btn.label.set_text(f"Record {GIF_SECONDS} s GIF")
            ax.set_title("Recording cancelled")
        else:
            rec["frames"] = []
            rec["active"] = True
            record_btn.label.set_text("Cancel recording")

    def update(_frame):
        if not sim.paused:
            for _ in range(STEPS_PER_FRAME):
                sim.step()
            if rec["active"]:
                rec["frames"].append(render_frame(sim))
                done = len(rec["frames"])
                if done >= total_frames:
                    finish_recording()
                else:
                    ax.set_title(f"Recording  {done / GIF_FPS:.1f} / {GIF_SECONDS} s")
        img.set_data(sim.p)
        return (img,)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    freq_slider.on_changed(on_freq)
    pause_btn.on_clicked(on_pause)
    clear_btn.on_clicked(on_clear)
    record_btn.on_clicked(on_record)

    anim = FuncAnimation(fig, update, frames=itertools.count(), interval=30,
                         blit=False, cache_frame_data=False)

    return {
        "fig": fig, "ax": ax, "sim": sim, "rec": rec, "anim": anim,
        "update": update, "on_record": on_record, "widgets": (freq_slider, pause_btn, clear_btn, record_btn),
    }


def main():
    for name in ("MacOSX", "TkAgg", "QtAgg"):
        try:
            plt.switch_backend(name)
            break
        except Exception:
            continue
    else:
        sys.exit(
            "No interactive window backend found. Install one, then run again:\n"
            "  pip install PyQt6\n"
            "or on Linux:  sudo apt install python3-tk"
        )
    ui = build_ui()
    plt.show()


if __name__ == "__main__":
    main()