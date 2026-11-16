"""Render available HumanML3D joints when an original preview is missing."""

from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .common import fingerprint, require

# HumanML3D's 22-joint T2M chains, in the supplied new_joints ordering.
CHAINS = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
          [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]


def render_joints(source, destination, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    source, destination = Path(source), Path(destination)
    require(not destination.is_symlink(), "Render destination must not be a dataset preview symlink")
    tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    require(all(tools.values()), "Fallback rendering needs FFmpeg and ffprobe in the active environment")
    encoders = subprocess.run([tools["ffmpeg"], "-hide_banner", "-encoders"], capture_output=True, text=True, check=True).stdout
    require(any(line.split()[1:2] == ["libx264"] for line in encoders.splitlines()),
            f"FFmpeg at {tools['ffmpeg']} needs libx264. Use this project's GPL FFmpeg conda environment.")
    facts = fingerprint(source)
    points = np.load(source, allow_pickle=False)
    if points.shape == (22, 3):
        points = points[None]
    require(points.ndim == 3 and points.shape[1:] == (22, 3) and len(points) > 0 and points.dtype.kind == "f" and np.isfinite(points).all(),
            f"Expected finite HumanML3D (T,22,3) joints: {source}")
    require(fingerprint(source)["sha256"] == facts["sha256"], f"Joints changed while loading: {source}")
    # HumanML3D is y-up. Reorder the display axes for Matplotlib's z-up view.
    display = points[..., [0, 2, 1]].astype(float)
    floor = float(display[..., 2].min())
    span = max(2.4, float(np.ptp(display - display[:, :1], axis=(0, 1)).max()) * 1.4)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-65)
    lines = [ax.plot([], [], [], linewidth=3, color=color)[0] for color in ("#c23b3b", "#315ba5", "#20252b", "#3c865b", "#8d56a3")]
    trail, = ax.plot([], [], [], color="#a0a0a0", linewidth=1)
    floor_line, = ax.plot([], [], [], color="#d4dce2", linewidth=1)
    ax.set_axis_off()
    fig.suptitle(title)
    def update(frame):
        pose = display[frame]
        x, y = pose[0, :2]
        ax.set_xlim(x - span / 2, x + span / 2)
        ax.set_ylim(y - span / 2, y + span / 2)
        ax.set_zlim(floor, floor + span)
        for line, chain in zip(lines, CHAINS):
            segment = pose[chain]
            line.set_data_3d(segment[:, 0], segment[:, 1], segment[:, 2])
        path = display[:frame + 1, 0]
        trail.set_data_3d(path[:, 0], path[:, 1], np.full(len(path), floor))
        floor_line.set_data_3d([x - span / 2, x + span / 2, x + span / 2, x - span / 2, x - span / 2],
                               [y - span / 2, y - span / 2, y + span / 2, y + span / 2, y - span / 2], [floor] * 5)
        return lines + [trail, floor_line]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".render-", suffix=".mp4", delete=False) as stream:
            temporary = Path(stream.name)
        animation = FuncAnimation(fig, update, frames=len(points), interval=50, blit=False)
        with matplotlib.rc_context({"animation.ffmpeg_path": tools["ffmpeg"]}):
            writer = FFMpegWriter(fps=20, codec="libx264", extra_args=["-crf", "18", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            animation.save(temporary, writer=writer, dpi=100)
        probe = json.loads(subprocess.run([tools["ffprobe"], "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                           "stream=codec_name,pix_fmt,avg_frame_rate,nb_read_frames,width,height", "-of", "json", str(temporary)],
                                          capture_output=True, text=True, check=True).stdout)["streams"][0]
        require(probe["codec_name"] == "h264" and probe["pix_fmt"] == "yuv420p" and Fraction(probe["avg_frame_rate"]) == 20
                and int(probe["nb_read_frames"]) == len(points), "Encoded fallback video has unexpected codec, rate or frame count")
        os.replace(temporary, destination)
    finally:
        plt.close(fig)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    versions = {name: subprocess.run([path, "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
                for name, path in tools.items()}
    return {"source": "rendered_joints", "input": facts, "output": fingerprint(destination), "fps": 20, "frames": len(points),
            "chains": CHAINS, "view": "Root-following y-up display; all supplied frames retained", "probe": probe,
            "tools": {name: {"path": path, "version": versions[name]} for name, path in tools.items()}}
