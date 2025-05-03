# NOTICE: This file includes modifications generated with the assistance of generative AI.
# Original code structure and logic by the project author.
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

class StateRecorder:
    """
    Records a multi-channel (num_channels x H x W) state each frame.
    Specifically we now expect:
      - channel 0: height map in [-1..1]
      - channel 1: delta-height map
      - channels [2..4]: overhead RGB image in [0..1]

    Then saves an MP4 where each frame has three subplots:
      (1) 3D surface of the height
      (2) 3D surface of the delta
      (3) overhead RGB image
    """

    def __init__(self, recorder_config: dict):
        self.config = recorder_config

        # (A) Parse shape info from config
        map_cfg = self.config["height_map"]
        self.H = map_cfg.get("grid_h", 50)
        self.W = map_cfg.get("grid_w", 50)
        self.num_channels = map_cfg.get("num_channels", 5)
        self.height_idx = map_cfg.get("height_channel_index", 0)
        self.clip_val = map_cfg.get("clip_value", None)

        # (B) Video config
        vid_cfg = self.config.get("video", {})
        self.fps = vid_cfg.get("fps", 30)
        self.output_path = os.path.join(
            os.path.dirname(__file__),  # Root of the project
            vid_cfg.get("output_path", "output_heightmap.mp4")
        )

        # (C) Auto-save
        self.auto_save_every = self.config.get("auto_save_every", None)
        self.frames = []

        # (D) We'll create coordinate grids for 3D surfaces
        # shape => (H, W). So X => [0..W-1], Y => [0..H-1]
        self.Y, self.X = np.meshgrid(
            np.arange(self.H),
            np.arange(self.W),
            indexing='ij'
        )
        # Now X, Y => shape (H, W).

    def record_frame(self, central_state_vec: np.ndarray):
        """
        central_state_vec => (num_channels * H * W).
        We'll reshape => (num_channels, H, W).

        channel 0 => height
        channel 1 => delta
        channels 2..4 => overhead R,G,B
        """
        total_size = central_state_vec.size
        expected_size = self.num_channels * self.H * self.W
        if total_size < expected_size:
            print(f"[StateRecorder] Not enough data => have {total_size}, expected={expected_size}")
            return

        data_3d = central_state_vec.reshape(self.num_channels, self.H, self.W)

        # 1) height map => channel 0
        height_map = data_3d[self.height_idx]

        # 2) delta-height => channel 1 (if present)
        #    We'll do a simple fallback if we only had 4 channels
        delta_map = None
        if self.num_channels >= 2:
            delta_map = data_3d[1]

        # Optionally clip the height values
        if self.clip_val is not None:
            np.clip(height_map, -self.clip_val, self.clip_val, out=height_map)
            if delta_map is not None:
                np.clip(delta_map, -self.clip_val, self.clip_val, out=delta_map)

        # 3) overhead RGB => channels [2..4]
        if self.num_channels >= 5:
            r_channel = data_3d[2]
            g_channel = data_3d[3]
            b_channel = data_3d[4]
            overhead_rgb = np.stack([r_channel, g_channel, b_channel], axis=-1)
        else:
            overhead_rgb = np.zeros((self.H, self.W, 3), dtype=np.float32)

        # 4) Save the frame data
        self.frames.append((height_map.copy(), delta_map.copy() if delta_map is not None else None, overhead_rgb.copy()))

        # 5) Possibly auto-save
        if (self.auto_save_every is not None) and (len(self.frames) >= self.auto_save_every):
            self.save_video()
            self.frames.clear()

    def save_video(self):
        """
        Creates an MP4 with three subplots:
          - left => 3D surface of height
          - middle => 3D surface of delta
          - right => overhead image
        """
        if len(self.frames) == 0:
            print("[StateRecorder] No frames to save.")
            return

        # Ensure FFmpeg is available
        if shutil.which("ffmpeg") is None:
            print("[StateRecorder] FFmpeg is not installed or not in PATH. Please install FFmpeg.")
            return

        # Ensure the output directory exists
        output_dir = os.path.dirname(self.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # We'll do a figure with 1 row, 3 columns
        fig = plt.figure(figsize=(16, 9))  # Increase figure size for higher resolution

        # (1) 3D surface => height
        ax_surf = fig.add_subplot(1, 3, 1, projection='3d')
        ax_surf.set_xlabel("X (columns)")
        ax_surf.set_ylabel("Y (rows)")
        ax_surf.set_zlabel("Height")
        ax_surf.set_zlim(-1, 1)

        # (2) 3D surface => delta
        ax_delta = fig.add_subplot(1, 3, 2, projection='3d')
        ax_delta.set_xlabel("X (columns)")
        ax_delta.set_ylabel("Y (rows)")
        ax_delta.set_zlabel("Delta")
        ax_delta.set_zlim(-1, 1)

        # (3) overhead image => 2D
        ax_img = fig.add_subplot(1, 3, 3)
        ax_img.set_title("Overhead RGB")
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        # Configure FFMpegWriter with higher quality settings
        writer = FFMpegWriter(
            fps=self.fps,
            metadata={"title": "State Recorder Video", "artist": "StateRecorder"},
            extra_args=[
                "-crf", "18",          # Set constant rate factor (lower is better quality)
              
                "-b:v", "5000k",       # Set video bitrate to 5000 kbps
                "-pix_fmt", "yuv420p"  # Ensure compatibility with most players
            ]
        )

        with writer.saving(fig, self.output_path, dpi=150):  # Reduce DPI if resolution is too high
            surf_plot = None
            delta_plot = None
            img_plot = None

            for i, (height_map, delta_map, overhead_rgb) in enumerate(self.frames):
                # A) Clear old surfaces
                if surf_plot is not None:
                    surf_plot.remove()
                if delta_plot is not None:
                    delta_plot.remove()

                # B) 3D surface => height
                surf_plot = ax_surf.plot_surface(
                    self.X, self.Y, height_map,
                    cmap='viridis', edgecolor='none'
                )
                ax_surf.set_title(f"Height")
                ax_surf.view_init(elev=40, azim=(45 + i * 0.3))

                # C) 3D surface => delta
                if delta_map is not None:
                    delta_plot = ax_delta.plot_surface(
                        self.X, self.Y, delta_map,
                        cmap='plasma', edgecolor='none'
                    )
                    ax_delta.set_title(f"Delta Height")
                    ax_delta.view_init(elev=40, azim=(45 + i * 0.3))

                # D) overhead RGB => 2D
                if img_plot is None:
                    img_plot = ax_img.imshow(overhead_rgb, vmin=0.0, vmax=1.0)
                else:
                    img_plot.set_data(overhead_rgb)

                # finalize
                writer.grab_frame()

        plt.close(fig)
        print(f"[StateRecorder] Video saved => {self.output_path}")
