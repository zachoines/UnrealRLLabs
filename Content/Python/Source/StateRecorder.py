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
    After removing the delta height map, we now expect:
      - channel 0: height map in [-1..1]
      - channels [1..3]: overhead RGB image in [0..1] (R, G, B respectively)

    Then saves an MP4 where each frame has two subplots:
      (1) 3D surface of the height
      (2) overhead RGB image
    """

    def __init__(self, recorder_config: dict):
        self.config = recorder_config

        # (A) Parse shape info from config
        map_cfg = self.config["height_map"]
        self.H = map_cfg.get("grid_h", 50)
        self.W = map_cfg.get("grid_w", 50)
        # Expecting 4 channels now: Height + R + G + B
        self.num_channels = map_cfg.get("num_channels", 4)
        self.height_idx = map_cfg.get("height_channel_index", 0) # Should still be 0
        self.clip_val = map_cfg.get("clip_value", None)

        # (B) Video config
        vid_cfg = self.config.get("video", {})
        self.fps = vid_cfg.get("fps", 30)
        self.output_path = os.path.join(
            # Assuming Train.py is in Content/Python and StateRecorder is in Content/Python/Source
            # To save in Content/Python/
            os.path.dirname(os.path.dirname(__file__)), # Moves up one level from Source to Python
            vid_cfg.get("output_path", "height_auto.mp4")
        )


        # (C) Auto-save
        self.auto_save_every = self.config.get("auto_save_every", None)
        self.frames = [] # Will now store tuples of (height_map, overhead_rgb)

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
        channels 1,2,3 => overhead R,G,B
        """
        total_size = central_state_vec.size
        # Update expected size for 4 channels
        expected_size = self.num_channels * self.H * self.W
        if total_size < expected_size:
            print(f"[StateRecorder] Not enough data => have {total_size}, expected={expected_size} for {self.num_channels} channels.")
            return

        # Reshape based on the configured number of channels (should be 4)
        data_3d = central_state_vec.reshape(self.num_channels, self.H, self.W)

        # 1) height map => channel 0
        height_map = data_3d[self.height_idx]

        # 2) Delta-height map is removed.

        # Optionally clip the height values
        if self.clip_val is not None:
            np.clip(height_map, -self.clip_val, self.clip_val, out=height_map)

        # 3) overhead RGB => channels [1..3]
        # Ensure we have enough channels for RGB (at least 1 for R, 2 for G, 3 for B, assuming height is 0)
        if self.num_channels >= (self.height_idx + 4) or (self.height_idx == 0 and self.num_channels >=4) : # More robust check
            r_channel_idx = self.height_idx + 1
            g_channel_idx = self.height_idx + 2
            b_channel_idx = self.height_idx + 3

            r_channel = data_3d[r_channel_idx]
            g_channel = data_3d[g_channel_idx]
            b_channel = data_3d[b_channel_idx]
            overhead_rgb = np.stack([r_channel, g_channel, b_channel], axis=-1)
        else:
            print(f"[StateRecorder] Warning: Not enough channels for RGB. Expected {self.height_idx + 3} channels, got {self.num_channels}.")
            overhead_rgb = np.zeros((self.H, self.W, 3), dtype=np.float32)


        # 4) Save the frame data (height_map, overhead_rgb)
        self.frames.append((height_map.copy(), overhead_rgb.copy()))

        # 5) Possibly auto-save
        if (self.auto_save_every is not None) and (len(self.frames) >= self.auto_save_every):
            self.save_video()
            self.frames.clear()

    def save_video(self):
        """
        Creates an MP4 with two subplots:
          - left => 3D surface of height
          - right => overhead image
        """
        if len(self.frames) == 0:
            print("[StateRecorder] No frames to save.")
            return

        if shutil.which("ffmpeg") is None:
            print("[StateRecorder] FFmpeg is not installed or not in PATH. Please install FFmpeg.")
            return

        output_dir = os.path.dirname(self.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Figure with 1 row, 2 columns
        fig = plt.figure(figsize=(12, 7)) # Adjusted for 2 plots

        # (1) 3D surface => height
        ax_surf = fig.add_subplot(1, 2, 1, projection='3d')
        ax_surf.set_xlabel("X (columns)")
        ax_surf.set_ylabel("Y (rows)")
        ax_surf.set_zlabel("Height")
        ax_surf.set_zlim(-1.5, 1.5) # Adjusted zlim slightly if needed, original was -1,1

        # (2) overhead image => 2D
        ax_img = fig.add_subplot(1, 2, 2)
        ax_img.set_title("Overhead RGB")
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        writer = FFMpegWriter(
            fps=self.fps,
            metadata={"title": "TerraShift State Recorder", "artist": "StateRecorder"},
            extra_args=["-crf", "18", "-b:v", "5000k", "-pix_fmt", "yuv420p"]
        )

        with writer.saving(fig, self.output_path, dpi=150):
            surf_plot = None
            img_plot = None

            for i, (height_map, overhead_rgb) in enumerate(self.frames): # Iterate over new frame tuple
                # A) Clear old surface
                if surf_plot is not None:
                    surf_plot.remove()

                # B) 3D surface => height
                surf_plot = ax_surf.plot_surface(
                    self.X, self.Y, height_map,
                    cmap='viridis', edgecolor='none', vmin=-1.0, vmax=1.0 # Enforce consistent color mapping
                )
                ax_surf.set_title(f"Height Map (Frame {i+1})")
                ax_surf.view_init(elev=40, azim=(45 + i * 0.3)) # Dynamic view

                # C) Delta map subplot is removed.

                # D) overhead RGB => 2D
                if img_plot is None: # Initialize imshow object
                    img_plot = ax_img.imshow(overhead_rgb.clip(0,1), vmin=0.0, vmax=1.0) # Clip to ensure valid RGB
                else: # Update data for existing imshow object
                    img_plot.set_data(overhead_rgb.clip(0,1))
                ax_img.set_title(f"Overhead RGB (Frame {i+1})")

                writer.grab_frame()

        plt.close(fig)
        print(f"[StateRecorder] Video saved => {self.output_path}")