import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

class StateRecorder:
    """
    Records a multi-channel (num_channels x H x W) state each frame.
    Specifically expects:
      - channel 0: height map in [-1..1]
      - channels [1..3]: overhead RGB image in [0..1]
    Then saves an MP4 where each frame is drawn with:
      - a 3D surface of the height
      - a 2D image of the RGB overhead
    """

    def __init__(self, recorder_config: dict):
        self.config = recorder_config

        # (A) Parse shape info from config
        map_cfg = self.config["height_map"]
        self.H = map_cfg.get("grid_h", 50)                  # image height
        self.W = map_cfg.get("grid_w", 50)                  # image width
        self.num_channels = map_cfg.get("num_channels", 4)  # e.g., 4 = 1 for height + 3 for rgb
        self.height_idx = map_cfg.get("height_channel_index", 0)
        self.clip_val = map_cfg.get("clip_value", None)

        # (B) Video config
        vid_cfg = self.config.get("video", {})
        self.fps = vid_cfg.get("fps", 30)
        self.output_path = vid_cfg.get("output_path", "output_heightmap.mp4")

        # (C) Auto-save logic
        self.auto_save_every = self.config.get("auto_save_every", None)
        self.frames = []

        # (D) We'll create coordinate grids for the 3D surface plot
        #     shape => (H, W). So X -> [0..W-1], Y -> [0..H-1]
        self.Y, self.X = np.meshgrid(
            np.arange(self.H),
            np.arange(self.W),
            indexing='ij'
        )
        # Now X, Y => shape (H, W), matching the shape for height data.

    def record_frame(self, central_state_vec: np.ndarray):
        """
        Expects a flattened array of length (num_channels * H * W).
        We'll do:
          data_3d = central_state_vec.reshape(num_channels, H, W)

          data_3d[0] => height
          data_3d[1..3] => overhead RGB
        """
        total_size = central_state_vec.size
        expected_size = self.num_channels * self.H * self.W
        if total_size < expected_size:
            print(f"[StateRecorder] Not enough data => have {total_size}, expected={expected_size}")
            return

        # 1) Reshape into (num_channels, H, W)
        data_3d = central_state_vec.reshape(self.num_channels, self.H, self.W)

        # 2) Extract the height map (channel 0)
        height_map = data_3d[self.height_idx]

        # 3) Optionally clip the height values
        if self.clip_val is not None:
            np.clip(height_map, -self.clip_val, self.clip_val, out=height_map)

        # 4) Extract overhead RGB from channels [1..3]
        #    We'll assume channels => 1=R, 2=G, 3=B
        if self.num_channels >= 4:
            # data_3d = data_3d[::-1, :, :]  # reverse the row order
            r_channel = data_3d[1]
            g_channel = data_3d[2]
            b_channel = data_3d[3]

            # We'll stack them into shape (H, W, 3)
            overhead_rgb = np.stack([r_channel, g_channel, b_channel], axis=-1)
            
        else:
            # fallback => a black image if no RGB
            overhead_rgb = np.zeros((self.H, self.W, 3), dtype=np.float32)

        # 5) Store for later usage
        #    We'll keep the entire frame as a tuple: (height_map, overhead_rgb)
        #    Make a copy if you plan to modify it in place later
        self.frames.append((np.copy(height_map), np.copy(overhead_rgb)))

        # 6) Possibly auto-save
        if (self.auto_save_every is not None) and (len(self.frames) >= self.auto_save_every):
            self.save_video()
            self.frames.clear()

    def save_video(self):
        """
        Creates an MP4 with side-by-side subplots:
          - left: 3D surface of height
          - right: overhead image
        """
        if len(self.frames) == 0:
            print("[StateRecorder] No frames to save.")
            return

        # We'll do a figure with 2 subplots
        fig = plt.figure(figsize=(10, 5))

        # Left => 3D surface for height
        ax_surf = fig.add_subplot(1, 2, 1, projection='3d')
        ax_surf.set_xlabel("X (columns)")
        ax_surf.set_ylabel("Y (rows)")
        ax_surf.set_zlabel("Height")
        ax_surf.set_zlim(-1, 1)

        # Right => overhead image
        ax_img = fig.add_subplot(1, 2, 2)
        ax_img.set_title("Overhead RGB")
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        writer = FFMpegWriter(fps=self.fps)
        with writer.saving(fig, self.output_path, dpi=100):
            surf_plot = None
            img_plot = None

            for i, (height_map, overhead_rgb) in enumerate(self.frames):
                # (A) Clear old surf if it exists
                if surf_plot is not None:
                    surf_plot.remove()

                # (B) 3D surface of height
                surf_plot = ax_surf.plot_surface(
                    self.X, self.Y, height_map,
                    cmap='viridis', edgecolor='none'
                )
                ax_surf.set_title(f"Height (frame {i+1}/{len(self.frames)})")
                # Optionally rotate the camera each frame, e.g.:
                ax_surf.view_init(elev=40, azim=45 + i*0.5)

                # (C) overhead image
                # we can either create or update an imshow
                if img_plot is None:
                    # first time => create
                    img_plot = ax_img.imshow(overhead_rgb, vmin=0.0, vmax=1.0)
                else:
                    # update the data
                    img_plot.set_data(overhead_rgb)

                writer.grab_frame()

        plt.close(fig)
        print(f"[StateRecorder] Video saved => {self.output_path}")
        # Done!

