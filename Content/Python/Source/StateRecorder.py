import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

class StateRecorder:
    """
    Records a 2D height map (H x W) each frame, then saves to a 3D plot MP4.
    Assumes your height data is already in [-1..1].
    """

    def __init__(self, recorder_config: dict):
        self.config = recorder_config

        # (A) Parse shape
        map_cfg = self.config["height_map"]
        self.H = map_cfg.get("grid_h", 50)
        self.W = map_cfg.get("grid_w", 50)
        self.num_channels = map_cfg.get("num_channels", 1)
        self.height_idx = map_cfg.get("height_channel_index", 0)
        self.clip_val = map_cfg.get("clip_value", None)

        # (B) Video config
        vid_cfg = self.config.get("video", {})
        self.fps = vid_cfg.get("fps", 30)
        self.output_path = vid_cfg.get("output_path", "output_heightmap.mp4")

        # (C) Auto-save
        self.auto_save_every = self.config.get("auto_save_every", None)
        self.frames = []

        # (D) We'll create X, Y for 3D surface
        #     shape => (H, W). So X-> [0..W-1] horizontally, Y-> [0..H-1]
        self.Y, self.X = np.meshgrid(
            np.arange(self.H),
            np.arange(self.W),
            indexing='ij'
        )
        # Now X, Y => shape (H, W).
        # We'll keep the data as shape (H, W) so everything lines up.

    def record_frame(self, central_state_vec: np.ndarray):
        """
        central_state_vec => flatten array with shape (num_channels*H*W).
        We'll reshape => (num_channels, H, W), then extract [height_idx].
        """
        data_3d = central_state_vec.reshape(self.num_channels, self.H, self.W)

        # Extract the chosen channel => shape (H,W)
        height_map = data_3d[self.height_idx]

        # Optionally clip to e.g. [-1..1]
        if self.clip_val is not None:
            np.clip(height_map, -self.clip_val, self.clip_val, out=height_map)

        # Append for later video writing
        self.frames.append(np.copy(height_map))

        # Possibly auto-save
        if (self.auto_save_every is not None) and (len(self.frames) >= self.auto_save_every):
            self.save_video()
            self.frames.clear()

    def save_video(self):
        """ Creates an MP4 of all frames. """
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel("X (columns)")
        ax.set_ylabel("Y (rows)")
        ax.set_zlabel("Height")
        ax.set_zlim(-1, 1)

        writer = FFMpegWriter(fps=self.fps)
        with writer.saving(fig, self.output_path, dpi=100):
            surf = None
            for i, height_map in enumerate(self.frames):
                if surf is not None:
                    surf.remove()

                # Plot shape => X,Y, height_map all (H,W).
                # The X, Y we made => (H,W). The data is also (H,W).
                surf = ax.plot_surface(
                    self.X, self.Y, height_map,
                    cmap='viridis', edgecolor='none'
                )
                ax.set_title(f"Frame {i+1}/{len(self.frames)}")
                # Adjust camera if you like:
                ax.view_init(elev=40, azim=45 + i*0.2)

                writer.grab_frame()

        plt.close(fig)
        print(f"[StateRecorder] Video saved => {self.output_path}")
