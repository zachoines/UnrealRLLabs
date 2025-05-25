# NOTICE: This file includes modifications generated with the assistance of generative AI (VSCode Copilot Assistant).
# Original code structure and logic by the project author.
# The modifications are intended to enhance the functionality and performance of the code.
# The author has reviewed all changes for correctness.
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

class StateRecorder:
    """
    Records a multi-channel (num_channels x H x W) state each frame.
    Assuming 2 channels after modifications:
      - channel 0: height map in [-1..1]
      - channel 1: luminance (grayscale) of overhead view in [0..1]

    Saves an MP4 where each frame has two subplots:
      (1) 3D surface of the height map
      (2) 2D grayscale image of the luminance channel
    """

    def __init__(self, recorder_config: dict):
        self.config = recorder_config

        map_cfg = self.config["height_map"]
        self.H = map_cfg.get("grid_h", 50)
        self.W = map_cfg.get("grid_w", 50)
        # Expecting 2 channels now: Height + Luminance
        self.num_channels = map_cfg.get("num_channels", 2)
        self.height_idx = map_cfg.get("height_channel_index", 0)
        # Luminance will be the channel after height if height_idx is 0
        self.luminance_idx = self.height_idx + 1

        self.clip_val = map_cfg.get("clip_value", None) # For height map

        vid_cfg = self.config.get("video", {})
        self.fps = vid_cfg.get("fps", 30)
        self.output_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), # Assumes Train.py is in Content/Python
            vid_cfg.get("output_path", "state_height_luminance.mp4")
        )

        self.auto_save_every = self.config.get("auto_save_every", None)
        self.frames: list[tuple[np.ndarray, np.ndarray]] = []

        self.Y, self.X = np.meshgrid(
            np.arange(self.H),
            np.arange(self.W),
            indexing='ij'
        )
        print(f"[StateRecorder] Initialized for {self.num_channels} channels (H:{self.H}, W:{self.W}). Output: {self.output_path}")


    def record_frame(self, central_state_vec: np.ndarray):
        """
        central_state_vec => (num_channels * H * W).
        Reshapes to (num_channels, H, W).

        channel 0 (self.height_idx)    => height
        channel 1 (self.luminance_idx) => luminance of overhead
        """
        total_size = central_state_vec.size
        expected_size = self.num_channels * self.H * self.W
        if total_size < expected_size:
            print(f"[StateRecorder] Record Frame: Not enough data. Have {total_size}, expected {expected_size} for {self.num_channels} channels.")
            return
        if self.num_channels < 2:
            print(f"[StateRecorder] Record Frame: Not enough configured channels ({self.num_channels}) to extract height and luminance.")
            return

        try:
            data_3d = central_state_vec.reshape(self.num_channels, self.H, self.W)
        except ValueError as e:
            print(f"[StateRecorder] Record Frame: Error reshaping central_state_vec (size {total_size}) to ({self.num_channels}, {self.H}, {self.W}). Error: {e}")
            return

        # 1) Height map
        if self.height_idx >= data_3d.shape[0]:
            print(f"[StateRecorder] Error: height_idx {self.height_idx} out of bounds for data_3d shape {data_3d.shape}.")
            return
        height_map = data_3d[self.height_idx]
        if self.clip_val is not None:
            np.clip(height_map, -self.clip_val, self.clip_val, out=height_map)

        # 2) Luminance map
        if self.luminance_idx >= data_3d.shape[0]:
            print(f"[StateRecorder] Warning: luminance_idx {self.luminance_idx} out of bounds for data_3d shape {data_3d.shape}. Using zeros for luminance.")
            luminance_map = np.zeros((self.H, self.W), dtype=np.float32)
        else:
            luminance_map = data_3d[self.luminance_idx]

        self.frames.append((height_map.copy(), luminance_map.copy()))

        if (self.auto_save_every is not None) and (len(self.frames) >= self.auto_save_every):
            self.save_video()
            self.frames.clear()

    def save_video(self):
        if len(self.frames) == 0:
            print("[StateRecorder] Save Video: No frames to save.")
            return

        if shutil.which("ffmpeg") is None:
            print("[StateRecorder] Save Video: FFmpeg is not installed or not in PATH. Please install FFmpeg.")
            return

        output_dir = os.path.dirname(self.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        fig = plt.figure(figsize=(12, 5.5)) # Adjusted for 2 plots, slightly less height

        ax_surf = fig.add_subplot(1, 2, 1, projection='3d')
        ax_surf.set_xlabel("X")
        ax_surf.set_ylabel("Y")
        ax_surf.set_zlabel("Height")
        ax_surf.set_zlim(-1.5, 1.5)

        ax_img = fig.add_subplot(1, 2, 2)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        writer = FFMpegWriter(
            fps=self.fps,
            metadata={"title": "TerraShift State (Height + Luminance)", "artist": "StateRecorder"},
            extra_args=["-crf", "20", "-b:v", "3000k", "-pix_fmt", "yuv420p"] # Adjusted bitrate
        )
        
        print(f"[StateRecorder] Attempting to save video to: {self.output_path} with {len(self.frames)} frames.")

        with writer.saving(fig, self.output_path, dpi=100): # Adjusted DPI
            surf_plot = None
            img_plot = None

            for i, (height_map, luminance_map) in enumerate(self.frames):
                if surf_plot is not None:
                    surf_plot.remove()
                surf_plot = ax_surf.plot_surface(
                    self.X, self.Y, height_map,
                    cmap='viridis', edgecolor='none', vmin=-1.0, vmax=1.0
                )
                ax_surf.set_title(f"Height Map") # Removed frame number for cleaner look per frame

                if img_plot is None:
                    img_plot = ax_img.imshow(luminance_map, cmap='gray', vmin=0.0, vmax=1.0)
                else:
                    img_plot.set_data(luminance_map)
                ax_img.set_title(f"Overhead Luminance")

                fig.suptitle(f"Frame {i+1}/{len(self.frames)}", fontsize=12)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
                
                try:
                    writer.grab_frame()
                except Exception as e:
                    print(f"[StateRecorder] Error grabbing frame {i+1}: {e}")
                    # Decide if you want to break or continue
                    break 
            
        plt.close(fig)
        # Check if the file was actually created and has size
        if os.path.exists(self.output_path) and os.path.getsize(self.output_path) > 0:
            print(f"[StateRecorder] Video successfully saved to: {self.output_path}")
        else:
            print(f"[StateRecorder] Error: Video file was not created or is empty at {self.output_path}. Check FFMpeg messages or permissions.")