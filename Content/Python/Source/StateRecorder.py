# NOTICE: This file includes modifications generated with the assistance of generative AI (VSCode Copilot Assistant).
# Original code structure and logic by the project author.
# The modifications are intended to enhance the functionality and performance of the code.
# The author has reviewed all changes for correctness.

# Fix OpenMP library conflict issues
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from typing import Dict, List, Any, Tuple
from datetime import datetime

class StateRecorder:
    def __init__(self, recorder_config: Dict[str, Any]):
        self.config = recorder_config
        self.is_enabled = self.config.get("enabled", False)
        if not self.is_enabled:
            print("[StateRecorder] Is disabled by config.")
            return

        video_cfg = self.config.get("video_settings", {})
        self.fps = video_cfg.get("fps", 15)
        # Ensure output_path is correct relative to Train.py if it's a relative path
        output_path_raw = video_cfg.get("output_path", "state_visualization.mp4")
        if not os.path.isabs(output_path_raw):
            # Assuming Train.py is in Content/Python, and output_path is relative to that
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Up to Content/Python
            self.base_output_path = os.path.join(base_dir, output_path_raw)
        else:
            self.base_output_path = output_path_raw
        
        # Store config for timestamping (applied per save_video call)
        self.add_timestamp = video_cfg.get("add_timestamp", True)
            
        self.dpi = video_cfg.get("dpi", 100)
        self.auto_save_every = self.config.get("auto_save_every", None)
        
        self.components_to_visualize: List[Dict[str, Any]] = []
        self.data_to_store_names: List[Tuple[str, str]] = [] # List of (source_key, component_name)

        plot_idx_counter = 0
        self.meshgrids: Dict[int, Tuple[np.ndarray, np.ndarray]] = {} # Store meshgrids by plot index

        for comp_def in self.config.get("components_to_record", []):
            if comp_def.get("enabled", False):
                self.data_to_store_names.append((comp_def["source_key"], comp_def["component_name"]))
                
                vis_type = comp_def.get("visualization_type", "none")
                if vis_type != "none":
                    comp_def["plot_idx"] = plot_idx_counter # Assign an index for subplotting
                    self.components_to_visualize.append(comp_def)
                    
                    if vis_type == "3d_surface":
                        shape_cfg = comp_def.get("shape_for_plot", {})
                        h = shape_cfg.get("h", 0)
                        w = shape_cfg.get("w", 0)
                        if h > 0 and w > 0:
                            Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
                            self.meshgrids[plot_idx_counter] = (X, Y) # Store X, Y for this plot
                        else:
                            print(f"[StateRecorder] Warning: 3D surface plot for '{comp_def['component_name']}'"
                                  f" needs 'shape_for_plot' with 'h' and 'w' > 0 in its config. Will attempt to infer from data.")
                    plot_idx_counter += 1
        
        self.num_subplots = len(self.components_to_visualize)
        self.frames: List[Dict[str, np.ndarray]] = [] # Stores dicts of component_name: data

        if self.is_enabled and self.num_subplots > 0 :
             timestamp_info = " (with timestamps)" if self.add_timestamp else ""
             print(f"[StateRecorder] Initialized. Will plot {self.num_subplots} components. Base output: {self.base_output_path}{timestamp_info}")
        elif self.is_enabled:
             print(f"[StateRecorder] Initialized, but no components configured for visualization. Will only store data if specified.")


    def record_frame(self, current_env_state_dict: Dict[str, Any]):
        """
        Records data for enabled components from the provided state dictionary.
        current_env_state_dict is for a single environment, with numpy arrays.
        e.g., {"central": {"height_map": np_array_HW, "gridobject_vectors": np_array_F}, 
               "agent": np_array_NA_Obs}
        """
        if not self.is_enabled:
            return

        frame_data: Dict[str, np.ndarray] = {}
        for source_key, component_name in self.data_to_store_names:
            if source_key in current_env_state_dict:
                source_data_dict_or_array = current_env_state_dict[source_key]
                data_to_store = None
                if isinstance(source_data_dict_or_array, dict): # e.g. "central"
                    data_to_store = source_data_dict_or_array.get(component_name)
                elif source_key == component_name: # e.g. "agent" data directly under its key
                    data_to_store = source_data_dict_or_array
                
                if data_to_store is not None:
                    # Ensure it's a numpy array and make a copy
                    frame_data[f"{source_key}_{component_name}"] = np.array(data_to_store, copy=True)
        
        if frame_data: # Only add if we actually stored something
            self.frames.append(frame_data)

        if self.auto_save_every is not None and len(self.frames) >= self.auto_save_every:
            self.save_video() # This will also clear frames


    def save_video(self):
        if not self.is_enabled or self.num_subplots == 0 or not self.frames:
            if self.is_enabled and not self.frames: print("[StateRecorder] Save Video: No frames to save.")
            elif self.is_enabled and self.num_subplots == 0: print("[StateRecorder] Save Video: No components configured for visualization.")
            self.frames.clear() # Clear frames even if not saving video
            return

        # Generate output path with fresh timestamp for each save_video call
        if self.add_timestamp:
            path_parts = os.path.splitext(self.base_output_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{path_parts[0]}_{timestamp}{path_parts[1]}"
        else:
            output_path = self.base_output_path

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Determine layout: try to make it squarish, or single row if few plots
        ncols = self.num_subplots
        nrows = 1
        if self.num_subplots > 3: # Example: if more than 3, try 2 columns
            ncols = 2 
            nrows = (self.num_subplots + 1) // 2
        if self.num_subplots > 4: # Example: if more than 4, try 3 columns
            ncols = 3
            nrows = (self.num_subplots + 2) // 3
            
        fig_width = 6 * ncols
        fig_height = 5 * nrows
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        axes_list = []
        for i, comp_viz_cfg in enumerate(self.components_to_visualize):
            is_3d = comp_viz_cfg.get("visualization_type") == "3d_surface"
            ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d' if is_3d else None)
            axes_list.append(ax)

        # Ensure ffmpeg path is explicitly set to avoid WinError 2
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path is None:
            print("[StateRecorder] Error: FFmpeg not found in PATH")
            plt.close(fig)
            self.frames.clear()
            return
        
        # On Windows, ensure the .exe extension is explicit
        import platform
        if platform.system() == 'Windows' and not ffmpeg_path.lower().endswith('.exe'):
            ffmpeg_path += '.exe'
            
        # Set matplotlib backend to Agg to avoid display issues
        plt.switch_backend('Agg')
            
        writer = FFMpegWriter(
            fps=self.fps,
            metadata={"title": "TerraShift State Visualization", "artist": "StateRecorder"},
            extra_args=["-crf", "20", "-b:v", "3000k", "-pix_fmt", "yuv420p"]
        )
        
        # Explicitly set the ffmpeg binary path
        writer.bin_path = lambda: ffmpeg_path
        
        print(f"[StateRecorder] Attempting to save video to: {output_path} with {len(self.frames)} frames.")
        print(f"[StateRecorder] Using ffmpeg at: {ffmpeg_path}")

        try:
            # Test if we can actually access the ffmpeg executable
            import subprocess
            try:
                result = subprocess.run([ffmpeg_path, '-version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode != 0:
                    print(f"[StateRecorder] Warning: FFmpeg test failed with return code {result.returncode}")
            except Exception as e:
                print(f"[StateRecorder] Warning: Cannot test ffmpeg executable: {e}")
            
            with writer.saving(fig, output_path, dpi=self.dpi):
                plot_artists = [None] * self.num_subplots # To store surface/image artists for removal/update

                for frame_idx, frame_data_dict in enumerate(self.frames):
                    for i, comp_viz_cfg in enumerate(self.components_to_visualize):
                        ax = axes_list[i]
                        ax.clear() # Clear previous frame's plot from this subplot
                        
                        data_key = f"{comp_viz_cfg['source_key']}_{comp_viz_cfg['component_name']}"
                        component_data = frame_data_dict.get(data_key)

                        if component_data is None:
                            ax.set_title(f"{comp_viz_cfg['plot_title']} (No Data)")
                            continue

                        ax.set_title(comp_viz_cfg["plot_title"])
                        vis_type = comp_viz_cfg["visualization_type"]

                        if vis_type == "3d_surface":
                            if component_data.ndim == 2:
                                H, W = component_data.shape
                                # Get or create meshgrid for this plot_idx (or specific H,W)
                                if comp_viz_cfg["plot_idx"] in self.meshgrids:
                                    X, Y = self.meshgrids[comp_viz_cfg["plot_idx"]]
                                    # Ensure meshgrid dimensions match data if dynamically inferred
                                    if X.shape[1] != W or Y.shape[0] != H :
                                        print(f"Warning: Meshgrid size mismatch for {data_key}. Recreating.")
                                        Y_new, X_new = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
                                        self.meshgrids[comp_viz_cfg["plot_idx"]] = (X_new, Y_new)
                                        X, Y = X_new, Y_new
                                elif comp_viz_cfg.get("shape_for_plot"): # Use explicitly defined shape if meshgrid wasn't precomputed
                                    shp = comp_viz_cfg["shape_for_plot"]
                                    Y_s, X_s = np.meshgrid(np.arange(shp["h"]), np.arange(shp["w"]), indexing='ij')
                                    X, Y = X_s, Y_s
                                else: # Fallback: infer from data (might be slow if changing per frame)
                                   Y_fb, X_fb = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
                                   X, Y = X_fb, Y_fb

                                surf = ax.plot_surface(
                                    X, Y, component_data,
                                    cmap=comp_viz_cfg.get("cmap_3d", "viridis"),
                                    edgecolor='none',
                                    vmin=comp_viz_cfg.get("val_range_2d", [-1.0, 1.0])[0], # Use val_range_2d for consistency or add specific 3d range
                                    vmax=comp_viz_cfg.get("val_range_2d", [-1.0, 1.0])[1]
                                )
                                ax.set_zlim(comp_viz_cfg.get("z_lim_3d", [-1.5, 1.5]))
                                plot_artists[i] = surf # Not strictly needed if clearing axes
                            else:
                                ax.text(0.5, 0.5, "Data not 2D for 3D plot", ha='center', va='center')
                                
                        elif vis_type == "2d_image_grey":
                            val_range = comp_viz_cfg.get("val_range_2d", [0.0, 1.0])
                            img = ax.imshow(
                                component_data,
                                cmap=comp_viz_cfg.get("cmap_2d", "gray"),
                                vmin=val_range[0], vmax=val_range[1]
                            )
                            ax.set_xticks([])
                            ax.set_yticks([])
                            plot_artists[i] = img # Not strictly needed if clearing axes
                        # Add other visualization types here ("none" is handled by not being in components_to_visualize)
                    
                    fig.suptitle(f"Frame {frame_idx + 1}/{len(self.frames)}", fontsize=12)
                    try:
                        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
                    except Exception: # Sometimes tight_layout fails with 3D plots
                        pass 
                    
                    try:
                        writer.grab_frame()
                    except Exception as e:
                        print(f"[StateRecorder] Error grabbing frame {frame_idx + 1}: {e}")
                        break 
            
        except Exception as e:
            print(f"[StateRecorder] Error initializing video writer: {e}")
            print("[StateRecorder] This usually means ffmpeg is not properly installed or configured,")
            print("[StateRecorder] or there's an issue with file permissions or output path.")
            plt.close(fig)
            self.frames.clear()
            return
            
        plt.close(fig)
        self.frames.clear() # Clear frames after saving or attempting to save

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"[StateRecorder] Video successfully saved to: {output_path}")
        else:
            print(f"[StateRecorder] Error: Video file was not created or is empty at {output_path}. Check FFMpeg messages or permissions.")