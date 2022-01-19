import imageio
import os
import torch
import color_jitter
import numpy as np
class VideoRecorder(object):
    def __init__(self, dir_name, height=448, width=448, camera_id=0, fps=25):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = 1
        self.fps = fps
        self.frames = []
        self.save_both_views = False # Enable to save both views

    def init(self, enabled=True):
        self.frames = []
        self.frames_2 = []

        self.enabled = self.dir_name is not None and enabled

    def record(self, env, mode=None):
        if self.enabled:
            frame = env.unwrapped.render_obs(
                mode='rgb_array',
                height=self.height,
                width=self.width,
                camera_id=self.camera_id
            )
            frame = frame[0]
            if self.save_both_views: frame_2 = frame[0]

            frame_2 = torch.FloatTensor(frame).permute(2,0,1).unsqueeze(0).cuda()
            if self.save_both_views: frame_2 = color_jitter.random_color_jitter(frame_2).div(255).squeeze(0).cpu().permute(1, 2, 0)

            if mode is not None and 'video' in mode:
                _env = env
                while 'video' not in _env.__class__.__name__.lower():
                    _env = _env.env
                frame = _env.apply_to(frame)
                if self.save_both_views: frame_2 = _env.apply_to(frame_2)
            self.frames.append(frame)
            if self.save_both_views: self.frames_2.append(frame_2)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            if self.save_both_views: path_2 = os.path.join(self.dir_name, '_x_'+file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
            if self.save_both_views: imageio.mimsave(path_2, self.frames_2, fps=self.fps)
