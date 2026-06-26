from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Cube2Equirec(nn.Module):
    def __init__(self, equ_h, equ_w, cube_length, CUDA=False, device_id=0):
        super(Cube2Equirec, self).__init__()
        self.cube_h = cube_length
        self.cube_w = cube_length
        self.equ_h = equ_h
        self.equ_w = equ_w
        self.fov = 90
        self.fov_rad = self.fov * np.pi / 180

        assert self.cube_w == self.cube_h
        self.radius = int(0.5 * cube_length)

        theta_start = math.pi - (math.pi / equ_w)
        theta_end = -math.pi
        theta_step = 2 * math.pi / equ_w
        theta_range = torch.arange(theta_start, theta_end, -theta_step)

        phi_start = 0.5 * math.pi - (0.5 * math.pi / equ_h)
        phi_end = -0.5 * math.pi
        phi_step = math.pi / equ_h
        phi_range = torch.arange(phi_start, phi_end, -phi_step)

        self.theta_map = theta_range.unsqueeze(0).repeat(equ_h, 1)
        self.phi_map = phi_range.unsqueeze(-1).repeat(1, equ_w)
        self.lonlat_map = torch.stack([self.theta_map, self.phi_map], dim=-1)

        # Project each face to 3D cube and convert to pixel coordinates
        # Face order: [back, down, front, left, right, up] => [0, 1, 2, 3, 4, 5]
        grid, orientation_mask = self.get_grid2()

        # register_buffer ensures .to(device) / .cuda() moves these automatically
        self.register_buffer('grid', grid)
        self.register_buffer('orientation_mask', orientation_mask)

        if CUDA:
            self.cuda(device_id)

    def get_grid2(self):
        x_3d = (self.radius * torch.cos(self.phi_map) * torch.sin(self.theta_map)).view(self.equ_h, self.equ_w, 1)
        y_3d = (self.radius * torch.sin(self.phi_map)).view(self.equ_h, self.equ_w, 1)
        z_3d = (self.radius * torch.cos(self.phi_map) * torch.cos(self.theta_map)).view(self.equ_h, self.equ_w, 1)

        self.grid_ball = torch.cat([x_3d, y_3d, z_3d], 2).view(self.equ_h, self.equ_w, 3)

        # Down
        radius_ratio_down = torch.abs(y_3d / self.radius)
        grid_down_raw = self.grid_ball / radius_ratio_down.view(self.equ_h, self.equ_w, 1).expand(-1, -1, 3)
        grid_down_w = (-grid_down_raw[:, :, 0].clone() / self.radius).unsqueeze(-1)
        grid_down_h = (-grid_down_raw[:, :, 2].clone() / self.radius).unsqueeze(-1)
        grid_down = torch.cat([grid_down_w, grid_down_h], 2).unsqueeze(0)
        mask_down = (((grid_down_w <= 1) * (grid_down_w >= -1)) * ((grid_down_h <= 1) * (grid_down_h >= -1)) *
                     (grid_down_raw[:, :, 1] == -self.radius).unsqueeze(2)).float()

        # Up
        radius_ratio_up = torch.abs(y_3d / self.radius)
        grid_up_raw = self.grid_ball / radius_ratio_up.view(self.equ_h, self.equ_w, 1).expand(-1, -1, 3)
        grid_up_w = (-grid_up_raw[:, :, 0].clone() / self.radius).unsqueeze(-1)
        grid_up_h = (grid_up_raw[:, :, 2].clone() / self.radius).unsqueeze(-1)
        grid_up = torch.cat([grid_up_w, grid_up_h], 2).unsqueeze(0)
        mask_up = (((grid_up_w <= 1) * (grid_up_w >= -1)) * ((grid_up_h <= 1) * (grid_up_h >= -1)) *
                   (grid_up_raw[:, :, 1] == self.radius).unsqueeze(2)).float()

        # Front
        radius_ratio_front = torch.abs(z_3d / self.radius)
        grid_front_raw = self.grid_ball / radius_ratio_front.view(self.equ_h, self.equ_w, 1).expand(-1, -1, 3)
        grid_front_w = (-grid_front_raw[:, :, 0].clone() / self.radius).unsqueeze(-1)
        grid_front_h = (-grid_front_raw[:, :, 1].clone() / self.radius).unsqueeze(-1)
        grid_front = torch.cat([grid_front_w, grid_front_h], 2).unsqueeze(0)
        mask_front = (((grid_front_w <= 1) * (grid_front_w >= -1)) * ((grid_front_h <= 1) * (grid_front_h >= -1)) *
                      (torch.round(grid_front_raw[:, :, 2]) == self.radius).unsqueeze(2)).float()

        # Back
        radius_ratio_back = torch.abs(z_3d / self.radius)
        grid_back_raw = self.grid_ball / radius_ratio_back.view(self.equ_h, self.equ_w, 1).expand(-1, -1, 3)
        grid_back_w = (grid_back_raw[:, :, 0].clone() / self.radius).unsqueeze(-1)
        grid_back_h = (-grid_back_raw[:, :, 1].clone() / self.radius).unsqueeze(-1)
        grid_back = torch.cat([grid_back_w, grid_back_h], 2).unsqueeze(0)
        mask_back = (((grid_back_w <= 1) * (grid_back_w >= -1)) * ((grid_back_h <= 1) * (grid_back_h >= -1)) *
                     (torch.round(grid_back_raw[:, :, 2]) == -self.radius).unsqueeze(2)).float()

        # Right
        radius_ratio_right = torch.abs(x_3d / self.radius)
        grid_right_raw = self.grid_ball / radius_ratio_right.view(self.equ_h, self.equ_w, 1).expand(-1, -1, 3)
        grid_right_w = (-grid_right_raw[:, :, 2].clone() / self.radius).unsqueeze(-1)
        grid_right_h = (-grid_right_raw[:, :, 1].clone() / self.radius).unsqueeze(-1)
        grid_right = torch.cat([grid_right_w, grid_right_h], 2).unsqueeze(0)
        mask_right = (((grid_right_w <= 1) * (grid_right_w >= -1)) * ((grid_right_h <= 1) * (grid_right_h >= -1)) *
                      (torch.round(grid_right_raw[:, :, 0]) == -self.radius).unsqueeze(2)).float()

        # Left
        radius_ratio_left = torch.abs(x_3d / self.radius)
        grid_left_raw = self.grid_ball / radius_ratio_left.view(self.equ_h, self.equ_w, 1).expand(-1, -1, 3)
        grid_left_w = (grid_left_raw[:, :, 2].clone() / self.radius).unsqueeze(-1)
        grid_left_h = (-grid_left_raw[:, :, 1].clone() / self.radius).unsqueeze(-1)
        grid_left = torch.cat([grid_left_w, grid_left_h], 2).unsqueeze(0)
        mask_left = (((grid_left_w <= 1) * (grid_left_w >= -1)) * ((grid_left_h <= 1) * (grid_left_h >= -1)) *
                     (torch.round(grid_left_raw[:, :, 0]) == self.radius).unsqueeze(2)).float()

        # Face map: [back=0, down=1, front=2, left=3, right=4, up=5]
        orientation_mask = (mask_back * 0 + mask_down * 1 + mask_front * 2 +
                            mask_left * 3 + mask_right * 4 + mask_up * 5)

        return torch.cat([grid_back, grid_down, grid_front, grid_left, grid_right, grid_up], 0), orientation_mask

    def _ToEquirec(self, batch, mode):
        batch_size, ch, H, W = batch.shape
        if batch_size != 6:
            raise ValueError("Batch size mismatch!!")

        device = self.grid.device
        output = torch.zeros(1, ch, self.equ_h, self.equ_w, device=device)

        for ori in range(6):
            grid = self.grid[ori].unsqueeze(0)                          # (1, equ_h, equ_w, 2)
            mask = (self.orientation_mask == ori).unsqueeze(0)          # (1, equ_h, equ_w, 1)
            masked_grid = grid * mask.float().expand(-1, -1, -1, 2)    # (1, equ_h, equ_w, 2)
            source_image = batch[ori].unsqueeze(0).float()              # (1, ch, H, W)

            sampled = F.grid_sample(
                source_image, masked_grid, mode=mode, align_corners=False
            )                                                            # (1, ch, equ_h, equ_w)

            mask_2d = mask.float().view(1, 1, self.equ_h, self.equ_w).expand(1, ch, -1, -1)
            output = output + sampled * mask_2d

        return output

    def ToEquirecTensor(self, batch, mode='bilinear'):
        assert mode in ['nearest', 'bilinear']
        batch_size = batch.size(0)
        if batch_size % 6 != 0:
            raise ValueError("Batch size should be 6x")

        processed = [self._ToEquirec(batch[i * 6:(i + 1) * 6], mode)
                     for i in range(batch_size // 6)]
        return torch.cat(processed, 0)

    def forward(self, batch, mode='bilinear'):
        return self.ToEquirecTensor(batch, mode)
