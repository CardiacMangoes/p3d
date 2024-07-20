import os
import sys
sys.path.append(os.getcwd() + "/submodules/OpenLRM")

import torch
import numpy as np
from accelerate import Accelerator

from openlrm.datasets.cam_utils import create_intrinsics, build_camera_principle, build_camera_standard
from openlrm.utils.hf_hub import wrap_model_hub
from openlrm.models import model_dict

import viser.transforms as tf

from p3d.models.base_model import NovelViewSynthesisModel
from tqdm import tqdm

class OpenLRM(NovelViewSynthesisModel):
    def load_model(self):
        with torch.no_grad():
            torch._dynamo.config.disable = True
            Accelerator()

            print("Accelerated")

            model_weight = 'zxhezexin/openlrm-mix-base-1.1'

            # build model
            hf_model_cls = wrap_model_hub(model_dict['lrm'])
            model = hf_model_cls.from_pretrained(model_weight).to(self.device)
            return model
    
    def process_image(self, image):
        canonical_rotation = torch.tensor([[1, 0, 0],
                                           [0, 0, -1],
                                           [0, 1, 0]], dtype=torch.float32)
        canonical_translation = torch.tensor([0, -2, 0], dtype=torch.float32)
        canonical_extrinsics = torch.cat([canonical_rotation, canonical_translation.unsqueeze(-1)], -1).float().unsqueeze(0)
        canonical_intrinsics = create_intrinsics(0.75, c=0.5).unsqueeze(0)

        source_camera = build_camera_principle(canonical_extrinsics, canonical_intrinsics).to(self.device)

        self.image = image.to(self.device)
        self.latent_repr = self.model.forward_planes(self.image, source_camera)

    def generate_images(self, camera_coordinates):
        with torch.no_grad():
            thetas, phis, psis = self.transform_camera_coordinates(camera_coordinates)

            renders = []
            for i in tqdm(range(len(camera_coordinates))):
                theta, phi, psi = thetas[i], phis[i], psis[i]
                
                x, y, z = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)
                position = np.array([x, y, z])

                roll = tf.SO3.from_z_radians(psi)
                dir  = tf.SO3.multiply(tf.SO3.from_z_radians(phi), tf.SO3.from_y_radians(theta))
                rotation = tf.SO3.multiply(dir, roll).as_matrix()

                rotation = torch.from_numpy(rotation)
                position = torch.from_numpy(position).unsqueeze(1)

                extrinsics = torch.cat([rotation, 2 * position], dim=1).unsqueeze(0).float()
                intrinsics = create_intrinsics(0.75, c=0.5).unsqueeze(0)
                camera_coordinates = build_camera_standard(extrinsics, intrinsics).unsqueeze(0).to(self.device)

                render_size = 266

                render_anchors = torch.zeros(1, camera_coordinates.shape[1], 2).to(self.device)
                render_resolutions = torch.ones(1, camera_coordinates.shape[1], 1).to(self.device) * render_size
                render_bg_colors = torch.ones(1, camera_coordinates.shape[1], 1, dtype=torch.float32).to(self.device) * 1.0

                for i in range(0, camera_coordinates.shape[0], 1):
                    outputs = self.model.synthesizer(
                                planes=self.latent_repr,
                                cameras=camera_coordinates[:, i:i+1],
                                anchors=render_anchors[:, i:i+1],
                                resolutions=render_resolutions[:, i:i+1],
                                bg_colors=render_bg_colors[:, i:i+1],
                                region_size=render_size,
                            )
                    frame = outputs['images_rgb'][0, 0].detach().cpu().permute(1, 2, 0)
                    renders.append(frame)

            return torch.stack(renders)
        

    def transform_camera_coordinates(self, camera_coordinates):
        # initial pos. is (90, 270, 0)
        thetas_, phis_, psis_ = camera_coordinates.T
        
        thetas = thetas_ / 180 * np.pi
        phis = phis_ / 180 * np.pi
        psis = (psis_ + 90) / 180 * np.pi
        return thetas, phis, psis