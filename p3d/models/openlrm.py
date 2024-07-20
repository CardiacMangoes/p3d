import torch
from accelerate import Accelerator

from openlrm.datasets.cam_utils import create_intrinsics, build_camera_principle
from openlrm.utils.hf_hub import wrap_model_hub

from openlrm.models import model_dict

from p3d.models.base_model import NovelViewSynthesisModel

class OpenLRM(NovelViewSynthesisModel):
    def load_model(self):
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
        canonical_translation = torch.tensor([0, -1, 0], dtype=torch.float32)
        canonical_extrinsics = torch.cat([canonical_rotation, canonical_translation.unsqueeze(-1)], -1).float().unsqueeze(0)
        canonical_intrinsics = create_intrinsics(0.75, c=0.5).unsqueeze(0)

        source_camera = build_camera_principle(canonical_extrinsics, canonical_intrinsics).to(self.device)

        self.image = image.to(self.device)
        self.latent_repr = self.model.forward_planes(self.image, source_camera)

    def generate_image(self, camera_coordinates):
        render_size = 288

        camera_coordinates = camera_coordinates.to(self.device)

        render_anchors = torch.zeros(1, camera_coordinates.shape[1], 2).to(self.device)
        render_resolutions = torch.ones(1, camera_coordinates.shape[1], 1).to(self.device) * render_size
        render_bg_colors = torch.ones(1, camera_coordinates.shape[1], 1, dtype=torch.float32).to(self.device) * 1.0

        frames = []
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
            frames.append(frame)

        return frames
        

    def transform_camera_coordinates(self, camera_coordinates):
        return camera_coordinates