import face_alignment
import torch

import utils
from model import Model


class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    def nparams(self):
        """
        Calculates number of trainable params.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parse_batch(self, batch):
        """
        Moves batch to the model's device.
        """
        device = next(self.parameters()).device
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        return batch


class LatentCodeRotation(BaseModule):
    """
    Class for the network which maps latent code
    into another one with given face rotation landmarks.
    """
    def __init__(
        self,
        latent_dim=256,
        landmarks_dim=68*2,
        num_layers=3
    ):
        super(LatentCodeRotation, self).__init__()
        self.latent_dim = latent_dim
        self.landmarks_dim = landmarks_dim
        self.num_layers = num_layers

        projection_dim = self.latent_dim + self.landmarks_dim
        self.projection = torch.nn.Linear(projection_dim, self.latent_dim)
        self.linear_layers = torch.nn.Sequential(*[
            torch.nn.Sequential(*[
                torch.nn.Linear(self.latent_dim, self.latent_dim),
                torch.nn.BatchNorm1d(self.latent_dim),
                torch.nn.ReLU()
            ])
            for _ in range(self.num_layers)
        ])

    def forward(self, z, c):
        assert len(z.shape) == 2
        assert len(c.shape) == 3
        B = z.shape[0]
        z_and_c = torch.cat([z, c.reshape(B, -1)], dim=1)
        outputs = self.projection(z_and_c)
        return self.linear_layers(outputs)


class FaceAlignment(torch.nn.Module):
    def __init__(self, device='cuda'):
        super(FaceAlignment, self).__init__()
        self.device = device
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D,
            flip_input=False, device=device
        )

    def forward(self, image):
        with torch.no_grad():
            return torch.FloatTensor(
                self.fa.get_landmarks(image.permute(1, 2, 0).cpu().detach().numpy() * 255)[0]
            ).to(self.device)


class Critic(BaseModule):
    def __init__(
        self,
        latent_dim=256,
        landmarks_dim=68*2,
        num_layers=2
    ):
        super(Critic, self).__init__()
        self.latent_dim = latent_dim
        self.landmarks_dim = landmarks_dim
        self.num_layers = num_layers

        projection_dim = self.latent_dim + self.landmarks_dim
        self.projection = torch.nn.Linear(projection_dim, self.latent_dim)
        self.linear_layers = torch.nn.Sequential(*[
            torch.nn.Sequential(*[
                torch.nn.Linear(self.latent_dim, self.latent_dim),
                torch.nn.BatchNorm1d(self.latent_dim),
                torch.nn.ReLU()
            ])
            for _ in range(self.num_layers)
        ])
        self.output_layer = torch.nn.Linear(self.latent_dim, 1)

    def forward(self, z, c):
        assert len(z.shape) == 2
        assert len(c.shape) == 3
        B = z.shape[0]
        z_and_c = torch.cat([z, c.reshape(B, -1)], dim=1)
        outputs = self.projection(z_and_c).relu()
        outputs = self.linear_layers(outputs)
        outputs = self.output_layer(outputs)
        return outputs


class FaceRotationModel(Model, BaseModule):
    """
    ALAE-based model for facial keypoints transfer task.
    """
    def __init__(
        self,
        landmarks_dim=68*2,
        rotation_num_layers=3,
        critic_num_layers=2,
        face_alignment_device='cuda',
        **kwargs
    ):
        super(FaceRotationModel, self).__init__(**kwargs)
        self._backbone_args = kwargs

        self.rotation = LatentCodeRotation(
            latent_dim=self._backbone_args['latent_size'],
            landmarks_dim=landmarks_dim,
            num_layers=rotation_num_layers
        )
        self.critic = Critic(
            latent_dim=self._backbone_args['latent_size'],
            landmarks_dim=landmarks_dim,
            num_layers=critic_num_layers
        )
        self.face_alignment = FaceAlignment(device=face_alignment_device)

    def modify_z(self, z, c):
        return self.rotation(z, c)
    
    def rotate_face_from_z(self, z, c):
        assert len(z.shape) == 2
        z_rotated = self.modify_z(z, c).unsqueeze(1)
        z_rotated = z_rotated.repeat(1, 12, 1)
        return self.generate(z=z_rotated)

    def generate(self, z=None):
        with torch.no_grad():
            if isinstance(z, type(None)):
                device = next(self.parameters()).device
                z = torch.Tensor(1, self._backbone_args['latent_size']).normal_().to(device)
                z = self.mapping_fl(z)
            outputs = self.decoder(z, lod=5, blend=1, noise=None)
            return outputs.clamp(-1, 1)

    def load_pretrained_alae(self, f):
        d = torch.load(f, map_location='cpu')
        decoder = self.decoder
        encoder = self.encoder
        mapping_tl = self.mapping_tl
        mapping_fl = self.mapping_fl
        dlatent_avg = self.dlatent_avg
        
        model_dict = {
            'discriminator_s': encoder,
            'generator_s': decoder,
            'mapping_tl_s': mapping_tl,
            'mapping_fl_s': mapping_fl,
            'dlatent_avg': dlatent_avg
        }
        for key in model_dict.keys():
            model_dict[key].load_state_dict(d['models'][key])
    
    def get_keypoints_from_batch(self, batch):
        keypoints = torch.stack([
            self.face_alignment(image) for image in batch
        ])
        return keypoints

    def _generate_keypoints_from_z(self, z):
        assert len(z.shape) == 2
        with torch.no_grad():
            z = z.unsqueeze(1)
            z = z.repeat(1, 12, 1)
            x = self.generate(z=z)
            return self.get_keypoints_from_batch(x)

    def encode(self, x):
        return self.encoder(x, lod=5, blend=1).squeeze(dim=1)

    def _encode_ci(self, ci):
        z = torch.stack([self.encode(ci_) for ci_ in ci]).mean(dim=1)
        return z

    def forward(self, x):
        """
        Makes forward pass for training purposes.
        :param x: batch dictionary with keys: 'image', 'keypoints', 'ci'
        :return: dictionary of multiple outputs:
            'critic_outputs_t': critic outputs on source-to-target modified latent codes z_t_hat with given target keypoints c_t
            'critic_outputs_s': critic outputs on source latent code z_t and its keypoints c_s
            'z_t': z_t
            'z_t_hat': z_t_hat
            'z_s': z_s
            'z_s_restored': z_s_restored
        """
        images, c_t, ci = x['image'], x['keypoints'], x['ci']
        z_s = self._encode_ci(ci)
        c_s = self._generate_keypoints_from_z(z_s)

        z_t = self.encode(images)
        z_t_hat = self.modify_z(z=z_s, c=c_t)
        z_s_restored = self.modify_z(z_t, c_s)

        critic_outputs_t = self.critic(z_t_hat, c_t)
        critic_outputs_s = self.critic(z_s, c_s)

        outputs = {
            'critic_outputs_t': critic_outputs_t,
            'critic_outputs_s': critic_outputs_s,
            'z_t': z_t,
            'z_t_hat': z_t_hat,
            'z_s': z_s,
            'z_s_restored': z_s_restored
        }
        return outputs

    def inference(self, sources, targets):
        """
        Performs keypoints transfer from target image to the source image.
        :param sources: batch of source face identities on which manipulations with keypoints transfer would be done
        :param target: batch of target images, from which keypoints would be extracted and transfered to the sources
        """
        keypoints = self.get_keypoints_from_batch(targets)
        z_s = self.encode(sources)
        z_modified = self.modify_z(z=z_s, c=keypoints).unsqueeze(1).repeat(1, 12, 1)
        outputs = self.generate(z=z_modified)
        return outputs

    def to_cuda(self):
        """
        Model contains unused modules from vanilla ALAE: for example, discriminator.
        Loads on GPU only modules necessary for facial keypoints transfer.
        """
        modules = [
            self.encoder,
            self.decoder,
            self.mapping_fl,
            self.rotation,
            self.critic
        ]
        for module in modules:
            module.cuda()
