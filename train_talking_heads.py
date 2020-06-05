import argparse

import torch
from albumentations import HorizontalFlip, Compose, KeypointParams
from albumentations.pytorch import ToTensor

import utils
from rotation import FaceRotationModel
from data.dataset import SingleFaceDataset, BatchCollate
from losses import FaceRotationModelLoss
from trainer import ModelTrainer
from logger import Logger


def train(args):
    # Configuring
    print('Build up experiment configuration...')
    cfg = utils.build_configuration()
    cfg.merge_from_file('configs/rotation.yaml')

    # Initializing model
    print('Initializing model...')
    model = FaceRotationModel(
        landmarks_dim=cfg.MODEL.LANDMARKS_DIM,
        rotation_num_layers=cfg.MODEL.ROTATION_NUM_LAYERS,
        critic_num_layers=cfg.MODEL.CRITIC_NUM_LAYERS,
        face_alignment_device='cuda',
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
        truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER
    )
    if args.finetune:
        model.load_state_dict(torch.load(args.finetune, map_location='cpu'))
    d = torch.load(args.pretrained, map_location='cpu')
    model.load_state_dict(d, strict=False)
    model = model.cuda()
    model = model.train()

    # Initialize data loading logic
    print('Initializing data loading...')
    transform = Compose(
        [HorizontalFlip(), ToTensor()],
        keypoint_params=KeypointParams(format='xy')
    )
    transform_ci = Compose(
        [ToTensor()],
        keypoint_params=KeypointParams(format='xy')
    )
    dataset = SingleFaceDataset(
        root=args.data_dir,
        center_identity_size=cfg.TRAIN.CENTER_IDENTITY_SIZE,
        center_identity_step=cfg.TRAIN.CENTER_IDENTITY_STEP,
        transform=transform,
        transform_ci=transform_ci,
        size=cfg.TRAIN.ROTATION_BATCH_SIZE*100
    )
    batch_collate = BatchCollate()
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.TRAIN.ROTATION_BATCH_SIZE,
        shuffle=True,
        collate_fn=batch_collate,
        drop_last=False
    )

    # Training logic
    print('Initializing training logic...')
    logger = Logger(logdir=cfg.TRAIN.LOGDIR)
    criterion = FaceRotationModelLoss()
    trainer = ModelTrainer(
        config=cfg,
        optimizers={
            'generator': torch.optim.Adam(model.rotation.parameters(), betas=(0, 0.99)),
            'critic': torch.optim.Adam(model.critic.parameters(), betas=(0, 0.99))
        },
        logger=logger,
        criterion=criterion
    )
    
    # Main loop
    print('Start training...')
    iteration = 0
    for _ in range(cfg.TRAIN.N_EPOCHS):
        for batch in dataloader:
            batch = model.parse_batch(batch)
            losses, loss_stats = trainer.compute_losses(model, batch, training=True)
            trainer.run_backward(model, losses=losses)
            trainer.log_training(iteration, loss_stats)
            trainer.save_checkpoint(iteration, model)
            dataloader.dataset.step()
            iteration += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', required=True, type=str)
    parser.add_argument('-p', '--pretrained', required=False, type=str)
    parser.add_argument('-f', '--finetune', required=False, default=None, type=str)
    args = parser.parse_args()

    train(args)
