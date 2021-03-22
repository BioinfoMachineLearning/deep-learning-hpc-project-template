import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets.mnist import MNIST


class LitAutoEncoder(pl.LightningModule):

    def __init__(self, num_epochs: int = 5, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_mse_loss', loss)
        return loss

    # ---------------------
    # training setup
    # ---------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingWarmRestarts(optimizer, self.hparams.num_epochs, eta_min=1e-4)
        metric_to_track = 'train_mse_loss'
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': metric_to_track
        }


def cli_main():
    pl.seed_everything(42)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--multi_gpu_backend', type=str, default='ddp', help="Backend to use for multi-GPU training")
    parser.add_argument('--num_gpus', type=int, default=-1, help="Number of GPUs to use (e.g. -1 = all available GPUs)")
    parser.add_argument('--profiler_method', type=str, default='simple', help="PyTorch Lightning profiler to use")
    parser.add_argument('--num_epochs', type=int, default=5, help="Maximum number of epochs to run for training")
    parser.add_argument('--batch_size', default=4096, type=int, help='Number of samples included in each data batch')
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--hidden_dim', type=int, default=128, help='Number of hidden units in each hidden layer')
    parser.add_argument('--num_dataloader_workers', type=int, default=6, help='Number of CPU threads for loading data')
    parser.add_argument('--experiment_name', type=str, default=None, help="Logger experiment name")
    parser.add_argument('--project_name', type=str, default='DLHPT', help="Logger project name")
    parser.add_argument('--entity', type=str, default='bml-lab', help="Logger entity (i.e. team) name")
    parser.add_argument('--offline', action='store_true', dest='offline', help="Whether to log locally or remotely")
    parser.add_argument('--online', action='store_false', dest='offline', help="Whether to log locally or remotely")
    parser.add_argument('--close_after_fit', action='store_true', dest='close_after_fit',
                        help="Whether to stop logger after calling fit")
    parser.add_argument('--open_after_fit', action='store_false', dest='close_after_fit',
                        help="Whether to stop logger after calling fit")
    parser.add_argument('--tb_log_dir', type=str, default='tb_log', help="Where to store TensorBoard log files")
    parser.add_argument('--ckpt_dir', type=str, default="checkpoints", help="Directory in which to save checkpoints")
    parser.add_argument('--ckpt_name', type=str, default=None, help="Filename of best checkpoint")
    parser.set_defaults(offline=False)  # Default to using online logging mode
    parser.set_defaults(close_after_fit=False)  # Default to keeping logger open after calling fit()
    args = parser.parse_args()

    # Set HPC-specific parameter values
    args.accelerator = args.multi_gpu_backend
    args.gpus = args.num_gpus
    args.profiler = args.profiler_method

    # ------------
    # data
    # ------------
    dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=args.batch_size, num_workers=args.num_dataloader_workers)
    val_loader = DataLoader(mnist_val, batch_size=args.batch_size, num_workers=args.num_dataloader_workers)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size, num_workers=args.num_dataloader_workers)

    # ------------
    # model
    # ------------
    model = LitAutoEncoder(args.num_epochs, args.lr)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.max_epochs = args.num_epochs

    # Resume from checkpoint if path to a valid one is provided
    args.ckpt_name = args.ckpt_name \
        if args.ckpt_name is not None \
        else 'LitAutoEncoder-{epoch:02d}-{train_mse_loss:.2f}.ckpt'
    checkpoint_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    trainer.resume_from_checkpoint = checkpoint_path if os.path.exists(checkpoint_path) else None

    # Create and use callbacks
    early_stop_callback = EarlyStopping(monitor='train_mse_loss', mode='min', min_delta=0.00, patience=3)
    checkpoint_callback = ModelCheckpoint(monitor='train_mse_loss', save_top_k=3, dirpath=args.ckpt_dir,
                                          filename='LitAutoEncoder-{epoch:02d}-{train_mse_loss:.2f}')
    lr_callback = LearningRateMonitor(logging_interval='epoch')  # Use with a learning rate scheduler
    trainer.callbacks = [early_stop_callback, checkpoint_callback, lr_callback]

    # Initialize logger
    args.experiment_name = f'LitAutoEncoder-e{args.num_epochs}-b{args.batch_size}' \
        if not args.experiment_name \
        else args.experiment_name

    # Log everything to Weights and Biases (WandB)
    logger = WandbLogger(name=args.experiment_name, project=args.project_name, entity=args.entity, offline=args.offline)

    # Assign specified logger (e.g. WandB) to Trainer instance
    trainer.logger = logger

    # Train with the provided model and data module
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)

    # ------------
    # finalizing
    # ------------
    logger.experiment.log_artifact(args.ckpt_dir)
    logger.experiment.stop()


if __name__ == '__main__':
    cli_main()
