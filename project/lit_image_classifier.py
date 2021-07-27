import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets.mnist import MNIST


class Backbone(torch.nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.l3 = nn.Linear(hidden_dim * 2, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        x = torch.softmax(x, dim=1)
        return x


class LitClassifier(pl.LightningModule):
    def __init__(self, backbone, num_epochs: int = 5, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone

        # Declare loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Define cross-validation metrics
        self.train_acc = tm.Accuracy()
        self.val_acc = tm.Accuracy()
        self.test_acc = tm.Accuracy()

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_acc', self.train_acc(y_hat, y))
        return loss

    def training_epoch_end(self, outputs):
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_acc', self.train_acc(y_hat, y), sync_dist=True)
        return loss

    def validation_epoch_end(self, outputs):
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_acc', self.train_acc(y_hat, y), sync_dist=True)
        return loss

    def test_epoch_end(self, outputs):
        self.test_acc.reset()

    # ---------------------
    # training setup
    # ---------------------
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingLR(optimizer, self.hparams.num_epochs)
        metric_to_track = 'val_acc'
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': metric_to_track
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
        return parser


def cli_main():
    pl.seed_everything(42)

    # ------------
    # Args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
    parser.add_argument('--max_hours', type=int, default=1, help='Maximum number of hours to allot for training')
    parser.add_argument('--max_minutes', type=int, default=55, help='Maximum number of minutes to allot for training')
    parser.add_argument('--multi_gpu_backend', type=str, default='ddp', help='Backend to use for multi-GPU training')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use (e.g. -1 = all available GPUs)')
    parser.add_argument('--num_compute_nodes', type=int, default=1, help='Number of compute nodes to use')
    parser.add_argument('--gpu_precision', type=int, default=32, help='Bit size used during training (e.g. 16-bit)')
    parser.add_argument('--profiler_method', type=str, default='simple', help='PyTorch Lightning profiler to use')
    parser.add_argument('--num_epochs', type=int, default=25, help='Maximum number of epochs to run for training')
    parser.add_argument('--batch_size', default=1024, type=int, help='Number of samples included in each data batch')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Number of hidden units in each hidden layer')
    parser.add_argument('--root', type=str, default='', help='Root directory for dataset')
    parser.add_argument('--num_dataloader_workers', type=int, default=6, help='Number of CPU threads for loading data')
    parser.add_argument('--log_dir', type=str, default='tb_logs', help='Logger log directory')
    parser.add_argument('--experiment_name', type=str, default=None, help='Logger experiment name')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='Directory in which to save checkpoints')
    parser.add_argument('--ckpt_name', type=str, default=None, help='Filename of best checkpoint')
    parser.add_argument('--grad_clip_val', type=float, default=0.5, help='Norm over which to clip gradients')
    parser.add_argument('--grad_clip_algo', type=str, default='norm', help='Algorithm with which to clip gradients')
    parser.add_argument('--stc_weight_avg', action='store_true', dest='stc_weight_avg', help='Smooth loss landscape')
    parser.add_argument('--min_delta', type=float, default=0.01, help='Minimum percentage of change required to'
                                                                      ' "metric_to_track" before early stopping'
                                                                      ' after surpassing patience')
    parser.set_defaults(stc_weight_avg=True)  # Default to using stochastic weight averaging to smooth loss landscape
    args = parser.parse_args()

    # Set Lightning-specific parameter values before constructing Trainer instance
    args.max_time = {'hours': args.max_hours, 'minutes': args.max_minutes}
    args.max_epochs = args.num_epochs
    args.profiler = args.profiler_method
    args.accelerator = args.multi_gpu_backend
    args.gpus = args.num_gpus
    args.num_nodes = args.num_compute_nodes
    args.precision = args.gpu_precision
    args.gradient_clip_val = args.grad_clip_val
    args.gradient_clip_algo = args.grad_clip_algo
    args.stochastic_weight_avg = args.stc_weight_avg

    # ------------
    # Plugins
    # ------------
    args.plugins = [
        # 'ddp_sharded',  # For sharded model training (to reduce GPU requirements)
        DDPPlugin(find_unused_parameters=False)
    ]

    # ------------
    # Data
    # ------------
    dataset = MNIST(args.root, train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST(args.root, train=False, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=args.batch_size, num_workers=args.num_dataloader_workers)
    val_loader = DataLoader(mnist_val, batch_size=args.batch_size, num_workers=args.num_dataloader_workers)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size, num_workers=args.num_dataloader_workers)

    # ------------
    # Model
    # ------------
    model = LitClassifier(Backbone(hidden_dim=args.hidden_dim), args.num_epochs, args.lr)

    # ------------
    # Checkpoint
    # ------------
    ckpt_provided = args.ckpt_name is not None
    checkpoint_path = os.path.join(args.ckpt_dir, args.ckpt_name) if ckpt_provided else None
    args.resume_from_checkpoint = checkpoint_path

    # ------------
    # Trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------
    # Logger
    # ------------
    # Initialize logger
    args.experiment_name = f'LitClassifierWithBackbone-e{args.num_epochs}-b{args.batch_size}' \
        if not args.experiment_name \
        else args.experiment_name

    # Log everything to TensorBoard
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.experiment_name)

    # Assign specified logger (e.g. TensorBoard) to Trainer instance
    trainer.logger = logger

    # ------------
    # Callbacks
    # ------------
    # Create and use callbacks
    early_stop_callback = EarlyStopping(monitor='val_acc', mode='max', min_delta=args.min_delta, patience=5)
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_top_k=3, dirpath=args.ckpt_dir,
                                          filename='LitClassifier-{epoch:02d}-{val_acc:.2f}')
    lr_callback = LearningRateMonitor(logging_interval='epoch')  # Use with a learning rate scheduler
    trainer.callbacks = [early_stop_callback, checkpoint_callback, lr_callback]

    # ------------
    # Training
    # ------------
    # Train with the provided model and data module
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # Testing
    # ------------
    trainer.test(dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()
