import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, random_split
from torchmetrics import AUROC
from torchvision import transforms
from torchvision.datasets.mnist import MNIST


class Backbone(torch.nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, hidden_dim * 2)
        self.l3 = torch.nn.Linear(hidden_dim * 2, 10)

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

        self.train_auroc = AUROC(num_classes=10)
        self.val_auroc = AUROC(num_classes=10)
        self.test_auroc = AUROC(num_classes=10)

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_auroc', self.train_auroc(y_hat, y))
        return loss

    def training_epoch_end(self, outputs):
        self.train_auroc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_auroc', self.val_auroc(y_hat, y))
        return loss

    def validation_epoch_end(self, outputs):
        self.val_auroc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_auroc', self.test_auroc(y_hat, y))
        return loss

    def test_epoch_end(self, outputs):
        self.test_auroc.reset()

    # ---------------------
    # training setup
    # ---------------------
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingWarmRestarts(optimizer, self.hparams.num_epochs, eta_min=1e-4)
        metric_to_track = 'val_auroc'
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
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
    parser.add_argument('--multi_gpu_backend', type=str, default='ddp', help='Backend to use for multi-GPU training')
    parser.add_argument('--num_gpus', type=int, default=-1, help='Number of GPUs to use (e.g. -1 = all available GPUs)')
    parser.add_argument('--profiler_method', type=str, default='simple', help='PyTorch Lightning profiler to use')
    parser.add_argument('--num_epochs', type=int, default=20, help='Maximum number of epochs to run for training')
    parser.add_argument('--batch_size', default=4096, type=int, help='Number of samples included in each data batch')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Number of hidden units in each hidden layer')
    parser.add_argument('--num_dataloader_workers', type=int, default=6, help='Number of CPU threads for loading data')
    parser.add_argument('--experiment_name', type=str, default=None, help='Logger experiment name')
    parser.add_argument('--project_name', type=str, default='DLHPT', help='Logger project name')
    parser.add_argument('--entity', type=str, default='bml-lab', help='Logger entity (i.e. team) name')
    parser.add_argument('--offline', action='store_true', dest='offline', help='Whether to log locally or remotely')
    parser.add_argument('--online', action='store_false', dest='offline', help='Whether to log locally or remotely')
    parser.add_argument('--close_after_fit', action='store_true', dest='close_after_fit',
                        help='Whether to stop logger after calling fit')
    parser.add_argument('--open_after_fit', action='store_false', dest='close_after_fit',
                        help='Whether to stop logger after calling fit')
    parser.add_argument('--tb_log_dir', type=str, default='tb_log', help='Where to store TensorBoard log files')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='Directory in which to save checkpoints')
    parser.add_argument('--ckpt_name', type=str, default=None, help='Filename of best checkpoint')
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
    model = LitClassifier(Backbone(hidden_dim=args.hidden_dim), args.num_epochs, args.lr)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.max_epochs = args.num_epochs

    # Initialize logger
    args.experiment_name = f'LitClassifierWithBackbone-e{args.num_epochs}-b{args.batch_size}' \
        if not args.experiment_name \
        else args.experiment_name

    # Log everything to Weights and Biases (WandB)
    logger = WandbLogger(name=args.experiment_name, project=args.project_name,
                         entity=args.entity, offline=args.offline, log_model=True)

    # Assign specified logger (e.g. WandB) to Trainer instance
    trainer.logger = logger

    # ------------
    # checkpoint
    # ------------
    # Resume from checkpoint if path to a valid one is provided
    args.ckpt_name = args.ckpt_name \
        if args.ckpt_name is not None \
        else 'LitClassifier-{epoch:02d}-{val_auroc:.2f}.ckpt'
    checkpoint_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    trainer.resume_from_checkpoint = checkpoint_path if os.path.exists(checkpoint_path) else None

    # ------------
    # training
    # ------------
    # Create and use callbacks
    early_stop_callback = EarlyStopping(monitor='val_auroc', mode='min', min_delta=0.00, patience=3)
    checkpoint_callback = ModelCheckpoint(monitor='val_auroc', save_top_k=3, dirpath=args.ckpt_dir,
                                          filename='LitClassifier-{epoch:02d}-{val_auroc:.2f}')
    lr_callback = LearningRateMonitor(logging_interval='epoch')  # Use with a learning rate scheduler
    trainer.callbacks = [early_stop_callback, checkpoint_callback, lr_callback]

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
    wandb.save(checkpoint_callback.best_model_path)


if __name__ == '__main__':
    cli_main()
