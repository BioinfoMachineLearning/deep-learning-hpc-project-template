import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets.mnist import MNIST


class LitClassifier(pl.LightningModule):
    def __init__(self, hidden_dim=128, num_epochs: int = 5, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim * 2)
        self.l3 = nn.Linear(self.hparams.hidden_dim * 2, 10)

        # Declare loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Define cross-validation metrics
        self.train_acc = tm.Accuracy(average='weighted', num_classes=10)
        self.val_acc = tm.Accuracy(average='weighted', num_classes=10)
        self.test_acc = tm.Accuracy(average='weighted', num_classes=10)

        self.train_auroc = tm.AUROC(average='weighted')
        self.val_auroc = tm.AUROC(average='weighted')
        self.test_auroc = tm.AUROC(average='weighted')

        self.train_auprc = tm.AveragePrecision()
        self.val_auprc = tm.AveragePrecision()
        self.test_auprc = tm.AveragePrecision()

        self.train_f1 = tm.F1(average='weighted', num_classes=10)
        self.val_f1 = tm.F1(average='weighted', num_classes=10)
        self.test_f1 = tm.F1(average='weighted', num_classes=10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        x = torch.softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_acc', self.train_acc(y_hat, y), sync_dist=True)
        self.log('train_auroc', self.train_auroc(y_hat, y), sync_dist=True)
        self.log('train_auprc', self.train_auprc(y_hat, y), sync_dist=True)
        self.log('train_f1', self.train_f1(y_hat, y), sync_dist=True)
        return loss

    def training_epoch_end(self, outputs):
        self.train_acc.reset()
        self.train_auroc.reset()
        self.train_auprc.reset()
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_acc', self.train_acc(y_hat, y), sync_dist=True)
        self.log('val_auroc', self.train_auroc(y_hat, y), sync_dist=True)
        self.log('val_auprc', self.train_auprc(y_hat, y), sync_dist=True)
        self.log('val_f1', self.train_f1(y_hat, y), sync_dist=True)
        return loss

    def validation_epoch_end(self, outputs):
        self.val_acc.reset()
        self.val_auroc.reset()
        self.val_auprc.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_acc', self.train_acc(y_hat, y), sync_dist=True)
        self.log('test_auroc', self.train_auroc(y_hat, y), sync_dist=True)
        self.log('test_auprc', self.train_auprc(y_hat, y), sync_dist=True)
        self.log('test_f1', self.train_f1(y_hat, y), sync_dist=True)
        return loss

    def test_epoch_end(self, outputs):
        self.test_acc.reset()
        self.test_auroc.reset()
        self.test_auprc.reset()
        self.test_f1.reset()

    # ---------------------
    # training setup
    # ---------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingLR(optimizer, self.hparams.num_epochs)
        metric_to_track = 'val_auroc'
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': metric_to_track
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=128)
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
    parser.add_argument('--num_compute_nodes', type=int, default=2, help='Number of compute nodes to use')
    parser.add_argument('--profiler_method', type=str, default='simple', help='PyTorch Lightning profiler to use')
    parser.add_argument('--num_epochs', type=int, default=25, help='Maximum number of epochs to run for training')
    parser.add_argument('--batch_size', default=16384, type=int, help='Number of samples included in each data batch')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Number of hidden units in each hidden layer')
    parser.add_argument('--root', type=str, default='', help='Root directory for dataset')
    parser.add_argument('--num_dataloader_workers', type=int, default=6, help='Number of CPU threads for loading data')
    parser.add_argument('--log_dir', type=str, default='tb_logs', help='Logger log directory')
    parser.add_argument('--experiment_name', type=str, default=None, help='Logger experiment name')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='Directory in which to save checkpoints')
    parser.add_argument('--ckpt_name', type=str, default=None, help='Filename of best checkpoint')
    args = parser.parse_args()

    # Set HPC-specific parameter values
    args.accelerator = args.multi_gpu_backend
    args.gpus = args.num_gpus
    args.num_nodes = args.num_compute_nodes
    args.profiler = args.profiler_method

    # ------------
    # data
    # ------------
    dataset = MNIST(args.root, train=True, download=False, transform=transforms.ToTensor())
    mnist_test = MNIST(args.root, train=False, download=False, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=args.batch_size, num_workers=args.num_dataloader_workers)
    val_loader = DataLoader(mnist_val, batch_size=args.batch_size, num_workers=args.num_dataloader_workers)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size, num_workers=args.num_dataloader_workers)

    # ------------
    # model
    # ------------
    model = LitClassifier(args.hidden_dim, args.num_epochs, args.lr)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.max_epochs = args.num_epochs

    # Initialize logger
    args.experiment_name = f'LitClassifier-e{args.num_epochs}-b{args.batch_size}' \
        if not args.experiment_name \
        else args.experiment_name

    # Log everything to TensorBoard
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.experiment_name)

    # Assign specified logger (e.g. TensorBoard) to Trainer instance
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
    early_stop_callback = EarlyStopping(monitor='val_auroc', mode='min', min_delta=0.01, patience=5)
    checkpoint_callback = ModelCheckpoint(monitor='val_auroc', save_top_k=3, dirpath=args.ckpt_dir,
                                          filename='LitClassifier-{epoch:02d}-{val_auroc:.2f}')
    lr_callback = LearningRateMonitor(logging_interval='epoch')  # Use with a learning rate scheduler
    trainer.callbacks = [early_stop_callback, checkpoint_callback, lr_callback]

    # Train with the provided model and data module
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()
