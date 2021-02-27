from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets.mnist import MNIST


class Backbone(torch.nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x


class LitClassifier(pl.LightningModule):
    def __init__(self, backbone, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_dataloader_workers', type=int, default=2)
    parser.add_argument('--name', type=str, default='DLHPT WandB Test on MNIST', help="Run name")
    parser.add_argument('--wandb', type=str, default='DLHPT', help="WandB project name")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # Define HPC-specific properties in-file
    args.accelerator = 'ddp'
    args.gpus = 6
    args.max_epochs = 5

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
    # checkpoint
    # ------------
    try:
        model = LitClassifier.load_from_checkpoint(f'Adam-{args.batch_size}-{args.learning_rate}.pth')
        print('Resuming from checkpoint...')
    except:
        # ------------
        # model
        # ------------
        print('Could not restore checkpoint. Skipping...')
        model = LitClassifier(Backbone(hidden_dim=args.hidden_dim), args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # Logging all args to wandb
    logger = WandbLogger(name=args.name, project=args.wandb) if args.name else WandbLogger(project=f'{args.wandb}')
    trainer.logger = logger

    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)

    # ------------
    # finalizing
    # ------------
    trainer.save_checkpoint(f'Adam-{args.batch_size}-{args.learning_rate}.pth')


if __name__ == '__main__':
    cli_main()
