import os
from argparse import ArgumentParser
import torch
from comet_ml import API
from pytorch_lightning.loggers import CometLogger
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split

from torchvision.datasets.mnist import MNIST
from torchvision import transforms


class LitAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
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
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--early_stop_callback', type=bool, default=True)
    parser.add_argument('--num_dataloader_workers', type=int, default=1)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

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
    model = LitAutoEncoder()

    # ------------
    # checkpoint
    # ------------
    try:
        api = API(api_key=os.environ.get('COMET_API_KEY'))
        experiment = api.get(f'workspace-name/project-name/EarlyStopping-Adam-{args.batch_size}-{args.learning_rate}',
                             output_path="./", expand=True)

        # Download an Experiment Model:
        experiment.download_model(f'EarlyStopping-Adam-{args.batch_size}-{args.learning_rate}-Model',
                                  output_path="./", expand=True)

        model = LitAutoEncoder.load_from_checkpoint(
            f'EarlyStopping-Adam-{args.batch_size}-{args.learning_rate}-Model.pth')
        print('Resuming from checkpoint...')
    except FileNotFoundError:
        print('Could not restore checkpoint. Skipping...')

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # The arguments made to CometLogger are passed on to the comet_ml.Experiment class
    comet_logger = CometLogger(
        api_key=os.environ.get('COMET_API_KEY'),
        save_dir='.',  # Optional
        project_name='dlhpt',  # Optional
        experiment_name=f'EarlyStopping-Adam-{args.batch_size}-{args.learning_rate}',  # Optional
        log_hyperparams=True
    )
    trainer.logger = comet_logger

    trainer.early_stop_callback = args.early_stop_callback
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)

    # ------------
    # finalizing
    # ------------
    torch.save(model.state_dict(), f'EarlyStopping-Adam-{args.batch_size}-{args.learning_rate}-Model.pth')
    comet_logger.experiment.log_model(f'EarlyStopping-Adam-{args.batch_size}-{args.learning_rate}-Model',
                                      f'EarlyStopping-Adam-{args.batch_size}-{args.learning_rate}-Model.pth')
    comet_logger.finalize()


if __name__ == '__main__':
    cli_main()
