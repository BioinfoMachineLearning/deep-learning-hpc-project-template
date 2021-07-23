from pytorch_lightning import Trainer, seed_everything
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from project.lit_mnist import LitClassifier


def test_lit_classifier():
    seed_everything(1234)

    dataset = MNIST('', train=True, download=False, transform=transforms.ToTensor())
    mnist_test = MNIST('', train=False, download=False, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=32, num_workers=1)
    val_loader = DataLoader(mnist_val, batch_size=32, num_workers=1)
    test_loader = DataLoader(mnist_test, batch_size=32, num_workers=1)

    model = LitClassifier()
    trainer = Trainer(limit_train_batches=50, limit_val_batches=20, max_epochs=2)
    trainer.fit(model, train_loader, val_loader)

    results = trainer.test(test_dataloaders=test_loader)
    assert results[0]['test_acc'] > 0.7
