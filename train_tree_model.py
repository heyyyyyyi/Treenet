import torch
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from tree_resnet.treenet import TreeModel_34
from tree_resnet.treetrain import TreeClassifier
import ml_collections

# Define CIFAR-10 data loaders
def get_cifar10_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

# Main training script
def main():
    # Initialize wandb logger
    wandb_logger = WandbLogger(project="TreeModel-CIFAR10", log_model=True)

    # Define model and training configuration
    model = TreeModel_34(num_channels=3, num_classes=10)
    optimizer_cfg = ml_collections.ConfigDict({
        "root_lr": 1e-3,
        "subroot_animal_lr": 1e-3,
        "subroot_vehicle_lr": 1e-3,
    })
    lr_schedule_cfg = ml_collections.ConfigDict({
        "max_lr": 1e-2,
        "total_steps": 1000,
    })

    tree_classifier = TreeClassifier(
        model=model,
        optimizer_cfg=optimizer_cfg,
        lr_schedule_cfg=lr_schedule_cfg,
        scheduler_t="cyclic",
        max_epochs=20,
    )

    # Load CIFAR-10 data
    trainloader, testloader = get_cifar10_dataloaders(batch_size=128)

    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val.acc",
        mode="max",
        save_top_k=1,
        dirpath="./checkpoints",
        filename="tree_model-{epoch:02d}-{val.acc:.2f}"
    )

    # Train the model
    trainer = pl.Trainer(
        max_epochs=20,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1
    )
    trainer.fit(tree_classifier, trainloader, testloader)

if __name__ == "__main__":
    main()
