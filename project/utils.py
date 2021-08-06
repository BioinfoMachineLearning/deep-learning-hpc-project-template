from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger


def construct_pl_logger(args):
    """Return a specific Logger instance requested by the user."""
    if args.logger_name.lower() == 'tensorboard':
        return construct_tensorboard_pl_logger(args)
    else:  # Default to using WandB
        return construct_wandb_pl_logger(args)


def construct_wandb_pl_logger(args):
    """Return an instance of WandbLogger with corresponding project and name strings."""
    return WandbLogger(name=args.experiment_name, project=args.project_name,
                       entity=args.entity, offline=args.offline, log_model=False)


def construct_tensorboard_pl_logger(args):
    """Return an instance of TensorBoardLogger with corresponding project and experiment name strings."""
    return TensorBoardLogger(save_dir=args.log_dir, name=args.experiment_name)
