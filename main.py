import glob
import argparse


import torch
import pytorch_lightning as pl
from pytorch_lightning import pl_callbacks, loggers as pl_loggers
import comet_ml


import model
from model import LitAutoEncoder
from load_dataset import get_train_val_test_dataloaders


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--logger', type=str, help="type of logger to use, options: 'tb' (tensorboard), 'comet' (comet.ml)", default='tb')


def main():
    args = parser.parse_args()

    train_ld, val_ld, test_ld = get_train_val_test_dataloaders()
    encdecoder = LitAutoEncoder()

    acc_early_stop = pl_callbacks.EarlyStopping(monitor="val_accuracy", patience=3, verbose=False, mode="max")
    # for custom stopping behaviour (for example, in training), EarlyStopping can be overriden

    if args.logger == 'tb':
        logger = pl_loggers.TensorBoardLogger()
    elif args.logger == 'commet':
        logger = pl_loggers.CometLogger(api_key="")

    tensorboard = pl_loggers.TensorBoardLogger()
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator='gpu',
        devices=1, 
        default_root_dir='./lightning_logs', # directory where all the logging goes
        callbacks=[acc_early_stop], # callbacks that execute on some condtions
        logger=logger
    )

    trainer.fit(model=encdecoder, train_dataloaders=train_ld, val_dataloaders=val_ld)
    trainer.test(model=encdecoder, dataloaders=test_ld)

    # model load from checkpoint
    checkpoint_path = glob.glob("./lightning_logs/version_{version}/checkpoints/*.ckpt")[0] # scenario where there is only one checkpoint
   
    # loading saved model, also loads up hyperparameters saved by self.save_hyperparameters
    # hyperparameters can be altered by passing matching kwargs
    # checkpoints can be loaded by pytorch natively, they will be treated as dict of weight tensor
    # example: checkpoint = torch.load(CHKP_PATH) encoder weights = checkpoint["encoder"]
    checkpoint = torch.load(checkpoint_path)
    print(list(checkpoint.keys()))

    encdecoder = LitAutoEncoder.load_from_checkpoint(checkpoint_path, learning_rate=3e-4) 
    encdecoder.eval()

    # model ready for inference


if __name__ == '__main__':
    main()