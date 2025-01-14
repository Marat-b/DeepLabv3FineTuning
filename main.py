from pathlib import Path

import click
import torch
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils import data
from torchvision import transforms

import datahandler
from model import createDeepLabv3
from trainer import train_model


@click.command()
@click.option(
    "--data_directory",
    required=True,
    help="Specify the data directory."
)
@click.option(
    "--exp_directory",
    required=True,
    help="Specify the experiment directory."
)
@click.option(
    "--epochs",
    default=25,
    type=int,
    help="Specify the number of epochs you want to run the experiment for."
)
@click.option(
    "--batch-size",
    default=4,
    type=int,
    help="Specify the batch size for the dataloader."
)
@click.option(
    "--out_name",
    default="weight",
    type=str,
    help="Name of output file"
)
def main(data_directory, exp_directory, epochs, batch_size, out_name):
    # Create the deeplabv3 resnet101 model which is pretrained on a subset
    # of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    model = createDeepLabv3(outputchannels=2)
    model.train()
    data_directory = Path(data_directory)
    # Create the experiment directory if not present
    exp_directory = Path(exp_directory)
    if not exp_directory.exists():
        exp_directory.mkdir()

    # Specify the loss function
    criterion = torch.nn.MSELoss(reduction='mean')
    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Specify the evaluation metrics
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    # Create the dataloader
    dataloaders = datahandler.get_dataloader_single_folder(
        data_directory, batch_size=batch_size
    )
    _ = train_model(
        model,
        criterion,
        dataloaders,
        optimizer,
        bpath=exp_directory,
        metrics=metrics,
        num_epochs=epochs
    )

    # Save the trained model
    to_pth = '{}.pth'.format(out_name)
    to_onnx = '{}.onnx'.format(out_name)
    torch.save(model, exp_directory / to_pth)
    # img = Image.open('./images/image_256.jpg')
    # image = transforms.ToTensor()(img).unsqueeze_(0)
    # torch.onnx.export(
    #     model, (image,), exp_directory / to_onnx, opset_version=12,
    #     do_constant_folding=True
    # )


if __name__ == "__main__":
    main()
