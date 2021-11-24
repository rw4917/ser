from pathlib import Path
import torch
from torch import optim

from ser import model, data, trainval, parameters

import typer
import json
import pickle
import os
from datetime import datetime
from bin.utils import EnhancedJSONEncoder


main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAVE_DIR = PROJECT_ROOT / "run"


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(..., "-e", "--epochs", help="Number of epochs."),
    batch_size: int = typer.Option(..., "-b", "--batch_size", help="Batch size."),
    learning_rate: float = typer.Option(
        ..., "-l", "--learning_rate", help="Learning rate."
    ),
):
    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # save the parameters!
    params = parameters.parameters(lr=learning_rate, b=batch_size, e=epochs)

    # Create directory
    date_time_obj = datetime.now()
    timestamp_str = date_time_obj.strftime("%d-%b-%Y_(%H:%M:%S)")
    directory_name = os.path.join(SAVE_DIR, name, "{}".format(timestamp_str))

    Path(directory_name).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(directory_name, "config.json"), "w") as fjson:
        json.dump(params, fjson, cls=EnhancedJSONEncoder)

    # save the parameters!

    # load model
    main_model = model.Net().to(device)

    # setup params
    optimizer = optim.Adam(main_model.parameters(), lr=learning_rate)

    # dataloaders
    training_dataloader = data.train_loader(batch_size)
    validation_dataloader = data.validation_loader(batch_size)

    trainval.train(
        main_model,
        epochs,
        training_dataloader,
        validation_dataloader,
        device,
        optimizer,
        directory_name,
    )
    # validate


@main.command()
def infer():
    print("This is where the inference code will go")
