import wandb
from sys import argv

from models.train import main

PROJECT = "mvc-test"

if __name__ == "__main__":
    print(f"Running {' '.join(argv[1:])}")
    wandb.login()
    wandb.agent(
        sweep_id = "qvitkzkq",
        project = PROJECT,
        function = lambda: main(PROJECT),
        count = 250,
    )
