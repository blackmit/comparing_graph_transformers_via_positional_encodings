import sys
import wandb
import yaml

if len(sys.argv) != 2:
    raise ValueError("Exactly one argument must be provided: filename of config.")

filename = sys.argv[1]
with open(filename, "r") as f:
    sweep_configuration = yaml.safe_load(f)

sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project=sweep_configuration["project"],
    entity=sweep_configuration["entity"]
)

with open("{}_id.txt".format(sweep_configuration['name']), "w") as f: 
    f.write(sweep_id)