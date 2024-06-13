# W&B Hyperparameter Search

W&B hyperparameter sweeps search over a space of hyperparameters. These parameters can be written in a yaml file that can be used to initialize a wandb hyperparamter sweep. More info on how to write a sweep config file can be found on the [W&B website](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration). 

To initialize a sweep from a configuration file, use the command

```bash
python init_sweep.py {configuration_file}.yaml
```

This will create a file named `{sweep_name}_id.txt` containing the sweep id. To execute this sweep, add the following lines to your graphgps yaml config file:
```yaml
wandb:
    use: True
    sweep:
        enable: True
        id: {sweep_id} # The sweep id is found in {sweep_name}_id.txt
```