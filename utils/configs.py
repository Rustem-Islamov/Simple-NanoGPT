import os
import yaml

from dataclasses import asdict, is_dataclass
from types import SimpleNamespace


def namespace_to_dict(ns):
    d = {}
    for k, v in vars(ns).items():
        if isinstance(v, SimpleNamespace):
            d[k] = namespace_to_dict(v)
        else:
            d[k] = v
    return d

def save_config(ns: SimpleNamespace, path: str):
    d = namespace_to_dict(ns)
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False, default_flow_style=False)


def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    def dict_to_namespace(d):
        ns = SimpleNamespace()
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(ns, k, dict_to_namespace(v))
            else:
                setattr(ns, k, v)
        return ns
    return dict_to_namespace(data)

def generate_grid_configs(reference_config:str, budget: int, lrs: list[float], batch_sizes: list[int], seq_length:int):
    output_dir = f"configs/grid_search/budget_{budget}"
    os.makedirs(output_dir, exist_ok=True)

    for batch_size in batch_sizes:
        for lr in lrs:
            num_iters = budget/(seq_length*batch_size)
            hyperparameters = load_config(reference_config)

            hyperparameters.project = f"experiment-budget{budget}"
            hyperparameters.run = f"adamw_lr{lr}_bs{batch_size}_it{int(num_iters)}"
            hyperparameters.num_iterations = int(num_iters)
            hyperparameters.lr_embed = lr
            hyperparameters.lr_matrix = lr
            hyperparameters.batch_size = batch_size
            hyperparameters.warmdown_iters = int(.28 * hyperparameters.num_iterations)

            # Set filename
            filename = f"{hyperparameters.run}.yaml"
            path = os.path.join(output_dir, filename)
            save_config(hyperparameters, path)


if __name__ == "__main__":
    reference_config = "./configs/adamw.yaml"
    seq_length = 1024
    batch_sizes = [512, 1024, 2048, 4096]
    lrs = [1.2e-4, 2.4e-4, 3.6e-4, 4.8e-4, 6.0e-4, 7.2e-4]
    for budget in [2673868800, 5347737600, 8021606400]:
        generate_grid_configs(reference_config, budget, lrs, batch_sizes, seq_length)
