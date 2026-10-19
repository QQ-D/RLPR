import os


def env_path(name, default):
    value = os.environ.get(name)
    if value:
        return os.path.expanduser(value)
    return default


def data_root(dataset_name, default_subdir):
    env_name = f"RLPR_{dataset_name.upper()}_DATA"
    default = os.path.join(os.path.expanduser("~"), "RLPR_data", default_subdir)
    return env_path(env_name, default)


def raw_root(dataset_name, default):
    env_name = f"RLPR_{dataset_name.upper()}_RAW"
    return env_path(env_name, default)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
