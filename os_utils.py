import os
from torch import save, load


class OSUtils(object):
    def __init__(self):
        super(OSUtils, self).__init__()

    @staticmethod
    def save_torch_object(obj, dirs, filename):
        if not os.path.exists(dirs):
            os.makedirs(dirs, exist_ok=True)
        save(obj, os.path.join(dirs, filename))

    @staticmethod
    def load_torch_object(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"The  path {path} does not exists")

        return load(path)
