from os.path import exists

from imp_arg_instance import ImplicitArgumentInstance, ImplicitArgumentNode
from imp_arg_loader import download_dataset, read_dataset
from imp_arg_loader import imp_arg_dataset_path

if not exists(imp_arg_dataset_path):
    download_dataset()

imp_arg_instances = read_dataset(imp_arg_dataset_path)
