


#THIS CODE OR THE BIGGER PART OF IT CAME FROM THE REPOSITORY  https://github.com/HLasse/data-centric-synthetic-data
import pickle as pkl
from pathlib import Path
from typing import Any
import os

def save_to_pickle(obj: Any, path: str):
    """Save the given object to a pickle file.
    Args:
        obj (Any): The object to save.
        path (str): The path to save the object to.
    """

    # Extrai o diretório pai do caminho
    parent_directory = os.path.dirname(path)

    # Cria o diretório pai se não existir
    os.makedirs(parent_directory, exist_ok=True)

    # Salva o objeto no arquivo pickle
    with open(path, "wb") as f:
        pkl.dump(obj, f)


def load_from_pickle(path: Path) -> Any:
    """Load an object from a pickle file.
    Args:
        path (Path): The path to load the object from.
    Returns:
        Any: The loaded object.
    """
    with path.open("rb") as f:
        return pkl.load(f)
    
