import logging
import sys
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np

import requests

logger = logging.getLogger(__name__)

def get_data(url: str, data_path: Path) -> bytes:
    """Acquires data from URL

    Args:
        url: URL where data to be acquired is stored
        data_path: Path where the acquired data will be stored

    """

    try:
        res = requests.get(url, timeout=15)
    except NameError as e:
        logger.error('Your URL could not be retrieved: %s', e)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        logger.error('Your URL could not be retrieved: %s', e)
        sys.exit(1)

    try:
        with open(data_path, 'w') as file:
            file.write(res.text)
        logger.info('Data written to %s', data_path)
    except NameError as e:
        logger.error('Your data could not be retrieved: %s', e)
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error('Your data could not be retrieved: %s', e)
        sys.exit(1)


def create_dataset(data_path: Path, class1_start: int, class1_end: int, class2_start: int, class2_end: int,
                   cols: list) -> pd.DataFrame:
    """Converts acquired data into dataframe

    Args:
        data_path: Path (including data file name) where the acquired data is stored
        class1_start: Starting index for first class within the dataset
        class1_end: Ending index for first class within the dataset
        class2_start: Starting index for second class within the dataset
        class2_end: Ending index for second class within the dataset
        cols: List of column names

    """

    try:
        with open(data_path, 'r') as f:
            data = [[s for s in line.split(' ') if s!=''] for line in f.readlines()]

        class1 = data[class1_start:class1_end]
        class1 = [[float(r.replace('/n', '')) for r in record] for record in class1]
        class1 = pd.DataFrame(class1, columns=cols)
        class1['class'] = np.random.choice([0,1], size=len(class1))

        class2 = data[class2_start:class2_end]
        class2 = [[float(r.replace('/n', '')) for r in record] for record in class2]
        class2 = pd.DataFrame(class2, columns=cols)
        class2['class'] = np.random.choice([0,1], size=len(class1))

        df = pd.concat([class1, class2])

        logger.info('Dataframe successfully created')

        return df
    except NameError as e:
        logger.error('Dataframe could not be found: %s', e)
        sys.exit(1)
    except KeyError as e:
        logger.error('Dataframe could not be created: %s', e)
        sys.exit(1)


def save_dataset(df: pd.DataFrame, data_path: Path) -> bytes:
    """Saves dataset to local disk

    Args:
        df: name of dataframe
        data_path: Location of saved dataframe

    """
    try:
        if bool(re.search(r'\.[a-zA-Z]{3}$', str(data_path))):
            df.to_pickle(os.path.splitext(data_path)[0]+'.pkl')
        else:
            df.to_pickle(data_path / 'dataset.pkl')
        logger.info('Dataset successfully saved to %s', data_path)
    except RuntimeError as e:
        logger.error('Dataframe could not be saved: %s', e)
        sys.exit(1)
