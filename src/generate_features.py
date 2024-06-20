import logging
import sys
import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def load_df(data_path: Path) -> pd.DataFrame:
    """Loads dataframe

    Args:
        data_path: Path where the df lives

    """

    try:
        df = pd.read_pickle(os.path.splitext(data_path)[0]+'.pkl')
        logger.info('Dataframe successfully loaded')
        return df
    except NameError as e:
        logger.error('Dataframe could not be loaded: %s', e)
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error('Dataframe could not be loaded: %s', e)
        sys.exit(1)



def generate_features(df: pd.DataFrame) -> bytes:
    """Adds generated features to loaded model

    Args:
        df: Name of dataframe

    """

    try:
        parent_dir = Path(__file__).resolve().parent.parent
        config_path = parent_dir / 'config' / 'config.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        logger.warning('Command executing, please ensure that the dataframe passed contains required columns')

        f1 = config['generate_features']['log_transform']['log_entropy']
        f2 = config['generate_features']['multiply']['col_a']
        f3 = config['generate_features']['calculate_norm_range']['max_col']
        f4 = config['generate_features']['calculate_norm_range']['min_col']
        f5 = config['generate_features']['calculate_norm_range']['mean_col']

        # Check if required columns are present in DataFrame
        required_columns = [f1, f2, f3, f4, f5]
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' is missing in the DataFrame")


        df['log_entropy'] = df[f1].apply(np.log)
        df['entropy_x_contrast'] = df[f2].multiply(df[f1])
        df['IR_range']  = df[f3] - df[f4]
        df['IR_norm_range'] = (df[f3] - df[f4]).divide(df[f5])

        logger.info('Features successfully generated')

    except ArithmeticError as e:
        logger.error('Arithmetic error, Features could not be added: %s', e)
        sys.exit(1)
    except KeyError as e:
        logger.error('Key error, Features could not be added: %s', e)
        raise
        #sys.exit(1)
    except RuntimeError as e:
        logger.error('Runtime error, Features could not be added: %s', e)
        sys.exit(1)
