import logging
import sys
import os
import traceback
from pathlib import Path
import re
import yaml
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.INFO)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

def get_figures(df: pd.DataFrame, data_path: Path) -> bytes:
    """Generates and saves EDA artifacts

    Args:
        df: Pandas dataframe to perform EDA on
        data_path: Designates directory to store EDA artifacts

    """

    try:
        parent_dir = Path(__file__).resolve().parent.parent
        config_path = parent_dir / 'config' / 'config.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info('Dataset loaded for EDA')
    except NameError as e:
        logger.error('EDA artifacts could not be saved: %s', e)
        traceback.print_exc()
        sys.exit(1)

    try:
        plt.rcParams['axes.labelcolor'] = config['axes']['label_color']
        plt.rcParams['axes.titlesize'] = config['axes']['title_size']
        plt.rc('axes', labelsize=config['axes']['label_size'])
        figs = []
        for feat in df.columns:
            fig, ax = plt.subplots(figsize=(config['figure']['width'], config['figure']['height']))

            ax.set_prop_cycle(color=config['axes']['prop_cycle'])

            ax.hist([
                df[df['class'] == 0][feat].values, df[df['class'] == 1][feat].values
            ], linewidth=config['lines']['line_width'])

            ax.set_xlabel(' '.join(feat.split('_')).capitalize(),
                          fontsize=config['xtick']['label_size'], color=config['text']['color'])

            ax.set_ylabel('Number of observations',
                          fontsize=config['ytick']['label_size'], color=config['text']['color'])

            figs.append(fig)

            # Save the figure as a PNG file
            if bool(re.search(r'\.[a-zA-Z]{3}$', str(data_path))):
                fig.savefig( os.path.join(os.path.splitext(data_path)[0], f'{feat}_histogram.png') )
            else:
                fig.savefig(os.path.join(data_path, f'{feat}_histogram.png'))
        logger.info('All EDA artifacts saved to %s', data_path)
    except RuntimeError as e:
        logger.error('EDA artifacts could not be saved: %s', e)
        traceback.print_exc()
        sys.exit(1)
