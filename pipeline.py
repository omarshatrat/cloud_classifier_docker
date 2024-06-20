from pathlib import Path
import logging
import logging.config
from joblib import dump
import yaml

parent_dir = Path(__file__).parent

logging_config = parent_dir / 'logging.conf'
logging.config.fileConfig(logging_config, disable_existing_loggers=True)

src_dir = parent_dir / 'src'

# pylint: disable=wrong-import-position
from src import acquire_data
from src import aws_utils
from src import eda
from src import generate_features
from src import train_model


logger = logging.getLogger('clouds') # what to do with this? do i remove it?

def main():
    '''Runs cloud classification pipeline and stores all artifacts.

    '''

    # Define config file

    config_path = parent_dir / 'config' / 'config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Save config file

    artifacts_path = parent_dir / 'artifacts'
    dump(config, (artifacts_path) / 'config.yaml')

    # Obtain data from URL
    data_path = parent_dir / 'data' / 'data.txt'
    acquire_data.get_data(config['run_config']['data_source'], data_path)

    # Create and save dataset

    df = acquire_data.create_dataset(data_path, 53, 1076, 1082, 2105, config['run_config']['column_names'])
    acquire_data.save_dataset(df, data_path)
    acquire_data.save_dataset(df, (artifacts_path))

    # Load df

    df = generate_features.load_df(data_path)

    # Create features

    generate_features.generate_features(df)

    # EDA

    eda.get_figures(df, (artifacts_path))

    # Split and save training and test data

    x_train, x_test, y_train, y_test = train_model.save_data((artifacts_path),
                                                             df[['log_entropy', 'entropy_x_contrast',
                                                                 'IR_range', 'IR_norm_range', 'class']], 'class')

    # Training model

    rf_model = train_model.train_model(x_train, y_train)

    # Save model

    train_model.save_model((artifacts_path), rf_model)

    # Score model and save metrics

    train_model.score_model((artifacts_path), rf_model, x_test, y_test)

    # Upload all artifacts to S3

    if config['aws']['upload'] is True:
        uploaded_files = aws_utils.upload_artifacts(str(artifacts_path), config)
        print('Files uploaded to S3:', uploaded_files)
        print()


if __name__ == '__main__':
    main()
