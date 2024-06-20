import logging
import sys
from pathlib import Path
import pandas as pd
import sklearn
import sklearn.ensemble
from joblib import dump

logger = logging.getLogger(__name__)


def save_data(data_path: Path, df: pd.DataFrame, outcome_name: str, test_size: float = 0.4) -> pd.DataFrame:
    """Splits and saves training data and test data for reproducability

    Args:
        data_path: Path where data will be stored
        df: Pandas dataframe to perform EDA on
        initial_features: Features to include for model training
        outcome_name: Name of dependent variable
        test_size: Percentage of data held back for testing (default = 0.4)

    """
    try:
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(df, df[outcome_name],
                                                                                    test_size=test_size)
        dump((x_train, x_test, y_train, y_test), data_path / 'train_test_data.joblib')
        logger.info('Data split and saved to %s', data_path)
        return x_train, x_test, y_train, y_test
    except RuntimeError as e:
        logger.error('Data could not be split: %s', e)
        sys.exit(1)
    except NameError as e:
        logger.error('Data could not be split: %s', e)
        sys.exit(1)


def train_model(x_train: pd.DataFrame, y_train: pd.DataFrame, n_estimators: int = 10,
                max_depth: int = 10) -> sklearn.ensemble.RandomForestClassifier:
    """Performs train/test split of data and trains model

    Args:
        x_train: Pandas dataframe of independent training data
        y_train: Pandas dataframe of dependent training data
        n_estimators: Number of tree estimators (default = 10)
        max_depth: Max depth of each tree (default = 10)

    """

    try:
        rf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        rf.fit(x_train, y_train)
        logger.info('Random Forest Classifier successfully trained')
        return rf
    except TypeError as e:
        logger.error('Model could not be trained: %s', e)
        sys.exit(1)
    except NameError as e:
        logger.error('Model could not be trained: %s', e)
        sys.exit(1)




def save_model(data_path: Path, trained_model: sklearn.ensemble._forest.RandomForestClassifier):
    """Saves training data and test data for reproducability

    Args:
        data_path: Path where model will be stored
        trained_model: Trained model object (e.g. 'rf')

    """

    try:
        dump(trained_model, data_path / 'rf_classifer.joblib')
        logger.info('Trained model has been saved to %s', data_path)
    except NameError as e:
        logger.error('Model could not be trained: %s', e)
        sys.exit(1)


def score_model(data_path: Path, model: sklearn.ensemble._forest.RandomForestClassifier, x_test: pd.DataFrame,
                y_test: pd.DataFrame):
    '''Scores model and saves performance metrics
    
    Args:
        data_path: Path where model artifacts will be stored
        model: Trained model object
        x_test: Pandas dataframe of independent testing data
        y_test: Pandas dataframe of dependent testing data

    '''

    # Score model
    try:
        ypred_proba_test = model.predict_proba(x_test)[:,1]
        ypred_bin_test = model.predict(x_test)
    except NameError as e:
        logger.error('Model could not be trained: %s', e)
        sys.exit(1)
    except TypeError as e:
        logger.error('Model could not be trained: %s', e)
        sys.exit(1)

    # Evaluate performance
    try:
        auc = sklearn.metrics.roc_auc_score(y_test, ypred_proba_test)
    except ValueError:
        logger.warning('Error: AUC could not be computed because your y variable only contains one class.')
    except TypeError:
        logger.warning('Error: AUC could not be computed because your y variable only contains one class.')
    confusion = sklearn.metrics.confusion_matrix(y_test, ypred_bin_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, ypred_bin_test)
    classification_report = sklearn.metrics.classification_report(y_test, ypred_bin_test)

    try:
        print(f'AUC on test: {auc:.3f}')
    except ValueError as e:
        logger.warning('Warning, AUC could not be printed: %s', e)
    print(f'Accuracy on test: {accuracy:.3f}')
    print()
    try:
        print(pd.DataFrame(confusion,
                    index=['Actual negative','Actual positive'],
                    columns=['Predicted negative', 'Predicted positive']))
    except TypeError as e:
        logger.warning('Because AUC could not be calculated, confusion matrix may be incomplete: %s', e)
        print(confusion)
    print()

    try:
        metrics = {'auc': auc,
               'confusion': confusion,
               'accuracy': accuracy,
               'classification_report': classification_report}
    except TypeError as e:
        metrics = {'confusion': confusion,
                   'accuracy': accuracy,
                   'classification_report': classification_report}
        logger.info('AUC could not be calculated and is thus not included in metrix export: %s', e)

    dump(metrics, data_path / 'model_metrics.joblib')
    logger.info('Metrics successfully saved!')
