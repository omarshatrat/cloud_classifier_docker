from pathlib import Path
import sys
import pandas as pd
import pytest

parent_dir = Path(__file__).resolve().parent.parent
src_dir = parent_dir / 'src'
sys.path.append(str(src_dir))

# pylint: disable=wrong-import-position
import generate_features

# Happy testing

happy_data = pd.DataFrame([
    [1.1, 2.2, 3.3, 4.4, 5.5],
    [1.4, 2.3, 3.7, 4.9, 5.2],
    [1.8, 2.6, 3.3, 8.9, 9.9]
], columns = ['visible_entropy', 'visible_contrast', 'IR_min', 'IR_max', 'IR_mean'])


def test_happy_test_feature1():
    '''
    Unit Testing

    Purpose of function: take a dataframe with column names from config and generates the following features:
        ['log_entropy', 'entropy_x_contrast', 'IR_range', 'IR_norm_range']

    Each column must comprise of floats or ints.
    '''
    test_data = happy_data
    generate_features.generate_features(test_data)
    assert 'log_entropy' in test_data.columns

def test_happy_test_feature2():
    '''
    Unit Testing

    Purpose of function: take a dataframe which must have the following column names: 
        ['visible_entropy', 'visible_contrast', 'IR_min', 'IR_max', 'IR_mean'] and generates the following features:
        ['log_entropy', 'entropy_x_contrast', 'IR_range', 'IR_norm_range']

    Each column must comprise of floats or ints.
    '''
    test_data = happy_data
    generate_features.generate_features(test_data)
    assert 'entropy_x_contrast' in test_data.columns

def test_happy_test_feature3():
    '''
    Unit Testing

    Purpose of function: take a dataframe which must have the following column names: 
        ['visible_entropy', 'visible_contrast', 'IR_min', 'IR_max', 'IR_mean'] and generates the following features:
        ['log_entropy', 'entropy_x_contrast', 'IR_range', 'IR_norm_range']

    Each column must comprise of floats or ints.
    '''
    test_data = happy_data
    generate_features.generate_features(test_data)
    assert 'IR_range' in test_data.columns

def test_happy_test_feature4():
    '''
    Unit Testing

    Purpose of function: take a dataframe which must have the following column names: 
        ['visible_entropy', 'visible_contrast', 'IR_min', 'IR_max', 'IR_mean'] and generates the following features:
        ['log_entropy', 'entropy_x_contrast', 'IR_range', 'IR_norm_range']

    Each column must comprise of floats or ints.
    '''
    test_data = happy_data
    generate_features.generate_features(test_data)
    assert 'IR_norm_range' in test_data.columns

# Unhappy testing

unhappy_data = pd.DataFrame([
    [1.1, '2.2', 3.3, 4.4, 5.5],
    [1.4, 2.3, False, 4.9, 5.2],
    [1.8, 2.6, 3.3, None, 9.9]
], columns = ['visible_entropy', 'visible_contrast', 'IR_min', 'IR_max', 'IR_mean'])


def test_unhappy_test_feature1():
    '''
    Unit Testing

    Purpose of function: take a dataframe which must have the following column names: 
        ['visible_entropy', 'visible_contrast', 'IR_min', 'IR_max', 'IR_mean'] and generates the following features:
        ['log_entropy', 'entropy_x_contrast', 'IR_range', 'IR_norm_range']

    Each column must comprise of floats or ints.
    '''
    test_data = unhappy_data
    with pytest.raises(TypeError):
        generate_features.generate_features(test_data)

def test_unhappy_test_feature2():
    '''
    Unit Testing

    Purpose of function: take a dataframe which must have the following column names: 
        ['visible_entropy', 'visible_contrast', 'IR_min', 'IR_max', 'IR_mean'] and generates the following features:
        ['log_entropy', 'entropy_x_contrast', 'IR_range', 'IR_norm_range']

    Each column must comprise of floats or ints.
    '''
    test_data = unhappy_data
    with pytest.raises(TypeError):
        generate_features.generate_features(test_data)

def test_unhappy_test_feature3():
    '''
    Unit Testing

    Purpose of function: take a dataframe which must have the following column names: 
        ['visible_entropy', 'visible_contrast', 'IR_min', 'IR_max', 'IR_mean'] and generates the following features:
        ['log_entropy', 'entropy_x_contrast', 'IR_range', 'IR_norm_range']

    Each column must comprise of floats or ints.
    '''
    test_data = unhappy_data
    with pytest.raises(TypeError):
        generate_features.generate_features(test_data)

def test_unhappy_test_feature4():
    '''
    Unit Testing

    Purpose of function: take a dataframe which must have the following column names: 
        ['visible_entropy', 'visible_contrast', 'IR_min', 'IR_max', 'IR_mean'] and generates the following features:
        ['log_entropy', 'entropy_x_contrast', 'IR_range', 'IR_norm_range']

    Each column must comprise of floats or ints.
    '''
    test_data = unhappy_data
    with pytest.raises(TypeError):
        generate_features.generate_features(test_data)

def test_generate_features_missing_columns():
    df = pd.DataFrame({
        'visible_entropy': [1, 2, 3],
        # Missing 'visible_contrast', 'IR_max', 'IR_min', 'IR_mean' columns
    })
    with pytest.raises(KeyError):
        generate_features.generate_features(df)

def test_generate_features_incorrect_dtype():
    df = pd.DataFrame({
        'visible_entropy': ['a', 'b', 'c'],  # Incorrect data type
        'visible_contrast': [4, 5, 6],
        'IR_max': [7, 8, 9],
        'IR_min': [10, 11, 12],
        'IR_mean': [13, 14, 15]
    })
    with pytest.raises(TypeError):
        generate_features.generate_features(df)

def test_generate_features_empty_df():
    df = pd.DataFrame()
    with pytest.raises(KeyError):
        generate_features.generate_features(df)


#if __name__ == "__main__":
#    pytest.main()
