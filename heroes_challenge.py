import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import svm,ensemble, tree, neighbors, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# CORE OF THE PROGRAM

def prepare_data(data, test_size):
    y = data["Alignment"]
    X = data.drop(columns=['Alignment'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train_scaled = StandardScaler().fit_transform(X_train)
    X_test_scaled = StandardScaler().fit_transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

def compute_accuracy(method, method_kwargs, X_train, X_test, y_train, y_test):
    method = method(**method_kwargs)
    method.fit(X_train, y_train) 
    yhat = method.predict(X_test)
    return metrics.accuracy_score(y_test,yhat)

def run(**kwargs):
    data = pd.read_csv('train.csv')[chosen_columns]
    test_data = pd.read_csv('test.csv')[chosen_columns[1:]] #Dropping 'Alignment' column.
    cleaned_data = clean_data(data, **kwargs)
    cleaned_test_data = clean_data(test_data, **kwargs)
    X_train, X_test, y_train, y_test = prepare_data(cleaned_data, kwargs['test_size'])
    max_acc = 0.0
    for name, method in kwargs['methods'].items():
        method_kwargs = kwargs['methods_kwargs'][name]
        if 'random_state' in method_kwargs:
            random_iterator = np.arange(0, kwargs['n_iters'], 1)
        else:
            random_iterator = [None]
        N = len(random_iterator)
        acc = 0.0
        for random_seed in random_iterator:
            methods_kwargs.update({'random_state': random_seed})
            acc += compute_accuracy(method, method_kwargs, X_train, X_test, y_train, y_test) / N
        print('The accuracy for {} is {}.'.format(name, acc))
        if acc > max_acc:
            max_acc = acc
            best_model = name
    print('')
    print('RESULT:')
    print('The {} is the model with best accuracy, acc = {}.'.format(best_model, max_acc))

# METHODS

methods = {
    'SVC with C = 10': svm.SVC,
    'SVC with C = 100': svm.SVC,
    'SVC with C = 1000': svm.SVC,
    'GradientBoosting': ensemble.GradientBoostingClassifier,
    'DecisionTree': tree.DecisionTreeClassifier,
    'KNeighbors with n = 2': neighbors.KNeighborsClassifier,
    'NNeighbors with n = 7': neighbors.KNeighborsClassifier,
    'Gaussian Naive-Bayes': GaussianNB,
    'LinearSVC': LinearSVC,
    'RandomForest': RandomForestClassifier,
    'LogisticRegression': LogisticRegression,
}

methods_kwargs = {
    'SVC with C = 10': {'C': 10.0, 'gamma': 1e-5, 'random_state': None},
    'SVC with C = 100': {'C': 100.0, 'gamma': 1e-5, 'random_state': None},
    'SVC with C = 1000': {'C': 1000.0, 'gamma': 1e-5, 'random_state': None},
    'GradientBoosting': {'random_state': None},
    'DecisionTree': {},
    'KNeighbors with n = 2': {'n_neighbors': 2},
    'NNeighbors with n = 7': {'n_neighbors': 7},
    'Gaussian Naive-Bayes': {},
    'LinearSVC': {'random_state': None},
    'RandomForest': {'n_estimators': 100, 'random_state': None},
    'LogisticRegression': {},
}

# HUMAN MANIPULATIONS - TECHNICAL INFO

chosen_columns = ["Alignment", "Gender", "Eye color", "Race", "Hair color",
                  "Publisher", "Skin color", "Height", "Weight", "Agility",
                  "Accelerated Healing", "Lantern Power Ring",
                  "Dimensional Awareness", "Cold Resistance", "Durability",
                  "Stealth", "Energy Absorption", "Flight", "Danger Sense"]

eye_color_dic = {
    'bown': "brown", 
    'gold': 'yellow',
    'hazel': 'green',
    'violet': 'purple',
    'yellow (without irises)': 'yellow',
}

hair_color_dic = {
    'black': "Black",
    'Auburn': 'Red',
    'Magenta': 'Purple',
    'Pink': 'Purple',
    'Silver': 'Grey',
    'Strawberry Blond': 'Blond',
    'blond': 'Blond'
}

skin_dic = {
    'pink': 'purple',
    'silver': 'grey'
}

race_dic = {
    'Alpha': 'No idea',
    'Amazon': 'Human',
    'Asgardian': 'Human',
    'Atlantean': 'Human',
    'Bizarro': 'No idea',
    'Clone': 'Android',
    'Cosmic Entity': 'God',
    'Cyborg': 'Android',
    'Dathomirian Zabrak': 'No idea',
    'Demi-God': 'God',
    'Demon': 'Bad',
    'Eternal': 'God',
    'Flora Colossus': 'God',
    'God / Eternal': 'God',
    'Gorilla': 'Animal',
    'Gungan': 'No idea',
    'Human / Altered': 'Human',
    'Human / Clone': 'Human',
    'Human / Cosmic': 'Human',
    'Human / Radiation': 'Human',
    'Human-Kree': 'Human-Kree',
    'Human-Vuldarian': 'Human',
    'Icthyo Sapien': 'No idea',
    'Inhuman' : 'Bad',
    'Korugaran': 'No idea',
    'Kryptonian': 'Human',
    'Luphomoid': 'No idea',
    'Metahuman': 'Human',
    'Mutant': 'Human',
    'New God': 'God',
    'Neyaphem': 'No idea',
    'Rodian': 'Human',
    'Spartoi': 'No idea',
    'Symbiote': 'No idea',
    'Talokite': 'No idea',
    'Tamaranean': 'No idea',
    'Ungaran': 'No idea',
    'Vampire': 'Bad',
    'Yautja': 'No idea',
    "Yoda's species": 'No idea',
    'Zen-Whoberian': 'No idea',
    'Zombie': 'Bad'
}

# FUNCTIONS TO CLEAN DATA

def transform_categorical(data, column_name, method):
    if method == 'dummies':
        return pd.concat([data, pd.get_dummies(data[column_name])],
                          axis=1).drop(columns=[column_name])
    elif method == 'hashing':
        raise BaseException('Needs implementation.') # Pablo is going to implement this.
    else:
        raise BaseException('Choose a method to handle categorical data.')

def generate_distribution(data):
    np.random.seed(42)
    return np.random.normal(data.mean(), data.std()) # Not sure normal is the best dist., we should check.

def clean_data(data, **kwargs):
    data = data * 1 # With this command we transform all True False values to 0 or 1
    data.loc[:, 'Gender'] = data['Gender'].map({'Female': 0, 'Male': 1})
    data['Gender'].fillna(generate_distribution(data['Gender']), inplace=True)
    data.loc[data["Height"] < 0, 'Height'] = np.nan
    data["Height"].fillna(generate_distribution(data["Height"]), inplace=True)
    data.loc[data["Weight"] < 0, 'Weight'] = np.nan
    data["Weight"].fillna(generate_distribution(data['Weight']), inplace=True)
    data.loc[:, 'Eye color'] = data["Eye color"].map(kwargs['eyes_map'])
    data.loc[:, 'Eye color'] = data.loc[:, 'Eye color'] + '_eye'
    data = transform_categorical(data, 'Eye color', kwargs['categorical_handler'])
    data.loc[:, 'Hair color'] = data["Hair color"].map(kwargs['hairs_map'])
    data.loc[:, 'Hair color'] = data.loc[:, 'Hair color'] + '_hair'
    data = transform_categorical(data, 'Hair color', kwargs['categorical_handler'])
    data.loc[:, 'Race'] = data['Race'].map(kwargs['races_map'])
    data = transform_categorical(data, 'Race', kwargs['categorical_handler'])
    data = transform_categorical(data, 'Publisher', kwargs['categorical_handler'])
    data.loc[:, 'Skin color'] = data['Skin color'].map(kwargs['skins_map'])
    data.loc[:, 'Skin color'] = data.loc[:, 'Skin color'] + '_skin'
    data = transform_categorical(data, 'Skin color', kwargs['categorical_handler'])
    return data

## Parameters

parameters = {
    'methods': methods,
    'methods_kwargs': methods_kwargs,
    'columns': chosen_columns,
    'test_size': 0.2,
    'n_iters': 100, 
    'categorical_handler': 'dummies',
    'eyes_map': eye_color_dic,
    'hairs_map': hair_color_dic,
    'races_map': race_dic,
    'skins_map': skin_dic,
}

# RUNNER

run(**parameters)
