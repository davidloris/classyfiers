import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_validate
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

def acc_cross_validate(method, data, n_groups):
    return cross_validate(method, data.drop(columns=['Alignment']),
                          data['Alignment'], cv=n_groups)['test_score']

def outliers(data, column, max_val, min_val):
    return data[(data[column] < max_val) & (data[column] > min_val)][column]

def run(**kwargs):
    data = pd.read_csv('train.csv')[kwargs['columns']]
    test_data = pd.read_csv('test.csv')[kwargs['columns'][1:]] #Dropping 'Alignment' column.
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
            acc += np.mean(acc_cross_validate(method(**method_kwargs), cleaned_data, kwargs['n_groups'])) / N
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
                  "Publisher", "Skin color", "Height", "Weight",
                  'Agility', 'Accelerated Healing', 'Durability', 'Flight',
                  'Intelligence', 'Super Strength', 'Energy Blasts', 'Stamina',
                  'Super Speed', 'Reflexes', 'Empathy', ]
                #   "Accelerated Healing", "Lantern Power Ring",
                #   "Dimensional Awareness", "Cold Resistance", "Durability",
                #   "Stealth", "Energy Absorption", "Flight", "Danger Sense"]

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
    'Alpha': 'Mutant', #change in mutant
    'Amazon': 'Semihuman', #1 semi hmns
    'Asgardian': 'God', #New cat
    'Atlantean': 'Semihuman', #New cat
    'Bizarro': 'Semihuman', #1 semi hmns
    'Clone': 'Semihuman', #just one
    'Cosmic Entity': 'God', #no good
    'Cyborg': 'Android',
    'Dathomirian Zabrak': 'Alien', #alien? 1
    'Demi-God': 'God', 
    'Demon': 'Demon', #let it
    'Eternal': 'God', #God? 1
    'Flora Colossus': 'Alien', #alien? 1
    'God / Eternal': 'God',
    'Gorilla': 'Animal',
    'Gungan': 'Alien', #Alien? 1
    'Human / Altered': 'Semihuman', #1 semi hmns
    'Human / Clone': 'Semihuman', #1 semi hmns
    'Human / Cosmic': 'Semihuman', #1 semi hmns
    'Human / Radiation': 'Radioactive', #let it
    'Human-Kree': 'Semihuman', # semi hmns?
    'Human-Vuldarian': 'Semihuman', #1 semi hmns
    'Icthyo Sapien': 'Semihuman', #1 semi hmns
    'Inhuman' : 'Alien', #Alien?
    'Korugaran': 'Alien', #Alien?
    'Kryptonian': 'Alien', #Alien
    'Luphomoid': 'Semihuman', #1 semi hmns
    'Metahuman': 'Mutant', #mutant
    'Mutant': 'Mutant', #let it
    'New God': 'God',
    'Neyaphem': 'Demon', #Demon
    'Rodian': 'Alien', #alien
    'Spartoi': 'Alien', #alien
    'Symbiote': 'Alien', #alien
    'Talokite': 'Alien', #alien
    'Tamaranean': 'Alien', #alien
    'Ungaran': 'Alien', #alien
    'Vampire': 'Semihuman', #semihmn
    'Yautja': 'Alien', #alien
    "Yoda's species": 'Alien', #alien
    'Zen-Whoberian': 'Alien', #alien
    'Zombie': 'Semihuman' #Semihmns
    }

publ_dic = {
    'SyFy': 'other',
    'Star Trek': 'other',
    'Wildstorm' : 'other',
    'Shueisha' : 'other',
    'ABC Studios' : 'other',
    'IDW Publishing' : 'other',
    'Team Epic TV' : 'other',
    'Rebellion' : 'other',
    'HarperCollins' : 'other',
    'Icon Comics' : 'other',
    'J. K. Rowling' : 'other'
    }

# FUNCTIONS TO CLEAN DATA

def StringHash(a, m=257, C=1024):  # m=257, C=1024
# m represents the estimated cardinality of the items set
# C represents a number that is larger that ord(c)
    hash=0
    for i in range(len(a)):
        hash = (hash * C + ord(a[i])) % m
    return hash

def transform_categorical(data, column_name, method):
    if method == 'dummies':
        return pd.concat([data, pd.get_dummies(data[column_name])],
                          axis=1).drop(columns=[column_name])
    elif method == 'hashing':
        data[column_name] = data[column_name].replace(np.nan, 'NaN')
        data[column_name] = data[column_name].apply(lambda x: StringHash(x))
        data[column_name] = data[column_name].replace(StringHash('NaN'), np.nan)
        data[column_name] = generate_distribution(data[column_name])
        return data
    else:
        raise BaseException('Choose a method to handle categorical data.')

def generate_distribution(data):
    return np.random.normal(data.mean(), data.std()) # Not sure normal is the best dist., we should check.

def clean_data(data, **kwargs):
    data = data * 1 # With this command we transform all True False values to 0 or 1
    data.loc[:, 'Gender'] = data['Gender'].map({'Female': 0, 'Male': 1})
    data['Gender'].fillna(generate_distribution(data['Gender']), inplace=True)
    data.loc[data["Height"] < 0, 'Height'] = np.nan
    data["Height"].fillna(generate_distribution(data["Height"]), inplace=True)
    # data['Height'] = outliers(data, 'Height', kwargs['max_height'], kwargs['min_height'])
    data.loc[data["Weight"] < 0, 'Weight'] = np.nan
    data["Weight"].fillna(generate_distribution(data['Weight']), inplace=True)
    # data['Weight'] = outliers(data, 'Weight', kwargs['max_weight'], kwargs['min_weight'])
    data.loc[:, 'Eye color'] = data["Eye color"].map(kwargs['eyes_map'])
    data.loc[:, 'Eye color'] = data.loc[:, 'Eye color'] + '_eye'
    data = transform_categorical(data, 'Eye color', kwargs['categorical_handler'])
    data.loc[:, 'Hair color'] = data["Hair color"].map(kwargs['hairs_map'])
    data.loc[:, 'Hair color'] = data.loc[:, 'Hair color'] + '_hair'
    data = transform_categorical(data, 'Hair color', kwargs['categorical_handler'])
    data.loc[:, 'Race'] = data['Race'].map(kwargs['races_map'])
    data = transform_categorical(data, 'Race', kwargs['categorical_handler'])
    data.loc[:, 'Publisher'] = data['Publisher'].map(kwargs['publisher_map'])
    data.loc[:, 'Publisher'] = data.loc[:, 'Publisher'] + 'publish'
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
    'publisher_map': publ_dic,
    'n_groups': 10,
    'max_height': 300,
    'min_height': 100,
    'max_weight': 600,
    'min_weight': 0,
}

# RUNNER
np.random.seed(42)
run(**parameters)
