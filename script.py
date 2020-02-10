import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
from plotly import tools
from sklearn.model_selection import ShuffleSplit

from assignment1.learning_curve import plot_learning_curve, plot_validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler

import plotly.graph_objs as go
import seaborn as sns

plot_graphs = True
do_whole_grid_search = False
# Reading the dataset
data_cars = pd.read_csv('car_evaluation.csv')
print(data_cars.shape)
data_cars.columns = ['Price', 'Maintenance Cost', 'Number of Doors', 'Capacity', 'Size of Luggage Boot', 'safety',
                     'Decision']
print(data_cars.describe())

data_mobiles = pd.read_csv('mobile_cost.csv')
print(data_mobiles.shape)
print(data_mobiles.describe())
print(data_mobiles.head())

# price = pd.crosstab(data_cars['Price'], data_cars['Decision'])
# price.div(price.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (10, 7))

# plt.title('Decisions taken on each Price Category', fontsize = 20)
# plt.xlabel('Price Range in Increasing Order', fontsize = 15)
# plt.ylabel('Count', fontsize = 15)
# plt.legend()
# plt.show()


# Label Encoding
data_cars.Decision.replace(('unacc', 'acc', 'good', 'vgood'), (0, 1, 2, 3), inplace=True)
data_cars['Size of Luggage Boot'].replace(('small', 'med', 'big'), (0, 1, 2), inplace=True)
data_cars['safety'].replace(('low', 'med', 'high'), (0, 1, 2), inplace=True)
data_cars['Price'].replace(('low', 'med', 'high', 'vhigh'), (0, 1, 2, 3), inplace=True)
data_cars['Maintenance Cost'].replace(('low', 'med', 'high', 'vhigh'), (0, 1, 2, 3), inplace=True)
data_cars['Number of Doors'].replace('5more', 5, inplace=True)
data_cars['Capacity'].replace('more', 5, inplace=True)

cars_x = data_cars.iloc[:, :6]
cars_y = data_cars.iloc[:, 6]

data_mobiles_features = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'int_memory', 'm_dep',
                         'mobile_wt', 'n_cores',
                         'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen',
                         'wifi']

mobiles_x = data_mobiles[data_mobiles_features]
mobiles_y = data_mobiles['price_range']

cars_x_train, cars_x_test, cars_y_train, cars_y_test = train_test_split(cars_x, cars_y, test_size=0.25, random_state=0)
mobiles_x_train, mobiles_x_test, mobiles_y_train, mobiles_y_test = train_test_split(mobiles_x, mobiles_y,
                                                                                    test_size=0.25, random_state=0)
scaler = MinMaxScaler()
cars_x_scaled = scaler.fit_transform(cars_x)
mobiles_x_scaled = scaler.fit_transform(mobiles_x)

cars_x_scaled_train, cars_x_scaled_test, cars_y_scaled_train, cars_y_scaled_test = train_test_split(cars_x_scaled,
                                                                                                    cars_y,
                                                                                                    test_size=0.25,
                                                                                                    random_state=0)
mobiles_x_scaled_train, mobiles_x_scaled_test, mobiles_y_scaled_train, mobiles_y_scaled_test = train_test_split(
    mobiles_x_scaled, mobiles_y,
    test_size=0.25, random_state=0)

corr_cars = data_cars.corr()
fig = plt.figure(figsize=(15, 12))
r = sns.heatmap(corr_cars, cmap='Purples')
r.set_title("Correlation ")
plt.savefig('correlation_cars.png')
plt.close()
corr_cars_price = corr_cars.sort_values(by=["Decision"], ascending=False).iloc[0].sort_values(ascending=False)
print('correlation_cars')
print(corr_cars_price)

corr_mobiles = data_mobiles.corr()
fig = plt.figure(figsize=(15, 12))
r = sns.heatmap(corr_mobiles, cmap='Purples')
r.set_title("Correlation ")
plt.savefig('correlation_mobiles.png')
plt.close()
corr_mobiles_price = corr_mobiles.sort_values(by=["price_range"], ascending=False).iloc[0].sort_values(ascending=False)
print('correlation_mobiles')
print(corr_mobiles_price)

print("Shape of mobiles_x_train: ", mobiles_x_train.shape)
print("Shape of mobiles_y_train: ", mobiles_y_train.shape)

plt1 = plt
axes1 = None

dataset_info_optimal = {
    'cars': {
        'x': cars_x,
        'y': cars_y,
        'x_train': cars_x_train,
        'x_test': cars_x_test,
        'y_train': cars_y_train,
        'y_test': cars_y_test,
        'x_scaled': cars_x_scaled,
        'y_scaled': cars_y,
        'x_scaled_train': cars_x_scaled_train,
        'x_scaled_test': cars_x_scaled_test,
        'y_scaled_train': cars_y_scaled_train,
        'y_scaled_test': cars_y_scaled_test,
        'models': {
            'DT': {
                'enabled': True,
                'title': 'Decision Tree (Cars)',
                'line_color': 'red',
                'classifier': DecisionTreeClassifier(),
                'learning_curve_cv_splits': 20,
                'params': {
                    'plot': {
                        'max_depth': range(2, 19),
                        'max_leaf_nodes': [2, 5, 10, 15, 20, 25, 30, 40, 50, 100]
                    },
                    'search': {
                        'max_depth': [10],  # range(2, 19),
                        'min_samples_leaf': [4],  # ,
                        'max_leaf_nodes': [2, 5, 10, 15, 20, 25, 30, 40, 50, 100],
                        'criterion': ['entropy'],  # ['entropy', 'gini'],
                        'ccp_alpha': [0.002],  # [0.001, 0.002, 0.005, 0.01, 0.02]
                        'random_state': [0]
                    }
                }
            },
            'KNN': {
                'enabled': True,
                'title': 'KNN (Cars)',
                'line_color': 'blue',
                'classifier': KNeighborsClassifier(),
                'learning_curve_cv_splits': 10,
                'params': {
                    'plot': {
                        'n_neighbors': range(1, 30),
                        'leaf_size': [10, 20, 30, 40, 60, 80, 90, 100, 120, 160]  # , 200, 300, 400]
                    },
                    'search': {
                        'n_neighbors': [7],  # range(1, 30),
                        'weights': ['distance'],  # ['uniform', 'distance'],
                        'algorithm': ['ball_tree'],  # ['ball_tree', 'kd_tree', 'brute'],
                        'leaf_size': [160],
                        'p': [1],  # [1,2],
                        'n_jobs': [-1]
                    }
                }
            },
            'ADA_BOOST': {
                'enabled': True,
                'title': 'ADA-Boost Decision Tree (Cars)',
                'line_color': 'green',
                'classifier': AdaBoostClassifier(),
                'learning_curve_cv_splits': 20,
                'params': {
                    'plot': {
                        'n_estimators': [10, 20, 30, 40, 70, 100, 130, 170, 210],
                        'learning_rate': [0.05, 0.1, 0.2, 0.5, 0.7, 1],
                    },
                    'search': {
                        'base_estimator': [
                            # DecisionTreeClassifier(max_depth=5, max_leaf_nodes=30, criterion='entropy', ccp_alpha=0.01, random_state=0),
                            DecisionTreeClassifier(max_depth=6, max_leaf_nodes=40, criterion='entropy', ccp_alpha=0.005,
                                                   random_state=0),
                            # DecisionTreeClassifier(max_depth=7, max_leaf_nodes=50, criterion='entropy', ccp_alpha=0.002, random_state=0),
                        ],
                        'n_estimators': [130],  # range(50, 200, 5),
                        'learning_rate': [0.5],  # [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1],
                        'algorithm': ['SAMME'],  # ['SAMME', 'SAMME.R'],
                        'random_state': [0]
                    }
                }
            },
            'SVM': {
                'enabled': True,
                'title': 'SVM (Cars)',
                'line_color': 'orange',
                'classifier': SVC(),
                'learning_curve_cv_splits': 20,
                'params': {
                    'plot': {
                        'C': [8, 16, 32, 64, 128, 256, 512, 640, 668, 672, 704, 768, 1024],
                        'max_iter': [8, 16, 32, 64, 128, 256, 512, 640, 768, 800, 832, 896, 1024],
                        'kernel': ['sigmoid', 'linear', 'rbf', 'poly']
                    },
                    'search': [
                        {
                            'C': [60],  # [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90],  #
                            'kernel': ['rbf'],  # ['rbf', 'linear', 'sigmoid'],  # ,
                            'tol': [0.1],  # [0.001, 0.01, 0.1],  # ,
                            'max_iter': [-1],
                            'decision_function_shape': ['ovr'],  # ['ovo', 'ovr'],
                            'random_state': [0]
                        },
                        # {
                        #     'C': [2],  # [0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        #     'kernel': ['poly'],  # [],
                        #     'degree': [6],  # [3,4,5,6,7,8,9,10],
                        #     'tol': [0.1],  # [1e-3, 0.01, 0.1, 1],
                        #     'max_iter': [-1 ],
                        #     'decision_function_shape': ['ovr'],  # ['ovo', 'ovr'],
                        #     'random_state': [0]
                        # },
                    ]
                }
            },
            'NN': {
                'enabled': True,
                'title': 'Neural Network (Cars)',
                'line_color': 'purple',
                'classifier': MLPClassifier(),
                'learning_curve_cv_splits': 10,
                'params': {
                    'plot': {
                        'hidden_layer_sizes': [80, 160, 320, 480, 640],
                        'learning_rate_init': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
                        'max_iter': [20, 40, 80, 120, 160, 200, 400, 800],
                    },
                    'search': {
                        'hidden_layer_sizes': [320],  # [80, 160, 320, 480],
                        'activation': ['relu'],  # ['identity', 'logistic', 'tanh', 'relu'],  # ,
                        'learning_rate_init': [0.1],
                        'max_iter': [400],  # [200, 400, 800],
                        'tol': [1e-4],  # ,
                        'random_state': [0]
                    }
                }
            }
        },
    },
    'mobiles': {
        'x': mobiles_x,
        'y': mobiles_y,
        'x_train': mobiles_x_train,
        'x_test': mobiles_x_test,
        'y_train': mobiles_y_train,
        'y_test': mobiles_y_test,
        'x_scaled': mobiles_x_scaled,
        'y_scaled': mobiles_y,
        'x_scaled_train': mobiles_x_scaled_train,
        'x_scaled_test': mobiles_x_scaled_test,
        'y_scaled_train': mobiles_y_scaled_train,
        'y_scaled_test': mobiles_y_scaled_test,
        'models': {
            'DT': {
                'enabled': True,
                'title': 'Decision Tree (Mobiles)',
                'line_color': 'red',
                'classifier': DecisionTreeClassifier(),
                'learning_curve_cv_splits': 20,
                'params': {
                    'plot': {
                        'max_depth': range(2, 20),
                        'max_leaf_nodes': [4, 8, 16, 32, 64, 96, 128]
                    },
                    'search': {
                        'max_depth': [8],  # range(2, 20),
                        'min_samples_leaf': [8],  # [2, 4, 8, 16, 32],
                        'max_leaf_nodes': [64],  # [4, 8, 16, 32, 64, 96, 128]
                        'criterion': ['entropy'],  # ['entropy', 'gini'],
                        'ccp_alpha': [0.005],  # [0.001, 0.002, 0.005, 0.009]
                        'random_state': [0]
                    }
                }
            },
            'KNN': {
                'enabled': True,
                'title': 'KNN (Mobiles)',
                'line_color': 'blue',
                'classifier': KNeighborsClassifier(),
                'learning_curve_cv_splits': 10,
                'params': {
                    'plot': {
                        'n_neighbors': range(1, 50, 4),
                        'leaf_size': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40]
                    },
                    'search': {
                        'n_neighbors': [26],  # (10, 50, 2),
                        'weights': ['distance'],  # ['uniform', 'distance'],
                        'algorithm': ['ball_tree'],  # ['ball_tree', 'kd_tree', 'brute'],
                        'leaf_size': [30],  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40]
                        'p': [1],  # [1,2],
                        'n_jobs': [-1]
                    }
                }
            },
            'ADA_BOOST': {
                'enabled': True,
                'title': 'ADA-Boost Decision Tree (Mobiles)',
                'line_color': 'green',
                'classifier': AdaBoostClassifier(),
                'learning_curve_cv_splits': 20,
                'params': {
                    'plot': {
                        'n_estimators': [10, 20, 40, 60, 90, 120, 150, 200],
                        'learning_rate': [0.1, 0.2, 0.5, 0.7, 1],
                    },
                    'search': {
                        'base_estimator': [
                            # DecisionTreeClassifier(max_depth=5, max_leaf_nodes=30, criterion='entropy', ccp_alpha=0.03, random_state=0),
                            # DecisionTreeClassifier(max_depth=6, max_leaf_nodes=40, criterion='entropy', ccp_alpha=0.02, random_state=0),
                            DecisionTreeClassifier(max_depth=7, max_leaf_nodes=50, criterion='entropy', ccp_alpha=0.01,
                                                   random_state=0),
                            # DecisionTreeClassifier(max_depth=8, max_leaf_nodes=64, criterion='entropy', ccp_alpha=0.005, random_state=0),
                        ],
                        'n_estimators': [130],  # range(10, 150, 10),
                        'learning_rate': [1],  # [0.7, 1],
                        'algorithm': ['SAMME'],  # ['SAMME', 'SAMME.R'],  #
                        'random_state': [0]
                    }
                }
            },
            'SVM': {
                'enabled': True,
                'title': 'SVM (Mobiles)',
                'line_color': 'orange',
                'classifier': SVC(),
                'learning_curve_cv_splits': 20,
                'params': {
                    'plot': {
                        'C': [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.1, 0.2],
                        'max_iter': [4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000],
                        'kernel': ['linear', 'rbf', 'poly'],  # ['sigmoid', 'linear', 'rbf', 'poly'],
                    },
                    'search': [
                        {
                            'C': [1],  # [0.1, 0.2, 0.5, 1, 2, 5, 10],,
                            'kernel': ['linear'],  # ['rbf', 'linear', 'sigmoid],
                            'tol': [0.1],  # [0.001, 0.005, 0.1, 0.15, 0.2, 0.5, 1],  # [],
                            'max_iter': [-1],
                            'decision_function_shape': ['ovr'],  # ['ovo', 'ovr'],
                            'random_state': [0]
                        },
                        # {
                        #     # 'C': [2],  # [0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        #     'kernel': ['poly'],  # [],
                        #     'degree': [3,4,5,6,7,8,9,10],
                        #     # 'tol': [1e-3, 0.01, 0.1, 1],
                        #     'max_iter': [-1 ],
                        #     'decision_function_shape': ['ovr'],  # ['ovo', 'ovr'],
                        #     'random_state': [0]
                        # },
                    ]
                }
            },
            'NN': {
                'enabled': True,
                'title': 'Neural Network (Mobiles)',
                'line_color': 'purple',
                'classifier': MLPClassifier(),
                'learning_curve_cv_splits': 10,
                'params': {
                    'plot': {
                        'hidden_layer_sizes': [5, 10, 20, 40, 80, 160],
                        'learning_rate_init': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
                        'max_iter': [20, 40, 80, 120, 160, 200, 400, 800],
                    },
                    'search': {
                        'hidden_layer_sizes': [40],  # [10, 20, 40, 80, 160],
                        'activation': ['logistic'],  # ['identity', 'logistic', 'tanh', 'relu'],  # ['relu'] # ,
                        'learning_rate_init': [0.01],  # [0.01, 0.02, 0.05, 0.1, 0.2],
                        'max_iter': [200],  # [100, 200, 400, 800],  # [400],  #
                        'tol': [1e-3],  # ,
                        'random_state': [0]
                    }
                }
            }
        },
    },
}

dataset_info_grid_search = {
    'cars': {
        'x': cars_x,
        'y': cars_y,
        'x_train': cars_x_train,
        'x_test': cars_x_test,
        'y_train': cars_y_train,
        'y_test': cars_y_test,
        'x_scaled': cars_x_scaled,
        'y_scaled': cars_y,
        'x_scaled_train': cars_x_scaled_train,
        'x_scaled_test': cars_x_scaled_test,
        'y_scaled_train': cars_y_scaled_train,
        'y_scaled_test': cars_y_scaled_test,
        'models': {
            'DT': {
                'enabled': True,
                'title': 'Decision Tree (Cars)',
                'line_color': 'red',
                'classifier': DecisionTreeClassifier(),
                'learning_curve_cv_splits': 20,
                'params': {
                    'plot': {
                        'max_depth': range(2, 19),
                        'max_leaf_nodes': [2, 5, 10, 15, 20, 25, 30, 40, 50, 100]
                    },
                    'search': {
                        'max_depth': range(2, 19),  # 10
                        'min_samples_leaf': [4],  # ,
                        'max_leaf_nodes': [2, 5, 10, 15, 20, 25, 30, 40, 50, 100],
                        'criterion': ['entropy', 'gini'],  # ['entropy'],  #
                        'ccp_alpha': [0.001, 0.002, 0.005, 0.01, 0.02],  # [0.002],  #
                        'random_state': [0]
                    }
                }
            },
            'KNN': {
                'enabled': True,
                'title': 'KNN (Cars)',
                'line_color': 'blue',
                'classifier': KNeighborsClassifier(),
                'learning_curve_cv_splits': 10,
                'params': {
                    'plot': {
                        'n_neighbors': range(1, 30),
                        'leaf_size': [10, 20, 30, 40, 60, 80, 90, 100, 120, 160]  # , 200, 300, 400]
                    },
                    'search': {
                        'n_neighbors': range(1, 30),  # [7],  #
                        'weights': ['uniform', 'distance'],  # ['distance'],  #
                        'algorithm': ['ball_tree', 'kd_tree', 'brute'],  # ['ball_tree'],  #
                        'leaf_size': [10, 20, 30, 40, 60, 80, 90, 100, 120, 160, 200, 300, 400],  # [160],
                        'p': [1, 2],  # [1],
                        'n_jobs': [-1]
                    }
                }
            },
            'ADA_BOOST': {
                'enabled': True,
                'title': 'ADA-Boost Decision Tree (Cars)',
                'line_color': 'green',
                'classifier': AdaBoostClassifier(),
                'learning_curve_cv_splits': 20,
                'params': {
                    'plot': {
                        'n_estimators': [10, 20, 30, 40, 70, 100, 130, 170, 210],
                        'learning_rate': [0.05, 0.1, 0.2, 0.5, 0.7, 1],
                    },
                    'search': {
                        'base_estimator': [
                            DecisionTreeClassifier(max_depth=5, max_leaf_nodes=30, criterion='entropy', ccp_alpha=0.01,
                                                   random_state=0),
                            DecisionTreeClassifier(max_depth=6, max_leaf_nodes=40, criterion='entropy', ccp_alpha=0.005,
                                                   random_state=0),
                            DecisionTreeClassifier(max_depth=7, max_leaf_nodes=50, criterion='entropy', ccp_alpha=0.002,
                                                   random_state=0),
                        ],
                        'n_estimators': range(50, 200, 5),  # [130],  #
                        'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1],  # [0.5],  #
                        'algorithm': ['SAMME', 'SAMME.R'],  # ['SAMME'],  #
                        'random_state': [0]
                    }
                }
            },
            'SVM': {
                'enabled': True,
                'title': 'SVM (Cars)',
                'line_color': 'orange',
                'classifier': SVC(),
                'learning_curve_cv_splits': 20,
                'params': {
                    'plot': {
                        'C': [8, 16, 32, 64, 128, 256, 512, 640, 668, 672, 704, 768, 1024],
                        'max_iter': [8, 16, 32, 64, 128, 256, 512, 640, 768, 800, 832, 896, 1024],
                        'kernel': ['sigmoid', 'linear', 'rbf', 'poly']
                    },
                    'search': [
                        {
                            'C': [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90],  # [60],  #
                            'kernel': ['rbf', 'linear', 'sigmoid'],  # ['rbf'],  # ,
                            'tol': [0.001, 0.01, 0.1],  # [0.1],  # ,
                            'max_iter': [-1],
                            'decision_function_shape': ['ovo', 'ovr'],  # ['ovr'],  #
                            'random_state': [0]
                        },
                        {
                            'C': [0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # [2],  #
                            'kernel': ['poly'],  # [],
                            'degree': [3, 4, 5, 6, 7, 8, 9, 10],  # [6],  #
                            'tol': [1e-3, 0.01, 0.1, 1],  # [0.1],  #
                            'max_iter': [-1],
                            'decision_function_shape': ['ovo', 'ovr'],  # ['ovr'],  #
                            'random_state': [0]
                        },
                    ]
                }
            },
            'NN': {
                'enabled': True,
                'title': 'Neural Network (Cars)',
                'line_color': 'purple',
                'classifier': MLPClassifier(),
                'learning_curve_cv_splits': 10,
                'params': {
                    'plot': {
                        'hidden_layer_sizes': [80, 160, 320, 480, 640],
                        'learning_rate_init': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
                        'max_iter': [20, 40, 80, 120, 160, 200, 400, 800],
                    },
                    'search': {
                        'hidden_layer_sizes': [80, 160, 320, 480],  # [320],
                        'activation': ['identity', 'logistic', 'tanh', 'relu'],  # ['relu'],  # ,
                        'learning_rate_init': [0.1],
                        'max_iter': [200, 400, 800],  # [400],
                        'tol': [1e-4],  # ,
                        'random_state': [0]
                    }
                }
            }
        },
    },
    'mobiles': {
        'x': mobiles_x,
        'y': mobiles_y,
        'x_train': mobiles_x_train,
        'x_test': mobiles_x_test,
        'y_train': mobiles_y_train,
        'y_test': mobiles_y_test,
        'x_scaled': mobiles_x_scaled,
        'y_scaled': mobiles_y,
        'x_scaled_train': mobiles_x_scaled_train,
        'x_scaled_test': mobiles_x_scaled_test,
        'y_scaled_train': mobiles_y_scaled_train,
        'y_scaled_test': mobiles_y_scaled_test,
        'models': {
            'DT': {
                'enabled': True,
                'title': 'Decision Tree (Mobiles)',
                'line_color': 'red',
                'classifier': DecisionTreeClassifier(),
                'learning_curve_cv_splits': 20,
                'params': {
                    'plot': {
                        'max_depth': range(2, 20),
                        'max_leaf_nodes': [4, 8, 16, 32, 64, 96, 128]
                    },
                    'search': {
                        'max_depth': range(2, 20),  # [8],  #
                        'min_samples_leaf': [2, 4, 8, 16, 32],  # [8],  #
                        'max_leaf_nodes': [4, 8, 16, 32, 64, 96, 128],  # [64],  #
                        'criterion': ['entropy', 'gini'],  # ['entropy'],  #
                        'ccp_alpha': [0.001, 0.002, 0.005, 0.009],  # [0.005],  #
                        'random_state': [0]
                    }
                }
            },
            'KNN': {
                'enabled': True,
                'title': 'KNN (Mobiles)',
                'line_color': 'blue',
                'classifier': KNeighborsClassifier(),
                'learning_curve_cv_splits': 10,
                'params': {
                    'plot': {
                        'n_neighbors': range(1, 50, 4),
                        'leaf_size': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40]
                    },
                    'search': {
                        'n_neighbors': (10, 50, 2),  # [26],  #
                        'weights': ['uniform', 'distance'],  # ['distance'],  #
                        'algorithm': ['ball_tree', 'kd_tree', 'brute'],  # ['ball_tree'],  #
                        'leaf_size': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40],  # [30],  #
                        'p': [1, 2],  # [1],
                        'n_jobs': [-1]
                    }
                }
            },
            'ADA_BOOST': {
                'enabled': True,
                'title': 'ADA-Boost Decision Tree (Mobiles)',
                'line_color': 'green',
                'classifier': AdaBoostClassifier(),
                'learning_curve_cv_splits': 20,
                'params': {
                    'plot': {
                        'n_estimators': [10, 20, 40, 60, 90, 120, 150, 200],
                        'learning_rate': [0.1, 0.2, 0.5, 0.7, 1],
                    },
                    'search': {
                        'base_estimator': [
                            DecisionTreeClassifier(max_depth=5, max_leaf_nodes=30, criterion='entropy', ccp_alpha=0.03,
                                                   random_state=0),
                            DecisionTreeClassifier(max_depth=6, max_leaf_nodes=40, criterion='entropy', ccp_alpha=0.02,
                                                   random_state=0),
                            DecisionTreeClassifier(max_depth=7, max_leaf_nodes=50, criterion='entropy', ccp_alpha=0.01,
                                                   random_state=0),
                            DecisionTreeClassifier(max_depth=8, max_leaf_nodes=64, criterion='entropy', ccp_alpha=0.005,
                                                   random_state=0),
                        ],
                        'n_estimators': range(10, 150, 10),  # [130],  #
                        'learning_rate': [0.7, 1],  # [1],
                        'algorithm': ['SAMME', 'SAMME.R'],  # ['SAMME'],  #
                        'random_state': [0]
                    }
                }
            },
            'SVM': {
                'enabled': True,
                'title': 'SVM (Mobiles)',
                'line_color': 'orange',
                'classifier': SVC(),
                'learning_curve_cv_splits': 20,
                'params': {
                    'plot': {
                        'C': [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.1, 0.2],
                        'max_iter': [4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000],
                        'kernel': ['linear', 'rbf', 'poly'],  # ['sigmoid', 'linear', 'rbf', 'poly'],
                    },
                    'search': [
                        {
                            'C': [0.1, 0.2, 0.5, 1, 2, 5, 10],  # [1],  #
                            'kernel': ['rbf', 'linear', 'sigmoid'],  # ['linear'],  #
                            'tol': [0.001, 0.005, 0.1, 0.15, 0.2, 0.5, 1],  # [0.1],  # ,
                            'max_iter': [-1],
                            'decision_function_shape': ['ovo', 'ovr'],  # ['ovr'],  #
                            'random_state': [0]
                        },
                        {
                            'C': [0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # [2],  #
                            'kernel': ['poly'],  # [],
                            'degree': [3, 4, 5, 6, 7, 8, 9, 10],
                            'tol': [1e-3, 0.01, 0.1, 1],
                            'max_iter': [-1],
                            'decision_function_shape': ['ovo', 'ovr'],  # ['ovr'],  #
                            'random_state': [0]
                        },
                    ]
                }
            },
            'NN': {
                'enabled': True,
                'title': 'Neural Network (Mobiles)',
                'line_color': 'purple',
                'classifier': MLPClassifier(),
                'learning_curve_cv_splits': 10,
                'params': {
                    'plot': {
                        'hidden_layer_sizes': [5, 10, 20, 40, 80, 160],
                        'learning_rate_init': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
                        'max_iter': [20, 40, 80, 120, 160, 200, 400, 800],
                    },
                    'search': {
                        'hidden_layer_sizes': [10, 20, 40, 80, 160],  # [40],  #
                        'activation': ['identity', 'logistic', 'tanh', 'relu'],  # ['logistic'],  #
                        'learning_rate_init': [0.01, 0.02, 0.05, 0.1, 0.2],  # [0.01],  #
                        'max_iter': [100, 200, 400, 800],  # [200, 400],  #   #
                        'tol': [1e-3],  # ,
                        'random_state': [0]
                    }
                }
            }
        },
    },
}


def executor(classifier, line_color, title, x, y, x_train, x_test, y_train, y_test, params, learning_curve_cv_splits):
    print(title)
    param_grid = params['search']

    grid = GridSearchCV(classifier, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    ts1 = datetime.now()
    grid.fit(x_train, y_train)
    ts2 = datetime.now()
    print('training_time')
    print(ts2 - ts1)
    results = pd.DataFrame(grid.cv_results_)
    print(title)
    print('best_params')
    print(grid.best_params_)
    print('best_score')
    print(grid.best_score_)
    print('best_estimator')
    print(grid.best_estimator_)

    ts3 = datetime.now()
    y_pred = grid.predict(x_test)
    ts4 = datetime.now()
    print('test_time')
    print(ts4 - ts3)

    print(title)
    print("\nTraining Accuracy: ", grid.score(x_train, y_train))
    print("Testing Accuracy: ", grid.score(x_test, y_test))
    # printing the confusion Matrix
    print(title)
    print('Confusion Matrix')
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    if (plot_graphs):
        cv = ShuffleSplit(n_splits=learning_curve_cv_splits, test_size=0.2, random_state=0)
        # plt1 = plt1 or plt
        global plt1, axes1
        plt1, axes1 = plot_learning_curve(grid.best_estimator_, title, x, y, cv=cv, plt=plt1, axes=axes1,
                                          line_color=line_color)
        for p in params['plot']:
            plot_validation_curve(grid.best_estimator_, title + '_' + p, x, y, cv=cv,
                                  param_name=p, param_range=params['plot'][p],
                                  best_val=grid.best_params_[p])


if do_whole_grid_search:
    dataset_info = dataset_info_grid_search
else:
    dataset_info = dataset_info_optimal

for d in dataset_info:
    dataset = dataset_info[d]
    for m in dataset['models']:
        model = dataset['models'][m]
        if model['enabled']:
            if m in ['SVM', 'NN']:
                executor(
                    classifier=model['classifier'],
                    line_color=model['line_color'],
                    title=model['title'],
                    x=dataset['x_scaled'],
                    y=dataset['y_scaled'],
                    x_train=dataset['x_scaled_train'],
                    x_test=dataset['x_scaled_test'],
                    y_train=dataset['y_scaled_train'],
                    y_test=dataset['y_scaled_test'],
                    learning_curve_cv_splits=model['learning_curve_cv_splits'],
                    params=model['params']

                )
            else:
                executor(
                    classifier=model['classifier'],
                    line_color=model['line_color'],
                    title=model['title'],
                    x=dataset['x'],
                    y=dataset['y'],
                    x_train=dataset['x_train'],
                    x_test=dataset['x_test'],
                    y_train=dataset['y_train'],
                    y_test=dataset['y_test'],
                    learning_curve_cv_splits=model['learning_curve_cv_splits'],
                    params=model['params']
                )
    plt1.savefig('trial' + d + '.png')
    plt1.close()
    plt1 = plt
    axes1 = None
print('done')
