# Computación Gráfica, Visión Computacional y Multimedia
#   @author: Carlos Alberto Mestas Escarcena
#   @author: David Jose Peña Ugarte

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

print("Histograma")
histogram = pd.read_csv('histogram.csv')
y = histogram.label
X = histogram.drop(columns='label').drop(columns='img')
X = X.astype(float)
# print(dates['label'].value_counts())


sc = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y)

X_train_array = sc.fit_transform(X_train.values)
X_train = pd.DataFrame(X_train_array, index=X_train.index, columns=X_train.columns)
X_test_array = sc.transform(X_test.values)
X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)

paramGrid = {'C': [1, 10, 100, 1000],
             'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, ], }
estimator = GridSearchCV(
    estimator=SVC(kernel="rbf"),
    param_grid=paramGrid,
    scoring='accuracy',
    n_jobs=8,
    return_train_score=True)
estimator.fit(X_train, y_train)

print("Resultados")
print(estimator.score(X_test, y_test))

print("Best estimator found by grid search:")
print(estimator.best_params_)

modelo = estimator.best_estimator_
predicciones = modelo.predict(X_test)

accuracy = accuracy_score(
    y_true=y_test,
    y_pred=predicciones,
    normalize=True
)
print("")
print(f"El accuracy de test es: {100 * accuracy}%")

recall = recall_score(
    y_true=y_test,
    y_pred=predicciones,
    average='macro'
)
print("")
print(f"El recall de test es: {100 * recall}%")

f1score = f1_score(
    y_true=y_test,
    y_pred=predicciones,
    average='macro'
)
print("")
print(f"El f1-score de test es: {100 * f1score}%")

confusion_matrix = pd.crosstab(
    y_test.ravel(),
    predicciones,
    rownames=['Real'],
    colnames=['Predicción']
)
print(confusion_matrix)


print("--------------------------------")

print("Momentos")
moments = pd.read_csv('statisticalMoments.csv')
y = moments.label
X = moments.drop(columns='label').drop(columns='img')
X = X.astype(float)
# print(moments['label'].value_counts())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y)

paramGrid = {'C': [1, 10, 100, 1000],
             'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, ], }
estimator = GridSearchCV(
    estimator=SVC(kernel="rbf"),
    param_grid=paramGrid,
    scoring='accuracy',
    n_jobs=8,
    return_train_score=True)

X_train_array = sc.fit_transform(X_train.values)
X_train = pd.DataFrame(X_train_array, index=X_train.index, columns=X_train.columns)
X_test_array = sc.transform(X_test.values)
X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)

estimator.fit(X_train, y_train)
print("Resultados")
print(estimator.score(X_test, y_test))
print("Best estimator found by grid search:")
print(estimator.best_params_)

modelo = estimator.best_estimator_
predicciones = modelo.predict(X_test)

accuracy = accuracy_score(
    y_true=y_test,
    y_pred=predicciones,
    normalize=True
)
print("")
print(f"El accuracy de test es: {100 * accuracy}%")

recall = recall_score(
    y_true=y_test,
    y_pred=predicciones,
    average='macro'
)
print("")
print(f"El recall de test es: {100 * recall}%")

f1score = f1_score(
    y_true=y_test,
    y_pred=predicciones,
    average='macro'
)
print("")
print(f"El f1-score de test es: {100 * f1score}%")

confusion_matrix = pd.crosstab(
    y_test.ravel(),
    predicciones,
    rownames=['Real'],
    colnames=['Predicción']
)
print(confusion_matrix)

print("--------------------------------")

print("LBP - Radio 1")
dates3 = pd.read_csv('lbp.csv')
y = dates3.label
X = dates3.drop(columns='label').drop(columns='img')
X = X.astype(float)
# print(dates3['label'].value_counts())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y)

paramGrid = {'C': [1, 10, 100, 1000],
             'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, ], }
estimator = GridSearchCV(
    estimator=SVC(kernel="rbf"),
    param_grid=paramGrid,
    scoring='accuracy',
    n_jobs=8,
    return_train_score=True)

X_train_array = sc.fit_transform(X_train.values)
X_train = pd.DataFrame(X_train_array, index=X_train.index, columns=X_train.columns)
X_test_array = sc.transform(X_test.values)
X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)

estimator.fit(X_train, y_train)
print("Resultados")
print(estimator.score(X_test, y_test))
print("Best estimator found by grid search:")
print(estimator.best_params_)
modelo = estimator.best_estimator_
predicciones = modelo.predict(X_test)

accuracy = accuracy_score(
    y_true=y_test,
    y_pred=predicciones,
    normalize=True
)
print("")
print(f"El accuracy de test es: {100 * accuracy}%")

recall = recall_score(
    y_true=y_test,
    y_pred=predicciones,
    average='macro'
)
print("")
print(f"El recall de test es: {100 * recall}%")

f1score = f1_score(
    y_true=y_test,
    y_pred=predicciones,
    average='macro'
)
print("")
print(f"El f1-score de test es: {100 * f1score}%")

confusion_matrix = pd.crosstab(
    y_test.ravel(),
    predicciones,
    rownames=['Real'],
    colnames=['Predicción']
)
print(confusion_matrix)

print("--------------------------------")

print("LBP- radio 2")
dates4 = pd.read_csv('lbp2.csv')
y = dates4.label
X = dates4.drop(columns='label').drop(columns='img')
X = X.astype(float)
# print(dates3['label'].value_counts())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y)

paramGrid = {'C': [1, 10, 100, 1000],
             'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, ], }
estimator = GridSearchCV(
    estimator=SVC(kernel="rbf"),
    param_grid=paramGrid,
    scoring='accuracy',
    n_jobs=8,
    return_train_score=True)

X_train_array = sc.fit_transform(X_train.values)
X_train = pd.DataFrame(X_train_array, index=X_train.index, columns=X_train.columns)
X_test_array = sc.transform(X_test.values)
X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)

estimator.fit(X_train, y_train)
print("Resultados ")
print(estimator.score(X_test, y_test))
print("Best estimator found by grid search:")
print(estimator.best_params_)


modelo = estimator.best_estimator_
predicciones = modelo.predict(X_test)

accuracy = accuracy_score(
    y_true=y_test,
    y_pred=predicciones,
    normalize=True
)
print("")
print(f"El accuracy de test es: {100 * accuracy}%")

recall = recall_score(
    y_true=y_test,
    y_pred=predicciones,
    average='macro'
)
print("")
print(f"El recall de test es: {100 * recall}%")

f1score = f1_score(
    y_true=y_test,
    y_pred=predicciones,
    average='macro'
)
print("")
print(f"El f1-score de test es: {100 * f1score}%")

confusion_matrix = pd.crosstab(
    y_test.ravel(),
    predicciones,
    rownames=['Real'],
    colnames=['Predicción']
)
print(confusion_matrix)

print("--------------------------------")

print("LBP")
dates5 = pd.read_csv('lbp3.csv')
y = dates5.label
X = dates5.drop(columns='label').drop(columns='img')
X = X.astype(float)
# print(dates3['label'].value_counts())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y)

paramGrid = {'C': [1, 10, 100, 1000],
             'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, ], }
estimator = GridSearchCV(
    estimator=SVC(kernel="rbf"),
    param_grid=paramGrid,
    scoring='accuracy',
    n_jobs=8,
    return_train_score=True)

X_train_array = sc.fit_transform(X_train.values)
X_train = pd.DataFrame(X_train_array, index=X_train.index, columns=X_train.columns)
X_test_array = sc.transform(X_test.values)
X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)
estimator.fit(X_train, y_train)

print("Resultados radio 3")
print(estimator.score(X_test, y_test))
print("Best estimator found by grid search:")
print(estimator.best_params_)

modelo = estimator.best_estimator_
predicciones = modelo.predict(X_test)

accuracy = accuracy_score(
    y_true=y_test,
    y_pred=predicciones,
    normalize=True
)
print("")
print(f"El accuracy de test es: {100 * accuracy}%")

recall = recall_score(
    y_true=y_test,
    y_pred=predicciones,
    average='macro'
)
print("")
print(f"El recall de test es: {100 * recall}%")

f1score = f1_score(
    y_true=y_test,
    y_pred=predicciones,
    average='macro'
)
print("")
print(f"El f1-score de test es: {100 * f1score}%")

confusion_matrix = pd.crosstab(
    y_test.ravel(),
    predicciones,
    rownames=['Real'],
    colnames=['Predicción']
)
print(confusion_matrix)
print("--------------------------------")
