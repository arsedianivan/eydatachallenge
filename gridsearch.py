import pandas
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

df = pandas.read_csv("EYData.csv")

del df['MRN']
del df['Presentation Visit Number']
del df['Arrival Date']
del df['Dr Seen Date']
del df['Depart Actual Date']
del df['Depart Status Code']
del df['Departure Status Desc.']
del df['Depart. Dest. Code']
del df['Depart. Dest. Desc.']
del df['TimeDiff Arrival-Actual Depart (mins)']
del df['TimeDiff TreatDrNr-Act. Depart (mins)']
del df['Presenting Complaint Code']
del df['Diag Code']


features_df = pandas.get_dummies(df, columns=['Presenting Complaint Desc.','Diagnosis Desc.'])
del features_df['Triage Priority']

X = features_df.as_matrix()
y = df['Triage Priority'].as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = ensemble.GradientBoostingRegressor()

param_grid = {
    'n_estimators' : [500, 1000, 3000, 5000],
    'max_depth' : [4,6],
    'min_samples_leaf': [3, 5, 9, 17],
    'learning_rate': [0.1, 0.05, 0.02, 0.01],
    'max_features': [1.0, 0.3, 0.1],
    'loss': ['ls', 'lad', 'huber']
}

gs_cv = GridSearchCV(model, param_grid, n_jobs=4)

gs_cv.fit(X_train, y_train)

print(gs_cv.best_params_)

mse = mean_absolute_error(y_train, gs_cv.predict(X_train))
print("Training Set Mean Absolute Error: %.4f" % mse)


mse = mean_absolute_error(y_test, gs_cv.predict(X_test))
print("Test Set Mean Absolute Error: %.4f" % mse)
