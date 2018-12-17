import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib

df = pd.read_csv("EYData.csv")

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



features_df = pd.get_dummies(df, columns=['Presenting Complaint Desc.','Diagnosis Desc.'])
del features_df['Triage Priority']

X = features_df.as_matrix()
y = df['Triage Priority'].as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = ensemble.GradientBoostingRegressor(
    n_estimators=1500,
    learning_rate = 0.1,
    max_depth=6,
    min_samples_leaf=9,
    max_features=0.1,
    loss='huber'
)

model.fit(X_train, y_train)

joblib.dump(model, 'trained_ed_triaging_model.pkl')

mse = mean_absolute_error(y_train, model.predict(X_train))
print("Training Set Mean Absolute Error: %.4f" % mse)

mse = mean_absolute_error(y_test, model.predict(X_test))
print("Test Set Mean Absolute Error: %.4f" % mse)