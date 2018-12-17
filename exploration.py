import pandas as pd
import webbrowser
import os

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

html = features_df[0:1000].to_html()

with open("data.html", "w") as f:
    f.write(html)

full_filename = os.path.abspath("data.html")
webbrowser.open("file://{}".format(full_filename))