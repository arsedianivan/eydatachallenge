from sklearn.externals import   joblib

model = joblib.load('trained_ed_triaging_model.pkl')

patient_to_triage = [
        12,
        0,
        1

]

patients_to_triage = [ patient_to_triage]

triage_priorities = model.predict(patients_to_triage)

triage_priority = triage_priorities[0]

print("The subject patient is triaged as priority #".format(triage_priority))