import joblib
import bentoml
with open("/userRepoData/taeefnajib/quickstart-classification/__sidetrek__/models/99604e83117863c6d9ac404493117504.joblib", 'rb') as f:
    model = joblib.load(f)
    saved_model = bentoml.sklearn.save_model(
        "quickstart_model",
        model,
    )
    print(saved_model) # This is required!
