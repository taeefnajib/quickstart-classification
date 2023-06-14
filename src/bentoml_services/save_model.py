"""
THIS IS JUST A TEMPLATE - CHANGE IT TO FIT YOUR NEEDS
"""

import joblib
import bentoml


with open("/userRepoData/__sidetrek__/taeefnajib/quickstart-classification/bentoml/models/2337d110f6e74a7ec9a62d202845c30a.joblib", "rb") as f:
    model = joblib.load(f)
    saved_model = bentoml.sklearn.save_model(
        "quickstart_model",
        model,
    )
    print(saved_model) # This is required!