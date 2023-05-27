import pandas as pd
import typing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dataclasses_json import dataclass_json
from dataclasses import dataclass

@dataclass_json
@dataclass
class Hyperparameters(object):
    filepath: str = "quickstart-classification-dataset.csv"
    test_size:float = 0.2
    random_state: int = 6
    n_estimators: int=200 
    max_depth:int= 10
    min_samples_leaf:int= 2
    max_leaf_nodes:int= 5
    n_jobs:int= 15

hp = Hyperparameters()


# create dataframe
def create_df(hp: Hyperparameters)->pd.DataFrame:
    return pd.read_csv(hp.filepath)

# split dataset
def split_ds(hp: Hyperparameters, df: pd.DataFrame)->typing.Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(["class"], axis=1)
    y = df["class"]
    return train_test_split(X, y, test_size=hp.test_size, random_state=hp.random_state)

# train model
def train_model(hp: Hyperparameters, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators = hp.n_estimators, 
                               max_depth = hp.max_depth, 
                               min_samples_leaf = hp.min_samples_leaf, 
                               max_leaf_nodes = hp.max_leaf_nodes, 
                               n_jobs = hp.n_jobs)
    return model.fit(X_train, y_train)

# predict test data
def predict_test(model: RandomForestClassifier, X_test: pd.DataFrame)->pd.Series:
    return model.predict(X_test)

# check accuracy
def check_acc(y_pred: pd.Series, y_true: pd.Series)->float:
    return accuracy_score(y_true, y_pred)

# workflow
def run_wf(hp:Hyperparameters)->RandomForestClassifier:
    df = create_df(hp=hp)
    X_train, X_test, y_train, y_test = split_ds(hp=hp, df=df)
    model = train_model(hp=hp, X_train=X_train, y_train=y_train)
    y_pred = predict_test(model=model, X_test=X_test)
    print(check_acc(y_pred=y_pred, y_true=y_test))
    return model

if __name__=="__main__":
    run_wf(hp=hp)