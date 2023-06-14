import pandas as pd
import typing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dataclasses_json import dataclass_json
from dataclasses import dataclass
from sidetrek.dataset import load_dataset
from sidetrek.types.dataset import SidetrekDataset
 
@dataclass_json
@dataclass
class Hyperparameters(object):
    test_size:float = 0.2
    random_state: int = 6
    n_estimators: int=200 
    max_depth:int= 10
    min_samples_leaf:int= 2
    max_leaf_nodes:int= 5
    n_jobs:int= 15

hp = Hyperparameters()


# create dataframe
def create_df(ds: SidetrekDataset)->pd.DataFrame:
    data = load_dataset(ds=ds, data_type="csv")
    cols = list(data)[0]
    data_dict = {}
    for k,v in enumerate(data):
        if k>0:
            data_dict[k]=v
    df = pd.DataFrame.from_dict(data=data_dict, columns=cols, orient="index")
    return df

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