import sklearn
import os
import typing
from flytekit import task, workflow, Resources
import sidetrek
from project.wf_81_451.main import Hyperparameters
from project.wf_81_451.main import create_df
from project.wf_81_451.main import split_ds
from project.wf_81_451.main import train_model

@task(requests=Resources(cpu="2",mem="1Gi"),limits=Resources(cpu="2",mem="1Gi"),retries=3)
def dataset_test_org_quickstart_classification_dataset()->sidetrek.types.dataset.SidetrekDataset:
	return sidetrek.dataset.build_dataset(io="upload",source="s3://sidetrek-datasets/test-org/quickstart-classification-dataset")



_wf_outputs=typing.NamedTuple("WfOutputs",train_model_0=sklearn.ensemble._forest.RandomForestClassifier)
@workflow
def wf_81(_wf_args:Hyperparameters)->_wf_outputs:
	dataset_test_org_quickstart_classification_dataset_o0_=dataset_test_org_quickstart_classification_dataset()
	create_df_o0_=create_df(ds=dataset_test_org_quickstart_classification_dataset_o0_)
	split_ds_o0_,split_ds_o1_,split_ds_o2_,split_ds_o3_=split_ds(hp=_wf_args,df=create_df_o0_)
	train_model_o0_=train_model(hp=_wf_args,X_train=split_ds_o0_,y_train=split_ds_o2_)
	return _wf_outputs(train_model_o0_)