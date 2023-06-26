from google.cloud import storage
import joblib
import pandas as pd
import pandas_gbq
from surprise import (
    Reader,
    Dataset,
    SVD
)
from surprise.model_selection import (
    train_test_split,
    GridSearchCV
)


PROJECT_ID: str = "proyectofinal-389001"
pandas_gbq.context.project = PROJECT_ID
pandas_gbq.context.dialect = "standard"

def train() -> None:
    query_users: str = f"""--sql
    SELECT
        user_id,
        business_id,
        rating
    FROM `{PROJECT_ID}.Google.Reviews`;
    """

    users_df = pd.read_gbq(
        query=query_users,
        location="us"
    )

    users_df = users_df.sample(frac=.2)
    res = users_df[users_df["rating"] >= 3]

    reader = Reader(line_format="user item rating", rating_scale=(1, 5))
    data = Dataset.load_from_df(res, reader)
    train_set, test_set = train_test_split(data, test_size=.2)

    # Dict of params with different values for number of latent factors,
    # number of epochs, learning rate and regularization
    params = {
        "n_factors": [5, 50, 100],
        "n_epochs": [5, 10, 20],
        "lr_all": [0.001, 0.002, 0.005],
        "reg_all": [0.002, 0.02, 0.2]
    }
    # Find best model based on this params
    gs = GridSearchCV(SVD, params, measures=["RMSE", "MAE"], cv=3, n_jobs=-1)
    gs.fit(data)

    # With params found, instantiate best_model
    best_model = SVD(n_factors=5, n_epochs=5, lr_all=0.005, reg_all=0.002)
    best_model = gs.best_estimator["rmse"]
    best_model.fit(train_set)
    predictions = best_model.test(test_set)

    with open("model.joblib", "wb") as jobfile_dump:
        joblib.dump(best_model, jobfile_dump, protocol=5)
    
    with open("model.joblib", "rb") as jobfile_load:
        upload_bucket(
            bucket_name="ml-models",
            source_file_object=jobfile_load,
            destination_blob_name="datawise-consulting/model.joblib"
        )


def upload_bucket(bucket_name: str,
                  source_file_object: str,
                  destination_blob_name: str
                  ) -> None:
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name=bucket_name)
    blob = bucket.blob(blob_name=destination_blob_name)
    
    blob.upload_from_file(file_obj=source_file_object)


if __name__ == "__main__":
    print("main")
    train()