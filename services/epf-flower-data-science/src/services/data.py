import kaggle
from src.schemas.message import MessageResponse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

def download_dataset() -> MessageResponse:
    """Download the dataset from Kaggle."""
    kaggle.api.authenticate()
    dataset_name = "uciml/iris"
    save_dir = "services\epf-flower-data-science\src\data"
    kaggle.api.dataset_download_files(dataset_name, path=save_dir, unzip=True)
    return MessageResponse(message="Dataset downloaded !")

def get_data():
    """Get the data from the dataset."""
    data = pd.read_csv("services\epf-flower-data-science\src\data\Iris.csv")
    return data.to_json(orient="records")

def preprocess_data():
    """Preprocess the data."""
    data = pd.read_json(get_data())
    data = data.drop("Id", axis=1)
    data['Species'] = data['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
    data = data.rename(columns={"Species": "target"})
    data.to_csv("services\epf-flower-data-science\src\data\preprocessed_data.csv", index=False)
    return data.to_json(orient="records")

def train_test_split_data():
    """Split the data into train and test sets."""
    data_preprocessed = pd.read_json(preprocess_data())
    data_train, data_test = train_test_split(data_preprocessed, test_size=0.2)
    return data_train.to_json(orient="records"), data_test.to_json(orient="records")

def train_model():
    """Train the model."""
    data_train = pd.read_json(train_test_split_data()[0])
    X_train = data_train.drop("target", axis=1)
    y_train = data_train["target"]
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    
    return model

def predict_model():
    """Predict the model."""
    model = train_model()
    data_test = pd.read_json(train_test_split_data()[1])
    X_test = data_test.drop("target", axis=1)
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred).to_json(orient="records")
    return y_pred

