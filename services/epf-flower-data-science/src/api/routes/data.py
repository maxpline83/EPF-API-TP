from fastapi import APIRouter
from src.services.data import download_dataset, get_data, preprocess_data, train_test_split_data, train_model, predict_model, create_firestore, get_data_firestore, update_data_firestore
from src.schemas.message import MessageResponse
import kaggle

router = APIRouter()

@router.get("/download_data")
def download_dataset_router():
    return download_dataset()

@router.get("/data")
def get_data_router():
    return get_data()

@router.get("/preprocess_data")
def preprocess_data_router():
    return preprocess_data()

@router.get("/train_test_split_data")
def train_test_split_data_router():
    return train_test_split_data()

@router.get("/train_model")
def train_model_router():
    return train_model()[0]

@router.get("/predict_model")
def predict_model_router():
    return predict_model()

@router.get("/create_firestore")
def create_firestore_router():
    return create_firestore()

@router.get("/get_data_firestore")
def get_data_firestore_router():
    return get_data_firestore()

@router.put("/update_data_firestore/")
def update_data_firestore_router():
    return update_data_firestore()