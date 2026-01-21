#!/bin/bash
touch .env 
touch .env_example
touch Dockerfile
touch .gitignore
touch params.yaml
touch .dockerignore

mkdir artifacts

mkdir data

mkdir metrics 

mkdir app
cd app
touch main.py
cd ..

cd src 

mkdir data_loading
cd data_loading
touch __init__.py
touch load_data.py
cd .. 

mkdir data_preprocessing
cd data_preprocessing
touch __init__.py
touch preprocess_data.py
cd .. 

mkdir feature_engineering
cd feature_engineering
touch __init__.py
touch engineer_features.py
cd .. 

mkdir model_evaluation
cd model_evaluation
touch __init__.py
touch evaluate_model.py
cd ..


mkdir model_training
touch __init__.py
touch train_model.py
cd ..

touch register_artifacts