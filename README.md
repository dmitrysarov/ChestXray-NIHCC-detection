# Instalation

1. Clone the repository
2. Run the following command in the terminal `bash install_requirements.sh`
3. For MLFLOW setup on aws instance do. 
    ```bash 
    sudo apt-get update && \
    sudo apt-get install -y python3-pip  python3-venv  unzip && \
    python3 -m venv venv && \
    source venv/bin/activate && \
    pip install mlflow psycopg2-binary boto3 setuptools && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    sudo ./aws/install && \
    mlflow server -h 0.0.0.0 --default-artifact-root s3://<your-s3-bucket>
    ``` 

# Train

1. BBox annotation data is in `data/` folder.
2. Adjust configs
3. To train 
   - on kaggle/colab you can use `EDA/nih-chestxray_kaggle.ipyb`
   - onpremiss `!cd ChestXrayCC-detection && MLFLOW_EXPERIMENT_NAME=chestxray_notebook AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID> AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY> bash scripts/dist_train.sh configs/yolox/yolox_tiny_8xb8-300e_coco_notebook.py 2 --run-name coco_pretrained`