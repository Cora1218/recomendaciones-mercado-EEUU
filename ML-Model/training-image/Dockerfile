FROM python:3.11-bullseye
WORKDIR /root

RUN pip3 install --no-cache-dir fsspec==2023.1.0 gcsfs==2023.1.0 google-cloud-storage==2.8.0 numpy==1.22.4 pandas==1.3.5 pandas-gbq==0.19.2 scikit-learn==1.0.2 scikit-surprise==1.1.1

COPY train.py ./train.py

ENTRYPOINT ["python3", "train.py"]
