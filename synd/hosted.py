import minio
from synd.models.base import BaseSynDModel
from io import BytesIO

model_host = "minios.jdrusso.dev"
model_bucket = "models"


def make_minio_client(access_key: str, secret_key: str) -> minio.Minio:
    client = minio.Minio(model_host, access_key=access_key, secret_key=secret_key)
    return client


def load_hosted_model(identifier: str, access_key: str, secret_key: str) -> BaseSynDModel:

    client = make_minio_client(access_key, secret_key)
    model = client.get_object(model_bucket, identifier)

    return model


def upload_model(model, identifier: str, access_key: str, secret_key: str):

    client = make_minio_client(access_key, secret_key)

    serialized = model.serialize()

    with BytesIO(serialized) as serialized_bytes:
        client.put_object(model_bucket, identifier, data=serialized_bytes, length=len(serialized))
