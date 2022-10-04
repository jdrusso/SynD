"""Store/retrieve SynD models from an S3 host."""
import minio
from synd.models.base import BaseSynDModel
from io import BytesIO

MODEL_HOST = "minios.jdrusso.dev"
MODEL_BUCKET = "models"


def make_minio_client(access_key: str, secret_key: str, model_host: str = MODEL_HOST, **client_kwargs) -> minio.Minio:
    client = minio.Minio(model_host, access_key=access_key, secret_key=secret_key, **client_kwargs)
    return client


def download_model(identifier: str, client: minio.Minio, bucket: str = MODEL_BUCKET) -> BaseSynDModel:

    model = client.get_object(bucket, identifier)

    return model


def upload_model(model, identifier: str, client: minio.Minio, bucket: str = MODEL_BUCKET):

    serialized = model.serialize()

    with BytesIO(serialized) as serialized_bytes:
        client.put_object(bucket, identifier, data=serialized_bytes, length=len(serialized))
