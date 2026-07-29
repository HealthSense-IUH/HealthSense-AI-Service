"""Service kết nối AWS S3 / MinIO để tải file dữ liệu sức khỏe."""

import io
import logging
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from app.config import settings

logger = logging.getLogger("uvicorn.error")


class S3Client:
    def __init__(self):
        self.bucket = settings.AWS_S3_BUCKET
        self.region = settings.AWS_S3_REGION
        self.endpoint_url = settings.AWS_S3_ENDPOINT_URL if settings.AWS_S3_ENDPOINT_URL else None
        
        config = Config(
            region_name=self.region,
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
        )
        
        client_kwargs = {
            "service_name": "s3",
            "region_name": self.region,
            "aws_access_key_id": settings.AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
            "config": config,
        }
        if self.endpoint_url:
            client_kwargs["endpoint_url"] = self.endpoint_url
            
        self.client = boto3.client(**client_kwargs)

    def download_file_as_bytes(self, s3_key: str) -> bytes:
        """Tải file từ bucket S3 trả về dưới dạng bytes."""
        try:
            logger.info(f"Downloading file from S3 bucket '{self.bucket}' with key '{s3_key}'...")
            response = self.client.get_object(Bucket=self.bucket, Key=s3_key)
            content = response["Body"].read()
            logger.info(f"Downloaded {len(content)} bytes successfully for key '{s3_key}'.")
            return content
        except ClientError as e:
            logger.error(f"S3 ClientError downloading file '{s3_key}': {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error downloading file '{s3_key}' from S3: {str(e)}")
            raise


s3_client = S3Client()
