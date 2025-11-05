"""Cloud utilities for GCS operations."""

import os
from datetime import timedelta
from typing import Optional
from google.cloud import storage
from google.oauth2 import service_account
from ganglia_common.logger import Logger

def upload_to_gcs(local_file_path: str, bucket_name: str, project_name: str, destination_blob_name: Optional[str] = None) -> bool:
    """Upload a file to Google Cloud Storage.

    Args:
        local_file_path: Path to the local file to upload
        bucket_name: Name of the GCS bucket
        project_name: GCP project name
        destination_blob_name: Optional name for the file in GCS. If not provided,
                             uses the base name of the local file

    Returns:
        bool: True if upload was successful, False otherwise
    """
    try:
        service_account_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not service_account_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")

        credentials = service_account.Credentials.from_service_account_file(service_account_path)
        storage_client = storage.Client(credentials=credentials, project=project_name)

        if not destination_blob_name:
            destination_blob_name = os.path.basename(local_file_path)

        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file_path)
        return True
    except Exception as error:
        Logger.print_error(f"Error uploading file to cloud: {error}")
        return False

def get_video_stream_url(blob: storage.Blob, expiration_minutes: int = 60, service_account_path: str = None) -> str:
    """Generate a signed URL for streaming a video from GCS.

    Args:
        blob: The GCS blob containing the video
        expiration_minutes: How long the URL should be valid for, in minutes
        service_account_path: Optional path to service account key file. If not provided,
                            will try to use GOOGLE_APPLICATION_CREDENTIALS environment variable

    Returns:
        str: A signed URL that can be used to stream the video

    Example:
        ```python
        uploaded_file = validate_gcs_upload(bucket_name, project_name)
        stream_url = get_video_stream_url(
            uploaded_file,
            service_account_path="path/to/service-account.json"
        )
        print(f"Stream video at: {stream_url}")
        ```

    Raises:
        ValueError: If no valid service account credentials are found
    """
    print("\n=== Generating Video Stream URL ===")

    # If service account path provided, use it to create new client
    if service_account_path:
        if not os.path.exists(service_account_path):
            raise ValueError(f"Service account file not found at: {service_account_path}")

        credentials = service_account.Credentials.from_service_account_file(
            service_account_path
        )
        storage_client = storage.Client(
            credentials=credentials,
            project=blob.bucket.client.project
        )
        # Get a new blob instance with the service account client
        bucket = storage_client.get_bucket(blob.bucket.name)
        blob = bucket.get_blob(blob.name)

    # Generate signed URL with content-type header for video streaming
    try:
        url = blob.generate_signed_url(
            expiration=timedelta(minutes=expiration_minutes),
            method='GET',
            response_type='video/mp4',  # Ensure proper content-type for video streaming
            version='v4'  # Use latest version of signing
        )
        print(f"✓ Generated stream URL (valid for {expiration_minutes} minutes)")
        return url
    except Exception as e:
        print("\nError generating signed URL. Make sure you have:")
        print("1. Set GOOGLE_APPLICATION_CREDENTIALS environment variable to point to your service account key file")
        print("   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json")
        print("2. OR provided the service_account_path parameter")
        print("3. The service account has Storage Object Viewer permissions")
        raise ValueError("Failed to generate signed URL. See above for troubleshooting steps.") from e
