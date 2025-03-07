import asyncio
import time
from typing import Optional, Callable
from google.oauth2 import service_account
from google.cloud import compute_v1
from app.core.config import settings
from app.logger import setup_logger

logger = setup_logger(__name__)


class GcloudService:

    def __init__(
        self,
        project_id: str = settings.GCLOUD_AVATAR_INSTANCE_PROJECT,
        zone: str = settings.GCLOUD_AVATAR_INSTANCE_ZONE,
        credentials_path: str = "gcloud-compute-engine-credentials.json",
    ):

        self.project_id = project_id
        self.zone = zone
        self.credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )

    def stop_instance(self, instance_name: str = settings.GCLOUD_AVATAR_INSTANCE_NAME):
        """
        Stops a Compute Engine instance using explicit credentials.

        :param project_id: Google Cloud project ID.
        :param zone: The zone where the instance is located (e.g., 'us-central1-a').
        :param instance_name: The name of the instance to stop.
        """
        instance_client = compute_v1.InstancesClient(credentials=self.credentials)

        operation = instance_client.stop(
            project=self.project_id, zone=self.zone, instance=instance_name
        )

        print(f"Stopping instance {instance_name}...")
        operation.result()  # Waits for the operation to complete
        print(f"Instance {instance_name} stopped successfully.")
