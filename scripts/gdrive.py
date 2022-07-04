import re
import yaml

from pathlib import Path
from typing import Optional
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


class GDrive:

    def __init__(
        self,
        settings_file: Path = Path(__file__).parent / 'pydrive_settings.yaml',
        ) -> None:
        self.auth = GoogleAuth(settings_file=settings_file)
        self.auth.CommandLineAuth()
        self.drive = GoogleDrive(self.auth)

    def list_folder(self, folder_id: str) -> list:
        return self.drive.ListFile({
            'q': f"'{folder_id}' in parents and trashed=false",
            'supportsAllDrives': True,
            'includeItemsFromAllDrives': True,
            }).GetList()

    @staticmethod
    def is_folder(f: 'pydrive.file.GoogleDriveFile'):
        return f['mimeType'] == 'application/vnd.google-apps.folder'

    def yaml_load(self, f: 'pydrive.file.GoogleDriveFile') -> dict:
        return yaml.safe_load(f.GetContentString())

    def create_folder(
            self, name: str, parent_id: str
        ) -> 'pydrive.file.GoogleDriveFile':
        return self.drive.CreateFile({
            'title': name,
            'supportsAllDrives': True,
            'includeItemsFromAllDrives': True,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [{
                "id": parent_id
                }]
            })

    def create_file(
            self, name: str, parent_id: str
        ) -> 'pydrive.file.GoogleDriveFile':
        return self.drive.CreateFile({
            'title': name,
            'supportsAllDrives': True,
            'includeItemsFromAllDrives': True,
            'parents': [{
                "id": parent_id
                }],
            })

    @staticmethod
    def get_folder_id(url: str) -> Optional[str]:
        match = re.match(
            r"https:\/\/drive\.google\.com\/drive\/folders\/(?P<folder_id>[^\?]*)",
            url
            )
        if match:
            return match.groupdict().get('folder_id', None)
        return None
