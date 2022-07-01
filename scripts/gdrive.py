import re
import yaml

from pathlib import Path
from typing import Optional
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


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

    def yaml_load(self, f: 'pydrive.file.GoogleDriveFile') -> dict:
        return yaml.safe_load(f.GetContentString())

    @staticmethod
    def get_folder_id(url: str) -> Optional[str]:
        match = re.match(
            r"https:\/\/drive\.google\.com\/drive\/folders\/(?P<folder_id>[^\?]*)",
            url
            )
        if match:
            return match.groupdict().get('folder_id', None)
        return None
