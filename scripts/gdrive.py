import re
import yaml

from tqdm import tqdm
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

    def get_by_name(self, parent_id: str, name: str) -> list:
        return self.drive.ListFile({
            'q': (
                f"'{parent_id}' in parents and trashed=false "
                f"and name='{name}'"
                ),
            'supportsAllDrives': True,
            'includeItemsFromAllDrives': True,
            }).GetList()[0]

    @staticmethod
    def is_folder(f: 'pydrive2.file.GoogleDriveFile'):
        return f['mimeType'] == 'application/vnd.google-apps.folder'

    @staticmethod
    def yaml_load(f: 'pydrive2.file.GoogleDriveFile') -> dict:
        return yaml.safe_load(f.GetContentString())

    @staticmethod
    def download_file(f: 'pydrive2.file.GoogleDriveFile', to: Path) -> None:
        pbar = tqdm(
            desc=f['title'],
            total=100.0,
            unit='%',
            bar_format='{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}]'
            )

        def callback(current, total):
            pbar.n = current / total * 100.0
            pbar.refresh()

        out = f.GetContentFile(str(to), callback=callback)
        pbar.close()
        return out

    def create_folder(
            self, name: str, parent_id: str
        ) -> 'pydrive2.file.GoogleDriveFile':
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
        ) -> 'pydrive2.file.GoogleDriveFile':
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
