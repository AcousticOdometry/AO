from pathlib import Path

from typing import Dict
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

auth = GoogleAuth(
    settings_file=str(Path(__file__).parent / 'pydrive_settings.yaml')
    )
auth.CommandLineAuth()
drive = GoogleDrive(auth)


DATASETS_FOLDER_ID = '1-OziUWGnU0Xzd2XB-TOK5wDB3XgQ9I86'

def dataset_shards(dataset: str) -> Dict[str, dict]:
    raise NotImplementedError

def upload_model(model_file: Path, models_folder_id: str):
    raise NotImplementedError

# TODO pydrive path from url
# https://drive.google.com/uc?id=10y3EVT-pfO_OH2DQ__hbBfUweyLUXKh3&amp;confirm=t&amp;uuid=045efc41-ec17-4a22-9e4f-3ad78b35bd40
# shard_url = 'pipe:curl -s -L "https://drive.google.com/uc?confirm=t&id=10y3EVT-pfO_OH2DQ__hbBfUweyLUXKh3"'
# dataset = wds.DataPipeline(
#     wds.SimpleShardList([shard_url]),
#     wds.tarfile_to_samples(),
#     wds.decode(wds.handle_extension('.npy', lambda x: np.load(io.BytesIO(x)))),
#     wds.to_tuple('npy', 'json'),
#     wds.shuffle(0),
#     )
# _, axs = plt.subplots(1, 6, squeeze=False, sharex=True, sharey=True)
# iter_dataset = iter(dataset)
# for col in range(6):
#     features, result = next(iter_dataset)
#     for row in range(1):
#         axs[row][col].pcolormesh(features[row, :, :], shading='flat')
#         axs[row][col].set_title(
#             f"Vx={result['Vx']:0.3f} m/s\n"
#             f"Vw={result['Vw']:0.1f} deg/s\n"
#             f"slip={result['slip']:0.4f}"
#             )

if __name__ == "__main__":
    folder_id = DATASETS_FOLDER_ID
    file_list = drive.ListFile({
        'q': f"'{folder_id}' in parents and trashed=false",
        'supportsAllDrives': True,
        'includeItemsFromAllDrives': True,
        }).GetList()
    for f in file_list:
        print(f['title'])