from gdrive import GDrive

import os

from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv


def upload_model(model_folder: Path, upload_to_id: str):
    if model_folder.name.startswith('version'):
        model_name = model_folder.parent.name
    else:
        model_name = model_folder.name
    gdrive = GDrive()
    # Trash previous model if there was one
    for _folder in gdrive.list_folder(upload_to_id):
        if _folder['title'] == model_name:
            _folder.Trash()
    # Create model folder
    upload_to = gdrive.create_folder(model_name, upload_to_id)
    upload_to.Upload()
    # Upload all files
    for f in tqdm([f for f in model_folder.iterdir() if f.is_file()],
                  desc='Upload model',
                  unit='file'):
        gdrive_file = gdrive.create_file(f.name, upload_to['id'])
        gdrive_file.SetContentFile(str(f))
        gdrive_file.Upload()


def upload_odometry(model_folder: Path, upload_to_id: str):
    if model_folder.name.startswith('version'):
        model_name = model_folder.parent.name
    else:
        model_name = model_folder.name
    gdrive = GDrive()
    # For every evaluation found in log_dir
    evaluations = sorted([
        f for f in Path(model_folder).iterdir() if f.is_dir()
        ])
    for evaluation in tqdm(evaluations, desc='Upload odometry', unit='folder'):
        # Go to evaluation_folder / evaluation.name
        for recording in gdrive.list_folder(upload_to_id):
            if recording['title'] == evaluation.name:
                break
        else:
            raise RuntimeError(f"Recording {evaluation.name} not found")
        # Go to evaluation_folder / subdirectory.name / AO
        for ao_folder in gdrive.list_folder(recording['id']):
            if ao_folder['title'] == 'AO':
                break
        else:
            ao_folder = gdrive.create_folder('AO', recording['id'])
            ao_folder.Upload()
        #  Make logger.name folder, delete existing one if there is
        for _folder in gdrive.list_folder(ao_folder['id']):
            if _folder['title'] == model_name:
                _folder.Trash()
        upload_to = gdrive.create_folder(model_name, ao_folder['id'])
        upload_to.Upload()
        # Upload evaluation contents into model_name folder
        for f in tqdm([f for f in evaluation.iterdir() if f.is_file()],
                      desc=evaluation.name,
                      unit='file'):
            gdrive_file = gdrive.create_file(f.name, upload_to['id'])
            gdrive_file.SetContentFile(str(f))
            gdrive_file.Upload()


if __name__ == '__main__':
    from argparse import ArgumentParser
    load_dotenv()

    parser = ArgumentParser("Upload Acoustic Odometry models to Google Drive")
    parser.add_argument('model', type=str)
    parser.add_argument(
        '-f',
        '--from-folder',
        type=Path,
        default=Path(__file__).parent.parent / 'models'
        )
    parser.add_argument('-v', '--version', type=int, default=None)
    parser.add_argument('--only-odometry', action='store_true')
    parser.add_argument(
        '--to-models-folder', type=str, default=os.getenv('MODELS_FOLDER')
        )
    parser.add_argument(
        '--to-evaluation-folder',
        type=str,
        default=os.getenv('EVALUATION_FOLDER')
        )
    args = parser.parse_args()

    model_folder = Path(args.model)
    if not model_folder.exists():
        model_folder = args.from_folder / args.model
    if not (model_folder / 'model.pt').exists():
        if args.version is None:
            versions = [
                int(d.name.replace('version_', ''))
                for d in model_folder.iterdir()
                if d.is_dir and d.name.startswith('version_')
                ]
            version = max(versions)
        else:
            version = args.version
        model_folder = model_folder / f"version_{version}"
    if not (model_folder / 'model.pt').exists():
        raise ValueError(
            f"Could not find model {args.model}, with --from-folder "
            f"{args.from_folder} and --version {args.version}"
            )

    if not args.only_odometry:
        models_folder_id = GDrive.get_folder_id(args.to_models_folder)
        if not models_folder_id:
            raise ValueError(
                f"Invalid models folder url: {args.to_models_folder}"
                )
        upload_model(model_folder, models_folder_id)
    evaluation_folder_id = GDrive.get_folder_id(args.to_evaluation_folder)
    if not evaluation_folder_id:
        raise ValueError(
            f"Invalid evaluation folder url: {args.to_evaluation_folder}"
            )
    upload_odometry(model_folder, evaluation_folder_id)
