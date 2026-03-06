import os
from pathlib import Path
from pydub import AudioSegment
import shutil
import random

NumberOfSongs = None

def make_folders(base_path):
    """
    Create 100 folders inside a base directory.
    Parameters:
    base_path (str): The base directory where folders will be created.
    """
    try:
        for i in range(1, NumberOfSongs + 1):
            path = os.path.join(base_path, f"{i}")
            os.makedirs(path, exist_ok=True)
            print(f"Created folder: {path}")
    except Exception as e:
        print(f"An error in make_folders: {e}")

def PutInputFilesToFoldersAndRename(input_dir, Piano_Dir, Non_Piano_Dir):
    """
    inputing Piano And non piano files to the same folder we created and renaming non piano files to numbers and piano to Piano.
    """
    try:
        i = 1
        for d in os.listdir(Piano_Dir):
            SoundPath = os.path.join(Piano_Dir, d)
            dst_folder = os.path.join(input_dir, f"{i}")
            dst_sound = os.path.join(dst_folder, f"Piano.wav")
            shutil.copyfile(SoundPath, dst_sound)
            print(f"Copied {SoundPath} to {dst_sound}")
            
            i += 1
            if i > NumberOfSongs:
                print(f"An error occurred in PutInputFilesToFoldersAndRename: Exceeded folder limit. in Piano files")
                break

        i = 1

        for d in os.listdir(Non_Piano_Dir):
            SoundPath = os.path.join(Non_Piano_Dir, d)
            dst_folder = os.path.join(input_dir, f"{i}")
            dst_sound = os.path.join(dst_folder, f"{i}.wav")
            shutil.copyfile(SoundPath, dst_sound)
            print(f"Copied {SoundPath} to {dst_sound}")
            
            i += 1
            if i > NumberOfSongs:
                print(f"An error occurred in PutInputFilesToFoldersAndRename: Exceeded folder limit. in Non Piano files")
                break

    except Exception as e:
        print(f"An error occurred in PutInputFilesToFoldersAndRename: {e}")

def mixAudioFilesAndTrimPianoAndDeleteSong(input_dir):
    """
    Mix two audio files using pydub.
    """
    # load two audio files
    try:
        for i in range(1, NumberOfSongs + 1):
            folder_path = os.path.join(input_dir, f"{i}")
            piano_path = os.path.join(folder_path, "Piano.wav")
            NonPiano_path = os.path.join(input_dir, f"{i}", f"{i}.wav")
            mix_path = os.path.join(folder_path, "mixture.wav")

            sound1 = AudioSegment.from_wav(piano_path)
            sound2 = AudioSegment.from_wav(NonPiano_path)

            target_len = min(len(sound1), len(sound2))

            piano_trimmed  = sound1[:target_len]
            song_trimmed   = sound2[:target_len]

            # mix sound2 with sound1
            output = piano_trimmed.overlay(song_trimmed)

            # save the result
            output.export(mix_path, format="wav")

            if len(sound1) > target_len:
                piano_trimmed.export(piano_path, format="wav")

            if os.path.exists(NonPiano_path):
                os.remove(NonPiano_path)
                print(f"[{i}] mixture created, song deleted")

    except Exception as e:
        print(f"An error occurred in mixAudioFilesAndTrimPiano: {e}")

def GenerateTrainValidSplit(INPUT_DIR, TRAIN_RATIO=0.8, OUTPUT_DIR="dataset"):

    try:
        random.seed(42)  # reproducible split

        tracks = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
        random.shuffle(tracks)

        split_idx = int(len(tracks) * TRAIN_RATIO)
        train_tracks = tracks[:split_idx]
        valid_tracks = tracks[split_idx:]

        for folder, names in [(train_tracks, "train"), (valid_tracks, "valid")]:
            out_path = os.path.join(OUTPUT_DIR, names)
            os.makedirs(out_path, exist_ok=True)
            for t in folder:
                src = os.path.join(INPUT_DIR, t)
                dst = os.path.join(out_path, t)
                shutil.copytree(src, dst)

        print(f"Train: {len(train_tracks)} tracks")
        print(f"Validation: {len(valid_tracks)} tracks")
    except Exception as e:
        print(f"An error occurred in GenerateTrainValidSplit: {e}")

def rename_folders_to_numbers(parent_dir):
    # list only folders
    folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]

    # sort for consistent order
    folders.sort()

    # rename each folder
    for i, folder in enumerate(folders, start=1):
        old_path = os.path.join(parent_dir, folder)
        new_path = os.path.join(parent_dir, str(i))

        # if the target name already exists, skip or handle it
        if os.path.exists(new_path):
            print(f"Skipping {folder} because {i} already exists")
            continue

        os.rename(old_path, new_path)
        print(f"Renamed {folder} -> {i}")


def Main():
    INPUT_DIR = r"input here"  # change to your input directory
    Piano_Dir = r"input here"  # change to your instrument directory
    Non_Piano_Dir = r"input here"  # change to your non-instrument directory
    NumberOfSongs = None  # change to the number of songs you have (should be the same for instrument and song)

    make_folders(INPUT_DIR)
    PutInputFilesToFoldersAndRename(INPUT_DIR, Piano_Dir, Non_Piano_Dir)
    mixAudioFilesAndTrimPianoAndDeleteSong(INPUT_DIR)
    GenerateTrainValidSplit(INPUT_DIR, TRAIN_RATIO=0.8, OUTPUT_DIR="dataset")
    rename_folders_to_numbers(r"input here")  # change to your valid directory path
    rename_folders_to_numbers(r"input here")  # change to your valid directory path



if __name__ == "__main__":
    Main()