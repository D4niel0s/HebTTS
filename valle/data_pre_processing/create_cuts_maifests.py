from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet, AudioSource, CutSet
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--src-path",
        type=str,
        default='fake_data.csv',
        help="Path to source csv.",
    )

    parser.add_argument(
        "--dest-path",
        type=str,
        default='.',
        help="Path to save the manifests.",
    )

    return parser.parse_args()


def load_daniels_shitty_csv(path):
    df = pd.read_csv(path, sep='|')
    print(df.head())
    return df


def main():
    args = get_args()
    
    df = load_daniels_shitty_csv(args.src_path)
    sample_rate = 16000
    prefix = "libritts"
    
    recordings = [Recording(
            id = str(row['id']),  # Convert id to string to be safe
            sources = [AudioSource(type = "file", channels = [0], source = row['path'])], # Use channels instead of channel_ids
            sampling_rate = sample_rate,
            num_samples = int(row['duration'] * sample_rate),   # e.g., 3 seconds * 16000 samples/sec
            duration = row['duration'],
        ) for _, row in df.iterrows()]
    
    supervisions = [SupervisionSegment(
        id = row['id'],
        recording_id = row['id'],  # must match the recording id above
        channel=0,
        start=0.001,
        duration=row['duration'] - 0.001,
        text= row['transcript'],
        speaker= row['speaker'] 
    ) for _, row in df.iterrows()]

    
    random_state = np.random.RandomState(42)
    
    total_size = len(recordings)
    train_size = int(0.8 * total_size)
    dev_size = int(0.1 * total_size)
    test_size = total_size - train_size - dev_size
    indices = np.arange(total_size)

    train_indices, temp_indices = train_test_split(
        indices, 
        train_size=train_size,
        test_size=dev_size + test_size,
        random_state=random_state
    )

    dev_indices = temp_indices[test_size:]
    test_indices = temp_indices[:test_size]
    
    part_indices = [
        (train_indices, "train"),
        (dev_indices, "dev"),
        (test_indices, "test"),
    ]
    
    for indices, part in part_indices:
        part_recordings = np.asarray(recordings, dtype = 'object')[indices]
        part_recording_set = RecordingSet.from_recordings(part_recordings)
        
        part_supervisions = np.asarray(supervisions, dtype = 'object')[indices]
        part_supervision_set = SupervisionSet.from_segments(part_supervisions)
        
        part_cut_set = CutSet.from_manifests(
            recordings=part_recording_set,
            supervisions=part_supervision_set,
        )
        
        part_cut_set.to_file(f"{args.dest_path}/{prefix}_cuts_{part}.jsonl.gz")
        part_recording_set.to_file(f"{args.dest_path}/{prefix}_recordings_{part}.jsonl.gz")
        part_supervision_set.to_file(f"{args.dest_path}/{prefix}_supervisions_{part}.jsonl.gz")
    
if __name__ == "__main__":
    main()