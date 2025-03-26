from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet, AudioSource, CutSet
import pandas as pd

def load_daniels_shitty_csv(path):
    df = pd.read_csv(path)
    return df


def main():
    df = load_daniels_shitty_csv('./fake_data.csv')
    sample_rate = 16000

    recording_set = RecordingSet.from_recordings(Recording(
            id = str(row['id']),  # Convert id to string to be safe
            sources = [AudioSource(type = "file", channels = [0], source = row['path'])], # Use channels instead of channel_ids
            sampling_rate = sample_rate,
            num_samples = int(row['duration'] * sample_rate),   # e.g., 3 seconds * 16000 samples/sec
            duration = row['duration'],
        ) for _, row in df.iterrows())

    recording_set.to_file("recordings.jsonl")
    # Create a SupervisionSet and save it.
    supervisions = [SupervisionSegment(
        id = row['id'],
        recording_id = row['id'],  # must match the recording id above
        channel=0,
        start=0.0,
        duration=row['duration'],
        text= row['transcript'],
        speaker= row['speaker']  # optional field
    ) for _, row in df.iterrows()]
    supervision_set = SupervisionSet.from_segments(supervisions)
    supervision_set.to_file("supervisions.jsonl")
    
    cutSet = CutSet.from_manifests(
        recordings=recording_set,
        supervisions=supervision_set,
    )
    
    cutSet.to_file("cuts.jsonl.gz")
    
if __name__ == "__main__":
    main()