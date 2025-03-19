from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet


def load_daniels_shitty_csv(path):
    df = pd.read_csv(path)
    return df


def main():
    df = load_daniels_shitty_csv('fake_data.csv')
    print(df)
    exit()
    sample_rate = 16000

    recordings = [Recording(
            id = row['id'],
            sources = [{"type": "file", "source": row['path'], "channels": [0]}],
            sampling_rate = sample_rate,
            num_samples = row['duration'] * sample_rate,   # e.g., 3 seconds * 16000 samples/sec
            duration = row['duration'],
        ) for _, row in df.iterrows()]

    # Create a supervision segment for the utterance.


    recording_set = RecordingSet.from_recordings(recordings)

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