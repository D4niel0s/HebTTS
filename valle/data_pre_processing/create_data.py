import torch, torchaudio, librosa, os, whisper, numpy as np, sys

from datasets import load_dataset
from silero_vad import load_silero_vad, get_speech_timestamps

MAIN_DIR = './processed_dataset'

podcast_name = sys.argv[1] + '_raw'

if podcast_name not in ['DK_raw', 'GK_raw', 'OH_raw', 'YO_raw', 'GQ_raw']:
    print('Illegal podcast name')
    raise Exception


data = load_dataset("SLPRL-HUJI/HebDB", podcast_name, streaming=True)
torch_train = data['train'].with_format("torch").select_columns('audio')


whisper_model = whisper.load_model("large-v2", download_root='/home/yandex/APDL2425a/group_6/')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
whisper_model.to(device)

vad_model = load_silero_vad()


skip_map = {
    'DK_raw' : 54,
    'GK_raw' : 10,
    'YO_raw' : 46,
    'OH_raw' : 5
}

torch_train = torch_train.skip(skip_map[podcast_name])
num_episodes = skip_map[podcast_name]

for episode in torch_train:

    # Grab audio and original sr
    audio = episode['audio']['array']
    sr = episode['audio']['sampling_rate'].item()


    # Resample to 16KHz
    res_librosa = librosa.resample(audio.detach().numpy(), orig_sr=sr, target_sr=16000)


    # Perform vad
    speech_timestamps = get_speech_timestamps(
    res_librosa,
    vad_model,
    return_seconds=True,
    )


    #Filter out irrelevant (too short) segments, and merge segments with too little silence gap in-between
    MIN_DURATION = 2  # seconds
    SILENCE_GAP = 0.1   # 100 ms
    # TODO: Remove filtering by duration. Already done in trainer.py. They chose 0.6-20 seconds.

    sr = 16000

    valid_segments = []
    for seg in speech_timestamps:
        duration = seg['end'] - seg['start']

        if duration >= MIN_DURATION:
            valid_segments.append(seg)


    merged_segments = []
    for seg in valid_segments:
        if not merged_segments:
            merged_segments.append(seg)
        else:
            prev = merged_segments[-1]
            # If current segment starts within the SILENCE_GAP from the previous segment’s end, merge them
            if seg['start'] - prev['end'] < SILENCE_GAP:
                merged_segments[-1]['end'] = seg['end']
            else:
                merged_segments.append(seg)




    # Grab enrollment (4 prev seconds) for each valid segment, transcribe, and save files
    MIN_ACTIVITY_FOR_ENROLL = 3
    PAD = 0.03

    pad_samples = int(PAD * sr)  # Number of samples for 30ms
    silence = torch.zeros(pad_samples)  # Shape: (1, pad_samples) for mono

    output_dir = MAIN_DIR + f"/{podcast_name}_samples/ep_{num_episodes}"
    os.makedirs(output_dir, exist_ok=True)

    num_samples = 0
    for i, seg in enumerate(merged_segments):
        start_sample = int(seg['start'] * sr)
        end_sample   = int(seg['end'] * sr)

        segment_waveform = torch.cat((silence, torch.tensor(res_librosa[start_sample:end_sample]), silence))

        previous_4_sec = res_librosa[start_sample-(sr*4) : start_sample]

        prev_activity = get_speech_timestamps(
            previous_4_sec,
            vad_model,
            return_seconds=True
            )
        speech_time_in_prev_4_sec = np.array([j['end']-j['start'] for j in prev_activity]).sum()

        # We want this segment only if we have an enrollment for it
        if speech_time_in_prev_4_sec >= MIN_ACTIVITY_FOR_ENROLL:


            result = whisper_model.transcribe(segment_waveform, language="he")  # specify Hebrew if needed

            with open('data.txt', 'a+', encoding='utf-8') as output_data:
                output_data.write(output_dir+f'/target_{num_samples}.wav' + '\t' + output_dir+f'/enrollment_{num_samples}.wav' + '\t' + result['text'] + '\n')

            # Save target audio to synthesize, and the enrollment
            if segment_waveform.ndim == 1:
                segment_waveform = segment_waveform.unsqueeze(0)
            torchaudio.save(output_dir+f'/target_{num_samples}.wav', segment_waveform, sr)

            # Adding a bit of noise to the enrollment length, to introduce wider dist.
            jitter1 = np.random.choice(int(sr * 0.5))
            jitter2 = np.random.choice(int(sr * 0.5))
            enroll = torch.cat((silence, torch.tensor(previous_4_sec[jitter1: (4*sr) - jitter2]), silence))

            if enroll.ndim == 1:
                enroll = enroll.unsqueeze(0)
            torchaudio.save(output_dir+f'/enrollment_{num_samples}.wav', enroll, sr)

            
            num_samples += 1
        else:
            continue

    print(f'Finished {podcast_name} episode No.{num_episodes}')
    num_episodes +=1
