import torch, torchaudio, librosa, os, whisper, numpy as np, sys

from datasets import load_dataset
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

verbose = False
if sys.argv[1] == 'verbose':
    verbose = True


data = load_dataset("SLPRL-HUJI/HebDB", "DK_raw", streaming=True)
torch_train = data['train'].with_format("torch")

if verbose: print('Loaded data')

# This is a fucked up way to grab the first element in torch_train. torch_train[0] is an error
for i in torch_train:
    res = i
    break

# Grab audio and original sr
audio = res['audio']['array']
sr = res['audio']['sampling_rate'].item()

if verbose: print(f'Got first sample, {sr=}')

# Resample to 16KHz
res_librosa = librosa.resample(audio.detach().numpy(), orig_sr=sr, target_sr=16000)


# Perform vad
vad_model = load_silero_vad()

speech_timestamps = get_speech_timestamps(
  res_librosa,
  vad_model,
  return_seconds=True,
)

if verbose: print(f'Did VAD using silero')

#Filter out irrelevant (too short) segments, and merge segments with too little silence gap in-between
MIN_DURATION = 1.5  # seconds
SILENCE_GAP = 0.1   # 100 ms

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



if verbose: print(f'Got merged segments')


# Grab enrollment (4 prev seconds) for each valid segment, transcribe, and save files
MIN_ACTIVITY_FOR_ENROLL = 3
PAD = 0.03

pad_samples = int(PAD * sr)  # Number of samples for 30ms
silence = torch.zeros(pad_samples)  # Shape: (1, pad_samples) for mono

output_dir = "DK_samples_1"
os.makedirs(output_dir, exist_ok=True)


whisper_model = whisper.load_model("large-v2")
vad_model = load_silero_vad()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
whisper_model.to(device)

if verbose: print(f'Loaded whisper, {device=}')

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

        if verbose: print(f'Found relevant sample')

        result = whisper_model.transcribe(segment_waveform, language="he")  # specify Hebrew if needed

        with open('data.txt', 'a') as output_data:
            output_data.write(output_dir+f'/target_{num_samples}.wav'+'\t<SEP>\t'+result['text']+'\t<SEP>\t'+output_dir+f'/enrollment_{num_samples}.wav'+'\n')


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

        if verbose: print(f'Added sample No.{num_samples}')
        
        num_samples += 1
    else:
        continue