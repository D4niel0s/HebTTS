from datasets import load_from_disk, concatenate_datasets
from datasets import Dataset
import numpy as np, os, torch, torchaudio

from tqdm import tqdm
from wav2vec2_hebrew import HebrewWav2Vec2Aligner


data1 = load_from_disk('/home/yandex/APDL2425a/group_6/hf_cache/all_dor_w_embed')
data2 = load_from_disk('/home/yandex/APDL2425a/group_6/hf_cache/all_GK_w_embed')
data3 = load_from_disk('/home/yandex/APDL2425a/group_6/hf_cache/all_GQ_w_embed')
data4 = load_from_disk('/home/yandex/APDL2425a/group_6/hf_cache/all_OH_w_embed')
data5 = load_from_disk('/home/yandex/APDL2425a/group_6/hf_cache/all_yo_w_embed')

data = concatenate_datasets([data1,data2,data3,data4,data5])


total_examples = len(data)

start_index = int(total_examples * 0)
end_index = int(total_examples * 0.25)


data = data.select(range(start_index, end_index))


print(data)


aligner = HebrewWav2Vec2Aligner(input_sample_rate=16000, use_cuda=True)


output_dir = '/home/yandex/APDL2425a/group_6/hf_cache/SAMPLE_WAVS_2'
os.makedirs(output_dir, exist_ok=True)


def process_long_audio(example, idx):
    sample_rate = 16000
    min_samples = 6 * sample_rate
    split_point = 3 * sample_rate

    n_samples = example['n_samples']
    audio = example['audio']['array']
    text = example['normalized_text']

    if n_samples < min_samples:
        return []  # Skip this row

    speaker_base = f"speaker_{idx}"
    
    
    wav_total = torch.tensor(audio)
    wav_total = (wav_total * 32767).short()
    if wav_total.ndim == 1: wav_total = wav_total.unsqueeze(0)
    fpath = os.path.join('.', f'TRASH_SWAP_1.wav')
    torchaudio.save(fpath, wav_total, sample_rate)

    res = aligner.align_data(fpath, text)[0]


    text1 = [seg.label for seg in res['segments'] if seg.start < split_point]
    text2 = [seg.label for seg in res['segments'] if seg.start >= split_point]

    text1 = ' '.join(text1)
    text2 = ' '.join(text2)


    wav1 = torch.tensor(audio[:split_point])
    wav2 = torch.tensor(audio[split_point:])

    wav1 = (wav1 * 32767).short()
    wav2 = (wav2 * 32767).short()

    if wav1.ndim == 1:
        wav1 = wav1.unsqueeze(0)
    if wav2.ndim == 1:
        wav2 = wav2.unsqueeze(0)


    fpath1 = os.path.join(output_dir, f'sample_{idx}_1.wav')
    fpath2 = os.path.join(output_dir, f'sample_{idx}_2.wav')

    torchaudio.save(fpath1, wav1, sample_rate)
    torchaudio.save(fpath2, wav2, sample_rate)


    part1 = {
        'path': fpath1,
        'duration': 3,
        'transcript': text1,
        'speaker': speaker_base
    }

    part2 = {
        'path': fpath2,
        'duration': (float(n_samples)/sample_rate) - 3,
        'transcript': text2,
        'speaker': speaker_base
    }

    return [part1, part2]


new_rows = []
for idx, example in enumerate(data):
    new_rows.extend(process_long_audio(example, idx+start_index))

    if len(new_rows) >= 312697:
        break

new_dataset = Dataset.from_list(new_rows)


new_dataset.save_to_disk('/home/yandex/APDL2425a/group_6/hf_cache/HebDB_w_enrollments_1')


