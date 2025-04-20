import os
import torch
import torchaudio
from omegaconf import OmegaConf
import argparse
from pathlib import Path

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from utils import AttributeDict


from valle.data import AudioTokenizer, tokenize_audio
from valle.data.collation import get_text_token_collater
from valle.models import get_model
from valle.data.hebrew_root_tokenizer import AlefBERTRootTokenizer, replace_chars


def load_model(checkpoint, device):
    if not checkpoint:
        return None

    checkpoint = torch.load(checkpoint, map_location=device)

    args = AttributeDict(checkpoint)
    model = get_model(args)

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model"], strict=True
    )
    assert not missing_keys
    model.to(device)
    model.eval()

    text_tokens = args.text_tokens_path

    return model, text_tokens


def infer(checkpoint_path, output_dir, texts, prompt_text, prompt_audio, top_k=50, temperature=1, args=None):

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    model, text_tokens = load_model(checkpoint_path, device)
    text_collater = get_text_token_collater(args.tokens_file)

    audio_tokenizer = AudioTokenizer(mbd=args.mbd)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    alef_bert_tokenizer = AlefBERTRootTokenizer(vocab_file=args.vocab_file)
    texts = texts.split("|")

    audio_prompts = list()
    encoded_frames = tokenize_audio(audio_tokenizer, prompt_audio)

    print(f'{encoded_frames[0][0].shape=}')
    audio_prompts.append(encoded_frames[0][0])
    audio_prompts = torch.concat(audio_prompts, dim=-1).transpose(2, 1)
    audio_prompts = audio_prompts.to(device)

    print(f'{len(audio_prompts)=}')
    print(f'{audio_prompts[0].shape=}')
    for n, text in enumerate(texts):
        text_without_space = [replace_chars(f"{prompt_text} {text}").strip().replace(" ", "_")]
        tokens = alef_bert_tokenizer._tokenize(text_without_space)

        prompt_text_without_space = [replace_chars(f"{prompt_text}").strip().replace(" ", "_")]
        prompt_tokens = alef_bert_tokenizer._tokenize(prompt_text_without_space)

        text_tokens, text_tokens_lens = text_collater(
            [
                tokens
            ]
        )
        _, enroll_x_lens = text_collater(
            [
                prompt_tokens
            ]
        )

        # synthesis
        encoded_frames = model.inference(
            text_tokens.to(device),
            text_tokens_lens.to(device),
            audio_prompts,
            enroll_x_lens=enroll_x_lens,
            top_k=top_k,
            temperature=temperature,
        )

        print(encoded_frames.shape)
        audio_path = f"{output_dir}/sample_{n}.wav"

        if args.mbd:
            samples = audio_tokenizer.mbd_decode(
                encoded_frames.transpose(2, 1)
            )
        else:
            samples = audio_tokenizer.decode(
                [(encoded_frames.transpose(2, 1), None)]
            )
        
        torchaudio.save(audio_path, samples[0].detach().cpu(), 24000)





def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--speaker",
        type=str,
        default="geek",
        help="A speaker from speakers.yaml",
    )

    parser.add_argument(
        "--mbd",
        type=bool,
        default=False,
        help="use of multi band diffusion",
    )

    parser.add_argument(
        "--text",
        type=str,
        default="ברוכים הבאים לפודקאסט עושים היסטוריה | אני ממש מקווה שהמודל הזה יכליל הרבה הרבה יותר טוב | ואם אתה משפר את איכות החיזוי אתה יכול להציל חיי אדם",
        help="Text to be synthesized.",
    )

    parser.add_argument(
        "--speaker-yaml",
        type=str,
        default="/home/yandex/APDL2425a/group_6/Documents/HebTTS/speakers/speakers.yaml",
        help="speaker yaml path",
    )

    parser.add_argument(
        "--vocab-file",
        type=str,
        default="/home/yandex/APDL2425a/group_6/Documents/HebTTS/tokenizer/vocab.txt",
        help="vocab file for AlephBert"
    )

    parser.add_argument(
        "--tokens-file",
        type=str,
        default="/home/yandex/APDL2425a/group_6/Documents/unique_chars_tokens.k2symbols",
        help="tokens file path"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/yandex/APDL2425a/group_6/Documents/HebTTS/valle/exp/valle_dev/checkpoint-750000.pt",
        help="Path to the saved checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default="/home/yandex/APDL2425a/group_6/Documents/output_wavs",
        help="Output directory to save the synthesized audio.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="top k sampling",
    )

    parser.add_argument(
        "--temperature",
        type=int,
        default=1,
        help="Temperature for sampling",
    )


    return parser.parse_args()


def jupyter_demo(text, speaker):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    model, text_tokens = load_model(CHECKPOINT_PATH, device)
    text_collater = get_text_token_collater(TOKENS_FILE)

    audio_tokenizer = AudioTokenizer(mbd=True)

    alef_bert_tokenizer = AlefBERTRootTokenizer(vocab_file=VOCAB_PATH)

    speaker_yaml = OmegaConf.load(SPEAKER_PATH)

    try:
        speaker = speaker_yaml[speaker]
    except:
        print(f"Invalid speaker {speaker}. Should be defined at speakers.yaml.")

    audio_prompt = str(Path(SPEAKER_PATH).parent / speaker["audio-prompt"])

    audio_prompts = list()
    encoded_frames = tokenize_audio(audio_tokenizer, audio_prompt)
    audio_prompts.append(encoded_frames[0][0])
    audio_prompts = torch.concat(audio_prompts, dim=-1).transpose(2, 1)
    audio_prompts = audio_prompts.to(device)

    text_without_space = [replace_chars(f"{speaker['text-prompt']} {text}").strip().replace(" ", "_")]
    tokens = alef_bert_tokenizer._tokenize(text_without_space)
    prompt_text_without_space = [replace_chars(f"{speaker['text-prompt']}").strip().replace(" ", "_")]
    prompt_tokens = alef_bert_tokenizer._tokenize(prompt_text_without_space)

    text_tokens, text_tokens_lens = text_collater(
        [
            tokens
        ]
    )
    _, enroll_x_lens = text_collater(
        [
            prompt_tokens
        ]
    )

    # synthesis
    encoded_frames = model.inference(
        text_tokens.to(device),
        text_tokens_lens.to(device),
        audio_prompts,
        enroll_x_lens=enroll_x_lens,
        top_k=50,
        temperature=1,
    )


    samples = audio_tokenizer.mbd_decode(
        encoded_frames
    )

    torchaudio.save("out.wav", samples[0].cpu(), 24000)



if __name__ == '__main__':
    args = get_args()
    speaker_yaml = OmegaConf.load(args.speaker_yaml)

    try:
        speaker = speaker_yaml[args.speaker]
    except:
        print(f"Invalid speaker {args.speaker}. Should be defined at speakers.yaml.")

    if os.path.exists(args.text):
        with open(args.text, 'r') as f:
            text = "|".join(f.readlines())

    else:
        text = args.text

    audio_prompt = str(Path(args.speaker_yaml).parent / speaker["audio-prompt"])

    infer(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        texts=text,
        prompt_text=speaker["text-prompt"],
        prompt_audio=audio_prompt,
        top_k=args.top_k,
        temperature=args.temperature,
        args=args
    )

