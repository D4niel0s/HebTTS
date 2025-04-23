import argparse
from jiwer import wer, cer
from pathlib import Path
from infer import load_model
from omegaconf import OmegaConf
from valle.data.collation import get_text_token_collater
from valle.data.hebrew_normalizer import HebrewNormalizer
from valle.data.hebrew_root_tokenizer import AlefBERTRootTokenizer, replace_chars
from valle.data.tokenizer import AudioTokenizer, tokenize_audio
import librosa
import os
import torch
import torchaudio
import argparse
from pathlib import Path
import whisper
import yaml

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mbd",
        type=bool,
        default=False,
        help="use of multi band diffusion",
    )

    parser.add_argument(
        "--text",
        type=str,
        default="/home/yandex/APDL2425a/group_6/Documents/HebTTS/valle/bin/sentences_for_eval.txt",
        help="Text to be synthesized.",
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
        default="/home/yandex/APDL2425a/group_6/Documents/HebTTS/valle/exp/valle_dev/checkpoint-150000.pt",
        help="Path to the saved checkpoint.",
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
    
    parser.add_argument(
        "--whisper-path",
        type=str,
        default="/home/yandex/APDL2425a/group_6/large-v2.pt",
        help="Path to the whisper model.",
    )

    parser.add_argument(
        "--speakers-path",
        type=str,
        default="/home/yandex/APDL2425a/group_6/Documents/HebTTS/speakers",
        help="Path to the speakers directory.",
    )

    return parser.parse_args()

def infer(model,
          audio_tokenizer,
          text_collater,
            text,
            prompt_text,
            prompt_audio,
            top_k,
            temperature,
            args):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    audio_prompts = list()
    encoded_frames = tokenize_audio(audio_tokenizer, str(Path(args.speakers_path)/prompt_audio))

    audio_prompts.append(encoded_frames[0][0])
    audio_prompts = torch.concat(audio_prompts, dim=-1).transpose(2, 1)
    audio_prompts = audio_prompts.to(device)

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

    if args.mbd:
        samples = audio_tokenizer.mbd_decode(
            encoded_frames.transpose(2, 1)
        )
    else:
        samples = audio_tokenizer.decode(
            [(encoded_frames.transpose(2, 1), None)]
        )
    
    return samples[0]

def main(model,
         audio_tokenizer,
         text_collater,
         whisper_model,
         texts,
         top_k,
         temperature,
         args):
    norm = HebrewNormalizer()
    speakers_file = str(Path(args.speakers_path) / 'speakers.yaml')
    with open(speakers_file, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    for speaker, content in data.items():
        calculated_wer = []
        calculated_cer = []
        # audio_prompt = torchaudio.load(str(Path(args.speakers_path)/content["audio-prompt"]))[0]
        text_prompt = content["text-prompt"]
        for text in texts:
            sample = infer(
                model,
                audio_tokenizer,
                text_collater,
                text,
                text_prompt,
                content["audio-prompt"],
                top_k=top_k,
                temperature=temperature,
                args=args
            )
            norm_ref = norm(text)
            sample = torch.Tensor(librosa.resample(sample.detach().cpu().numpy(), orig_sr=24000, target_sr=16000))
            norm_hyp = norm(whisper_model.transcribe(sample, language="he"))
            calculated_wer.append(wer(norm_ref, norm_hyp))
            calculated_cer.append(cer(norm_ref, norm_hyp))
            print(f"wer: {calculated_wer[-1]}")
            print(f"cer: {calculated_cer[-1]}")
        print(f"speaker: {speaker}")
        print(f"wer: {sum(calculated_wer) / len(calculated_wer)}")
        print(f"cer: {sum(calculated_cer) / len(calculated_cer)}")
            
if __name__ == "__main__":
    # parse some args 
    args = get_args()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
        
    text_collater = get_text_token_collater(args.tokens_file)
    checkpoint_path=args.checkpoint
    model, text_tokens = load_model(checkpoint_path, device)
    alef_bert_tokenizer = AlefBERTRootTokenizer(vocab_file=args.vocab_file)
    whisper_model = whisper.load_model(args.whisper_path)
    whisper_model.to(device)
    audio_tokenizer = AudioTokenizer(mbd=args.mbd)
    if os.path.exists(args.text):
        with open(args.text, 'r') as f:
            texts = f.readlines()
    
    main(
        model=model,
        audio_tokenizer=audio_tokenizer,
        text_collater=text_collater,
        whisper_model=whisper_model,
        texts=texts,
        top_k=args.top_k,
        temperature=args.temperature,
        args=args)