import sys
sys.path.append('third_party/Matcha-TTS')
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed
from tqdm import tqdm
from typing import Optional
import soundfile as sf
import torch
import os
import time
import torch
import torchaudio

os.makedirs("CosyVoice/output", exist_ok=True)
cosyvoice = AutoModel(model_dir="/data/ptmodels/tts/Fun-CosyVoice3-0.5B", load_trt=True, load_vllm=True, fp16=False)
def save_wav(path, wav, sr):
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    if wav.ndim == 2:
        wav = wav.squeeze(0)
    sf.write(path, wav, sr)

texts = [
        "有能力还",
    ]


def cosyvoice3_example():
    num = 0
    out_dir = "CosyVoice/output"
    os.makedirs(out_dir, exist_ok=True)

    for t_idx, text in enumerate(texts):
        num += 1

        t_start = time.perf_counter()
        first_packet_time = None
        audio_chunks = []

        for i, j in enumerate(
            cosyvoice.inference_zero_shot(
                text,
                '<|zh|>You are a helpful assistant.<|endofprompt|>希望你以后能做的比我还好哟',
                'CosyVoice/asset/zero_shot_prompt.wav',
                stream=True
            )
        ):
            # ---------- first packet latency ----------
            if first_packet_time is None:
                first_packet_time = time.perf_counter()
                latency = first_packet_time - t_start
                print(f"[CV3] text {num} first packet latency: {latency:.3f}s")

            # ---------- collect audio ----------
            chunk = j["tts_speech"]
            if hasattr(chunk, "detach"):
                chunk = chunk.detach().cpu()
            audio_chunks.append(chunk)

        # ---------- save wav ----------
        if len(audio_chunks) == 0:
            print(f"[CV3] text {num} no audio generated")
            continue

        wav = torch.cat(audio_chunks, dim=-1)  # [1, T]
        out_path = f"{out_dir}/cv3_{num}.wav"
        torchaudio.save(out_path, wav, cosyvoice.sample_rate)

        audio_len = wav.shape[1] / cosyvoice.sample_rate
        total_time = time.perf_counter() - t_start
        print(
            f"[CV3] text {num} saved: {out_path} | "
            f"audio_len={audio_len:.2f}s | "
            f"RTF={total_time / audio_len:.3f}"
        )

def main():
    cosyvoice3_example()

if __name__ == "__main__":
    main()
