import sys
import json
import os
import requests
import re
import struct
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def fill_hann_window(size, periodic=True):
    if periodic:
        return np.hanning(size + 1)[:-1]
    return np.hanning(size)


def irfft(n_fft, complex_input):
    return np.fft.irfft(complex_input, n=n_fft)


def fold(buffer, n_out, n_win, n_hop, n_pad):
    result = np.zeros(n_out)
    n_frames = len(buffer) // n_win

    for i in range(n_frames):
        start = i * n_hop
        end = start + n_win
        result[start:end] += buffer[i * n_win:(i + 1) * n_win]

    return result[n_pad:-n_pad] if n_pad > 0 else result


def process_frame(args):
    l, n_fft, ST, hann = args
    frame = irfft(n_fft, ST[l])
    frame = frame * hann
    hann2 = hann * hann
    return frame, hann2


def embd_to_audio(embd, n_codes, n_embd, n_thread=4):
    embd = np.asarray(embd, dtype=np.float32).reshape(n_codes, n_embd)

    n_fft = 1280
    n_hop = 320
    n_win = 1280
    n_pad = (n_win - n_hop) // 2
    n_out = (n_codes - 1) * n_hop + n_win

    hann = fill_hann_window(n_fft, True)

    E = np.zeros((n_embd, n_codes), dtype=np.float32)
    for l in range(n_codes):
        for k in range(n_embd):
            E[k, l] = embd[l, k]

    half_embd = n_embd // 2
    S = np.zeros((n_codes, half_embd + 1), dtype=np.complex64)

    for k in range(half_embd):
        for l in range(n_codes):
            mag = E[k, l]
            phi = E[k + half_embd, l]

            mag = np.clip(np.exp(mag), 0, 1e2)
            S[l, k] = mag * np.exp(1j * phi)

    res = np.zeros(n_codes * n_fft)
    hann2_buffer = np.zeros(n_codes * n_fft)

    with ThreadPoolExecutor(max_workers=n_thread) as executor:
        args = [(l, n_fft, S, hann) for l in range(n_codes)]
        results = list(executor.map(process_frame, args))

        for l, (frame, hann2) in enumerate(results):
            res[l*n_fft:(l+1)*n_fft] = frame
            hann2_buffer[l*n_fft:(l+1)*n_fft] = hann2

    audio = fold(res, n_out, n_win, n_hop, n_pad)
    env = fold(hann2_buffer, n_out, n_win, n_hop, n_pad)

    mask = env > 1e-10
    audio[mask] /= env[mask]

    return audio


def save_wav(filename, audio_data, sample_rate):
    num_channels = 1
    bits_per_sample = 16
    bytes_per_sample = bits_per_sample // 8
    data_size = len(audio_data) * bytes_per_sample
    byte_rate = sample_rate * num_channels * bytes_per_sample
    block_align = num_channels * bytes_per_sample
    chunk_size = 36 + data_size  # 36 = size of header minus first 8 bytes

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        chunk_size,
        b'WAVE',
        b'fmt ',
        16,                # fmt chunk size
        1,                 # audio format (PCM)
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )

    audio_data = np.clip(audio_data * 32767, -32768, 32767)
    pcm_data = audio_data.astype(np.int16)

    with open(filename, 'wb') as f:
        f.write(header)
        f.write(pcm_data.tobytes())


def process_text(text: str):
    text = re.sub(r'\d+(\.\d+)?', lambda x: x.group(), text.lower()) # TODO this needs to be fixed
    text = re.sub(r'[-_/,\.\\]', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()

def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "outetts-config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # usage:
    # python tts-outetts.py http://server-llm:port http://server-dec:port "text"

    if len(sys.argv) <= 3:
        print("usage: python tts-outetts.py http://server-llm:port http://server-dec:port \"text\"")
        exit(1)

    host_llm = sys.argv[1]
    host_dec = sys.argv[2]
    text = sys.argv[3]

    config = load_config()
    prefix = config["prefix"]
    suffix = config["suffix"]

    words = process_text(text)
    words = "<|text_sep|>".join([i.strip() for i in words])
    words += "<|text_end|>\n"

    response = requests.post(
        host_llm + "/completion",
        json={
            "prompt": [prefix + words, *suffix],
            "n_predict": 1024,
            "cache_prompt": True,
            "return_tokens": True,
            "samplers": ["top_k"],
            "top_k": 16,
            "seed": 1003,
        }
    )

    response_json = response.json()

    #print(json.dumps(response_json, indent=4))
    #print(json.dumps(response_json["prompt"], indent=4).replace("\\n", "\n"))
    #print(json.dumps(response_json["timings"], indent=4))
    #print(json.dumps(response_json["tokens"], indent=4))

    codes = response_json["tokens"]

    codes = [t - 151672 for t in codes if t >= 151672 and t <= 155772]

    response = requests.post(
        host_dec + "/embeddings",
        json={
            "input": [*codes],
        }
    )

    response_json = response.json()

    #print(json.dumps(response_json, indent=4))

    # spectrogram
    embd = response_json[0]["embedding"]

    n_codes = len(embd)
    n_embd = len(embd[0])

    print('spectrogram generated: n_codes: %d, n_embd: %d' % (n_codes, n_embd))

    # post-process the spectrogram to convert to audio
    print('converting to audio ...')
    audio = embd_to_audio(embd, n_codes, n_embd)
    print('audio generated: %d samples' % len(audio))

    filename = "output.wav"
    sample_rate = 24000 # sampling rate

    # zero out first 0.25 seconds
    audio[:24000 // 4] = 0.0

    save_wav(filename, audio, sample_rate)
    print('audio written to file "%s"' % filename)

if __name__ == "__main__":
    main()
