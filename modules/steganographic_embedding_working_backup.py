
from pathlib import Path
import hashlib
import struct

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "audio"

PAYLOAD_PATH = AUDIO_DIR / "encoded_biometric_payload.bin"
CARRIER_LIST = AUDIO_DIR / "selected_carrier_audio.txt"
OUTPUT_PATH = AUDIO_DIR / "stego_audio.wav"

MAGIC = b"CFD1"
VERSION = 1
HEADER_SIZE = 41

REPETITIONS = 8
QUANT_STEP = 0.002


def find_carrier():
    if CARRIER_LIST.exists():
        saved = CARRIER_LIST.read_text(encoding="utf-8").strip().strip('"')
        if saved:
            candidate = Path(saved)
            if candidate.exists():
                return candidate

            matches = list((ROOT / "Data").rglob(candidate.name))
            if matches:
                return matches[0]

    matches = list((ROOT / "Data").rglob("E30012.wav"))
    if matches:
        return matches[0]

    raise FileNotFoundError("Selected VCC2020 carrier WAV was not found.")


def make_frame(payload):
    header = (
        MAGIC
        + struct.pack(">B", VERSION)
        + struct.pack(">I", len(payload))
        + hashlib.sha256(payload).digest()
    )
    return header + payload


def bits_from_bytes(data):
    return np.unpackbits(
        np.frombuffer(data, dtype=np.uint8), bitorder="big"
    )


def embed_bit(a, b, bit):
    mean = (a + b) / 2.0
    difference = a - b
    base = int(np.floor(difference / QUANT_STEP))

    # Choose the closest quantization-bin center with the required parity.
    candidates = [
        k for k in range(base - 2, base + 3)
        if k % 2 == bit
    ]
    target_k = min(
        candidates,
        key=lambda k: abs(
            (k + 0.5) * QUANT_STEP - difference
        )
    )

    target_difference = (target_k + 0.5) * QUANT_STEP
    return (
        mean + target_difference / 2.0,
        mean - target_difference / 2.0
    )


def main():
    if not PAYLOAD_PATH.exists():
        raise FileNotFoundError(
            f"Payload not found: {PAYLOAD_PATH}\n"
            "Run modules.content_payload_encoder first."
        )

    carrier_path = find_carrier()
    payload = PAYLOAD_PATH.read_bytes()
    frame = make_frame(payload)
    bits = bits_from_bytes(frame)
    repeated_bits = np.repeat(bits, REPETITIONS)

    audio, sample_rate = sf.read(carrier_path, dtype="float64")
    mono = audio.ndim == 1

    # Preserve all channels; embed in channel 0 only for stereo audio.
    working = audio.copy() if mono else audio[:, 0].copy()

    required_samples = 2 * len(repeated_bits)
    if required_samples > len(working):
        raise ValueError(
            f"Insufficient carrier capacity: need {required_samples} "
            f"samples, available {len(working)}."
        )

    for i, bit in enumerate(repeated_bits):
        j = 2 * i
        working[j], working[j + 1] = embed_bit(
            working[j], working[j + 1], int(bit)
        )

    if mono:
        stego = working
    else:
        stego = audio.copy()
        stego[:, 0] = working

    # Avoid clipping; the carrier is expected to be normalized audio.
    if np.max(np.abs(stego)) > 1.0:
        raise ValueError(
            "Embedded waveform exceeds the normalized audio range. "
            "Use a normalized carrier to avoid clipping."
        )

    sf.write(OUTPUT_PATH, stego, sample_rate, subtype="PCM_16")

    print("=" * 60)
    print("WAVEFORM EMBEDDING COMPLETE")
    print("=" * 60)
    print(f"Carrier       : {carrier_path}")
    print(f"Payload bytes : {len(payload)}")
    print(f"Frame bytes   : {len(frame)}")
    print(f"Sample rate   : {sample_rate} Hz")
    print(f"Output        : {OUTPUT_PATH}")
    print(f"Bits embedded: {len(bits)}")
    print("Run the receiver before evaluating final results.")


if __name__ == "__main__":
    main()
