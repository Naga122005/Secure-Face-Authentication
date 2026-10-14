
from pathlib import Path
import hashlib
import struct

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "audio"

STEGO_PATH = AUDIO_DIR / "received_stego_audio.wav"
EXPECTED_PATH = AUDIO_DIR / "encoded_biometric_payload.bin"
RECOVERED_PATH = AUDIO_DIR / "recovered_biometric_payload.bin"

MAGIC = b"CFD1"
HEADER_SIZE = 41
REPETITIONS = 8
QUANT_STEP = 0.002


def extract_bits(audio, count):
    available_bits = min(count, len(audio) // (2 * REPETITIONS))
    recovered = []

    for i in range(available_bits):
        votes = []
        for r in range(REPETITIONS):
            pair_index = (i * REPETITIONS + r) * 2
            difference = audio[pair_index] - audio[pair_index + 1]
            quantized_index = int(np.floor(difference / QUANT_STEP))
            votes.append(quantized_index % 2)

        recovered.append(1 if sum(votes) > len(votes) // 2 else 0)

    return np.asarray(recovered, dtype=np.uint8)


def bits_to_bytes(bits):
    usable = (len(bits) // 8) * 8
    if usable == 0:
        return b""
    return np.packbits(bits[:usable], bitorder="big").tobytes()


def main():
    if not STEGO_PATH.exists():
        raise FileNotFoundError(f"Stego audio missing: {STEGO_PATH}")

    audio, sample_rate = sf.read(STEGO_PATH, dtype="float64")
    if audio.ndim == 2:
        audio = audio[:, 0]

    # Read the fixed 9-byte prefix: magic + version + payload length.
    prefix_bits = extract_bits(audio, 9 * 8)
    prefix = bits_to_bytes(prefix_bits)

    if len(prefix) < 9 or prefix[:4] != MAGIC:
        raise ValueError(
            "CFD1 header not recovered. Ensure this stego file was "
            "created by the matching steganographic_embedding module."
        )

    version = prefix[4]
    payload_length = struct.unpack(">I", prefix[5:9])[0]
    if version != 1:
        raise ValueError(f"Unsupported payload version: {version}")

    total_frame_bytes = HEADER_SIZE + payload_length
    total_frame_bits = total_frame_bytes * 8

    bits = extract_bits(audio, total_frame_bits)
    frame = bits_to_bytes(bits)

    if len(frame) != total_frame_bytes:
        raise ValueError("Not enough samples to recover the full payload.")

    payload = frame[HEADER_SIZE:]
    expected_digest = frame[9:41]
    actual_digest = hashlib.sha256(payload).digest()

    if expected_digest != actual_digest:
        raise ValueError(
            "Payload SHA-256 verification failed. "
            "The recovered data is corrupted."
        )

    RECOVERED_PATH.write_bytes(payload)

    print("=" * 60)
    print("PAYLOAD EXTRACTION RESULTS")
    print("=" * 60)
    print(f"Sample rate        : {sample_rate} Hz")
    print(f"Recovered bytes    : {len(payload)}")
    print(f"SHA-256 verified   : PASS")
    print(f"Recovered file     : {RECOVERED_PATH}")

    if EXPECTED_PATH.exists():
        expected = EXPECTED_PATH.read_bytes()
        print(f"Sender match       : {payload == expected}")
        if payload != expected:
            raise ValueError("Recovered payload differs from sender payload.")
    else:
        print("Sender match       : Not checked (sender file missing)")


if __name__ == "__main__":
    main()
