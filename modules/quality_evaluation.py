
from pathlib import Path
import math
import numpy as np
import soundfile as sf
import librosa

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "audio"
STEGO_PATH = AUDIO_DIR / "stego_audio.wav"
CARRIER_LIST = AUDIO_DIR / "selected_carrier_audio.txt"
PAYLOAD_PATH = AUDIO_DIR / "encoded_biometric_payload.bin"

REPETITIONS = 8
QUANT_STEP = 0.002

def read_carrier_path():
    if not CARRIER_LIST.exists():
        raise FileNotFoundError(f"Missing: {CARRIER_LIST}")
    path = Path(CARRIER_LIST.read_text(encoding="utf-8").strip())
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Carrier not found: {path}")
    return path

def mse(a, b):
    n = min(len(a), len(b))
    return float(np.mean((a[:n] - b[:n]) ** 2))

def cosine_similarity(a, b):
    a = np.asarray(a).ravel().astype(np.float64)
    b = np.asarray(b).ravel().astype(np.float64)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else float("nan")

def compare_vectors(name, original, recovered):
    # Align frame counts before comparing.
    frames = min(original.shape[1], recovered.shape[1])
    a = original[:, :frames]
    b = recovered[:, :frames]

    print(f"\n{name}")
    print(f"  Shape compared : {a.shape}")
    print(f"  MSE            : {mse(a, b):.10f}")
    print(f"  RMSE           : {math.sqrt(mse(a, b)):.10f}")
    print(f"  MAE            : {float(np.mean(np.abs(a-b))):.10f}")
    print(f"  Cosine similarity (flattened): "
          f"{cosine_similarity(a, b):.8f}")

def main():
    carrier_path = read_carrier_path()
    if not STEGO_PATH.exists():
        raise FileNotFoundError(f"Missing: {STEGO_PATH}")

    original, sr1 = sf.read(carrier_path, dtype="float32")
    stego, sr2 = sf.read(STEGO_PATH, dtype="float32")

    if original.ndim > 1:
        original = original.mean(axis=1)
    if stego.ndim > 1:
        stego = stego.mean(axis=1)
    if sr1 != sr2:
        raise ValueError("Carrier and stego sample rates differ.")

    n = min(len(original), len(stego))
    original, stego = original[:n], stego[:n]
    error = original - stego
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(error ** 2)

    snr = (10 * np.log10(signal_power / noise_power)
           if noise_power > 0 else float("inf"))

    print("=" * 62)
    print("AUDIO, CONTENT/STYLE AND CAPACITY EVALUATION")
    print("=" * 62)
    print(f"Carrier          : {carrier_path}")
    print(f"Stego            : {STEGO_PATH}")
    print(f"Sample rate      : {sr1} Hz")
    print(f"Duration         : {n / sr1:.4f} seconds")
    print(f"Raw SNR          : {snr:.4f} dB")
    print(f"MSE              : {np.mean(error ** 2):.10f}")
    print(f"RMSE             : {np.sqrt(np.mean(error ** 2)):.10f}")
    print(f"MAE              : {np.mean(np.abs(error)):.10f}")

    # Consistent MFCC-based proxy features for both signals.
    # This split is a diagnostic proxy, not proof of learned
    # semantic content/style disentanglement.
    def get_features(audio):
        return librosa.feature.mfcc(
            y=audio,
            sr=sr1,
            n_mfcc=13,
            n_fft=1024,
            hop_length=256
        )

    orig_mfcc = get_features(original)
    stego_mfcc = get_features(stego)

    compare_vectors("MFCC feature comparison", orig_mfcc, stego_mfcc)
    compare_vectors("Content proxy (MFCC rows 0-7)",
                    orig_mfcc[:8], stego_mfcc[:8])
    compare_vectors("Style proxy (MFCC rows 8-12)",
                    orig_mfcc[8:], stego_mfcc[8:])

    # PESQ: standard wideband PESQ operates at 16 kHz.
    try:
        from pesq import pesq
        ref16 = librosa.resample(
            original, orig_sr=sr1, target_sr=16000
        )
        deg16 = librosa.resample(
            stego, orig_sr=sr1, target_sr=16000
        )
        n16 = min(len(ref16), len(deg16))
        score = pesq(16000, ref16[:n16], deg16[:n16], "wb")
        print(f"\nPESQ wideband    : {score:.4f}")
    except Exception as exc:
        print("\nPESQ unavailable or failed:", exc)
        print("Check the pesq installation and input signal validity.")

    # Capacity estimate for this implementation:
    # one bit per sample pair, with each bit repeated 8 times.
    pair_count = n // 2
    capacity_bits = pair_count // REPETITIONS
    capacity_bytes = capacity_bits // 8
    frame_bytes = 41 + (
        PAYLOAD_PATH.stat().st_size if PAYLOAD_PATH.exists() else 0
    )
    frame_bits = frame_bytes * 8
    duration = n / sr1

    print("\nSTEGANOGRAPHIC CAPACITY")
    print(f"Available sample pairs       : {pair_count}")
    print(f"Estimated capacity           : {capacity_bytes} bytes")
    print(f"Embedded frame size estimate : {frame_bytes} bytes")
    print(f"Frame bits per second        : {frame_bits / duration:.2f} bps")
    print(f"Capacity utilization         : "
          f"{100 * frame_bytes / capacity_bytes:.2f}%"
          if capacity_bytes else "Capacity utilization unavailable")

    print("\nNote: capacity is a theoretical estimate based on the")
    print("current sample-pair and repetition settings.")

if __name__ == "__main__":
    main()
