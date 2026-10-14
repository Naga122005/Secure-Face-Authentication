
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "audio"

STEGO_PATH = AUDIO_DIR / "stego_audio.wav"
CARRIER_LIST = AUDIO_DIR / "selected_carrier_audio.txt"


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

    raise FileNotFoundError("Original carrier WAV was not found.")


def main():
    if not STEGO_PATH.exists():
        raise FileNotFoundError(f"Stego audio missing: {STEGO_PATH}")

    carrier_path = find_carrier()

    original, sr_original = sf.read(carrier_path, dtype="float64")
    stego, sr_stego = sf.read(STEGO_PATH, dtype="float64")

    if sr_original != sr_stego:
        raise ValueError(
            f"Sample rates differ: original={sr_original}, stego={sr_stego}"
        )

    if original.shape != stego.shape:
        raise ValueError(
            f"Audio shapes differ: original={original.shape}, "
            f"stego={stego.shape}. Compare matching channels and lengths."
        )

    error = original - stego
    signal_power = float(np.mean(original ** 2))
    noise_power = float(np.mean(error ** 2))

    mse = noise_power
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))

    if signal_power <= 0:
        snr = float("nan")
    elif noise_power <= 1e-30:
        snr = float("inf")
    else:
        snr = 10.0 * np.log10(signal_power / noise_power)

    # Optional diagnostic: compensate for constant gain differences.
    denominator = float(np.sum(stego ** 2))
    gain = (
        float(np.sum(original * stego)) / denominator
        if denominator > 1e-30 else 1.0
    )
    aligned_error = original - gain * stego
    aligned_noise_power = float(np.mean(aligned_error ** 2))

    aligned_snr = (
        float("inf")
        if aligned_noise_power <= 1e-30
        else 10.0 * np.log10(signal_power / aligned_noise_power)
    )

    print("=" * 65)
    print("STEGO AUDIO QUALITY EVALUATION")
    print("=" * 65)
    print(f"Original carrier : {carrier_path}")
    print(f"Stego audio      : {STEGO_PATH}")
    print(f"Sample rate      : {sr_original} Hz")
    print(f"Samples          : {original.shape[0]}")
    print(f"Original RMS     : {np.sqrt(signal_power):.8f}")
    print(f"Stego RMS        : {np.sqrt(np.mean(stego ** 2)):.8f}")

    print("\nMEASURED METRICS")
    print("-" * 65)
    print(f"Raw SNR          : {snr:.4f} dB")
    print(f"Gain-aligned SNR : {aligned_snr:.4f} dB")
    print(f"MSE              : {mse:.10f}")
    print(f"RMSE             : {rmse:.10f}")
    print(f"MAE              : {mae:.10f}")
    print(f"Best-fit gain    : {gain:.8f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
