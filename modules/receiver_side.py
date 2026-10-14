
"""
Receiver-side diagnostics for Secure Face Authentication using
Audio Steganography and Cancelable Biometrics.

Reports:
- Audio MSE, RMSE, MAE, SNR and correlation
- MFCC/content-feature recovery statistics
- Payload header validation

Run from the project root:
    python -m modules.receiver_side
"""

from pathlib import Path
import numpy as np
import soundfile as sf
import librosa


ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "audio"

RECEIVED_AUDIO = AUDIO_DIR / "received_stego_audio.wav"
#ORIGINAL_AUDIO = AUDIO_DIR / "carrier_audio.wav"
SELECTED_PATH_FILE = AUDIO_DIR / "selected_carrier_audio.txt"

if SELECTED_PATH_FILE.exists():
    ORIGINAL_AUDIO = Path(
        SELECTED_PATH_FILE.read_text(encoding="utf-8").strip()
    )
else:
    ORIGINAL_AUDIO = None
# Existing project feature files, when available
ORIGINAL_CONTENT = AUDIO_DIR / "content_features.npy"
MODIFIED_CONTENT = AUDIO_DIR / "modified_content_features.npy"

EXPECTED_MAGIC = b"CFD1"
EXPECTED_HEADER_SIZE = 41


def print_section(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def load_audio(path):
    if not path.exists():
        return None, None

    audio, sample_rate = sf.read(str(path), dtype="float32")

    # Convert stereo to mono for comparable measurements.
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    return audio, sample_rate


def align_signals(reference, received):
    """Align lengths for basic numerical comparison."""
    length = min(len(reference), len(received))
    return reference[:length], received[:length]


def calculate_audio_metrics(reference, received):
    reference, received = align_signals(reference, received)

    reference64 = reference.astype(np.float64)
    received64 = received.astype(np.float64)
    error = reference64 - received64

    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))

    signal_power = float(np.mean(reference64 ** 2))
    noise_power = mse

    if noise_power == 0:
        snr = float("inf")
    elif signal_power == 0:
        snr = float("-inf")
    else:
        snr = float(10 * np.log10(signal_power / noise_power))

    if np.std(reference64) == 0 or np.std(received64) == 0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(reference64, received64)[0, 1])

    return {
        "samples_compared": len(reference),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "snr_db": snr,
        "correlation": correlation,
    }


def extract_content_features(audio, sample_rate):
    """
    Diagnostic feature extraction: calculate 13 MFCCs and use the
    first 8 rows as content features, matching the project's 8+5 split.
    If your existing feature_decoupling.py uses a different operation,
    its exact output should be used for the final payload test.
    """
    mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=13,
    n_fft=2048,
    hop_length=512,
)
    content = mfcc[:8, :]
    style = mfcc[8:, :]
    return mfcc, content, style


def feature_metrics(reference_path, recovered):
    if not reference_path.exists():
        print(f"Reference feature file not found: {reference_path}")
        return

    reference = np.load(reference_path)

    if reference.shape != recovered.shape:
        print("Feature shapes differ:")
        print(f"  Reference: {reference.shape}")
        print(f"  Recovered: {recovered.shape}")
        print("Metrics below use the common overlapping region.")

        rows = min(reference.shape[0], recovered.shape[0])
        cols = min(reference.shape[1], recovered.shape[1])
        reference = reference[:rows, :cols]
        recovered = recovered[:rows, :cols]

    error = reference.astype(np.float64) - recovered.astype(np.float64)

    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    max_error = float(np.max(np.abs(error)))

    print(f"Reference file : {reference_path.name}")
    print(f"Compared shape : {reference.shape}")
    print(f"Feature MSE    : {mse:.6f}")
    print(f"Feature RMSE   : {rmse:.6f}")
    print(f"Feature MAE    : {mae:.6f}")
    print(f"Maximum error  : {max_error:.6f}")


def extract_diagnostic_payload(content):
    """
    Diagnostic only. This reads 4 bits from each content value using
    the same floor(value / 16) modulo-16 symbol mapping described in
    the current steganographic_embedding.py.

    It does not claim that the payload survived audio reconstruction.
    """
    flat = np.asarray(content, dtype=np.float64).reshape(-1)
    symbols = np.floor(flat / 16.0).astype(np.int64) % 16

    bits = []
    for symbol in symbols:
        bits.extend([
            (symbol >> 3) & 1,
            (symbol >> 2) & 1,
            (symbol >> 1) & 1,
            symbol & 1,
        ])

    usable_bits = (len(bits) // 8) * 8
    if usable_bits == 0:
        return b""

    bit_array = np.asarray(bits[:usable_bits], dtype=np.uint8)
    return np.packbits(bit_array).tobytes()



def main():
    print_section("RECEIVER-SIDE AUDIO AND FEATURE DIAGNOSTICS")
    print(f"Project root: {ROOT}")

    # --------------------------------------------------------
    # 1. LOAD RECEIVED STEGO AUDIO
    # --------------------------------------------------------
    received, received_sr = load_audio(RECEIVED_AUDIO)

    if received is None:
        print(f"\nERROR: Received audio not found:\n{RECEIVED_AUDIO}")
        return

    print("\nReceived stego audio")
    print(f"  Path       : {RECEIVED_AUDIO}")
    print(f"  Sample rate: {received_sr} Hz")
    print(f"  Samples    : {len(received)}")
    print(f"  Duration   : {len(received) / received_sr:.3f} seconds")

    # --------------------------------------------------------
    # 2. LOAD THE EXACT SELECTED CARRIER PATH
    # --------------------------------------------------------
    print_section("AUDIO QUALITY METRICS")

    selected_path_file = AUDIO_DIR / "selected_carrier_audio.txt"
    original_audio_path = None

    if selected_path_file.exists():
        saved_path = selected_path_file.read_text(
            encoding="utf-8"
        ).strip()

        if saved_path:
            original_audio_path = Path(saved_path)

    if original_audio_path is None:
        print("ERROR: Selected carrier path was not saved.")
        print(f"Expected path file: {selected_path_file}")

    elif not original_audio_path.exists():
        print("ERROR: Saved carrier WAV does not exist.")
        print(f"Saved path: {original_audio_path}")

    else:
        original, original_sr = load_audio(original_audio_path)

        print(f"Original carrier: {original_audio_path}")

        if original is None:
            print("ERROR: Could not load the original carrier.")
        elif original_sr != received_sr:
            print("ERROR: Sample rates differ.")
            print(f"Original sample rate: {original_sr} Hz")
            print(f"Received sample rate: {received_sr} Hz")
            print("Resample both signals consistently before comparison.")
        else:
            metrics = calculate_audio_metrics(original, received)

            print(f"Samples compared: {metrics['samples_compared']}")
            print(f"Audio MSE       : {metrics['mse']:.8f}")
            print(f"Audio RMSE      : {metrics['rmse']:.8f}")
            print(f"Audio MAE       : {metrics['mae']:.8f}")
            print(f"Audio SNR       : {metrics['snr_db']:.4f} dB")
            print(f"Correlation     : {metrics['correlation']:.6f}")

            if len(original) != len(received):
                print(
                    "\nWARNING: Audio lengths differ. Metrics use the "
                    "overlapping samples only."
                )

    # --------------------------------------------------------
    # 3. EXTRACT RECEIVED AUDIO FEATURES
    # --------------------------------------------------------
    print_section("RECEIVED AUDIO FEATURE EXTRACTION")

    try:
        mfcc, content, style = extract_content_features(
            received,
            received_sr,
        )
    except Exception as exc:
        print(f"ERROR: Feature extraction failed: {exc}")
        return

    print(f"MFCC shape    : {mfcc.shape}")
    print(f"Content shape : {content.shape}")
    print(f"Style shape   : {style.shape}")

    # --------------------------------------------------------
    # 4. FEATURE ERROR METRICS
    # --------------------------------------------------------
    print_section("FEATURE ERROR METRICS")

    if ORIGINAL_CONTENT.exists():
        feature_metrics(ORIGINAL_CONTENT, content)
    else:
        print(f"Original content features not found: {ORIGINAL_CONTENT}")

    if MODIFIED_CONTENT.exists():
        feature_metrics(MODIFIED_CONTENT, content)
    else:
        print(f"Modified content features not found: {MODIFIED_CONTENT}")

    # --------------------------------------------------------
    # 5. DIAGNOSTIC PAYLOAD HEADER CHECK
    # --------------------------------------------------------
    print_section("PAYLOAD HEADER DIAGNOSTIC")

    diagnostic_bytes = extract_diagnostic_payload(content)

    print(f"Extracted diagnostic bytes: {len(diagnostic_bytes)}")
    print(f"First 8 bytes             : {diagnostic_bytes[:8]!r}")
    print(f"Expected magic             : {EXPECTED_MAGIC!r}")

    if diagnostic_bytes.startswith(EXPECTED_MAGIC):
        print("Magic check: PASS")
        print(
            "Next: validate the payload length and checksum before "
            "attempting decryption."
        )
    else:
        print("Magic check: FAIL")
        print(
            "The expected header was not recovered. Do not decrypt "
            "the extracted bytes."
        )

    # --------------------------------------------------------
    # 6. FINAL STATUS
    # --------------------------------------------------------
    print_section("RECEIVER DIAGNOSTIC COMPLETE")
    print("Audio and feature diagnostics completed.")
    print("Payload extraction must pass integrity checks before authentication.")


if __name__ == "__main__":
    main()
