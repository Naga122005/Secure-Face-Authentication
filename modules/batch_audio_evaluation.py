
"""
Batch audio quality evaluation for Secure Face Authentication.

Measures waveform SNR, MSE, RMSE and MAE for up to 20
eligible VCC2020 carrier WAV files. Also records payload
integrity and sender-match results.

Run from the project root:
    python -m modules.batch_audio_evaluation
"""

from pathlib import Path
import csv
import math
import re
import shutil
import subprocess
import sys
import traceback

import numpy as np
import soundfile as sf


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(r"D:\Secure_Face_Authentication")

DATASET = (
    ROOT
    / "Data"
    / "audio_dtaset"
    / "VCC2020-database-master"
    / "vcc2020_database_evaluation"
)

# Your dataset may contain a second nested evaluation folder.
if not DATASET.exists():
    DATASET = (
        ROOT
        / "Data"
        / "audio_dtaset"
        / "VCC2020-database-master"
        / "vcc2020_database_evaluation"
        / "vcc2020_database_evaluation"
    )

AUDIO_DIR = ROOT / "data" / "audio"
RESULTS_DIR = AUDIO_DIR / "batch_results"

STEGO_FILE = AUDIO_DIR / "stego_audio.wav"
PAYLOAD_EXTRACTOR = "modules.payload_extractor"
EMBEDDING_MODULE = "modules.steganographic_embedding"

TARGET_COUNT = 20
REQUIRED_SAMPLE_RATE = 24000

# Based on the current embedding implementation:
# 607-byte frame * 8 bits * 16 samples per bit = 77,696.
MIN_SAMPLES = 77696

SELECTOR_NAME = "selected_carrier_audio.txt"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_selector_file():
    """Find the existing carrier-selection text file."""

    matches = []

    for path in ROOT.rglob(SELECTOR_NAME):
        path_text = str(path).lower()

        if any(
            part in path_text
            for part in ("venv", ".git", "__pycache__", "batch_results")
        ):
            continue

        matches.append(path)

    if not matches:
        raise FileNotFoundError(
            f"Cannot find {SELECTOR_NAME} under {ROOT}.\n"
            "Check where your carrier_audio_selector module "
            "stores the selected carrier path."
        )

    # Prefer a selector whose current contents mention the project.
    for path in matches:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            if "secure_face_authentication" in content.lower():
                return path
        except OSError:
            pass

    return matches[0]


def run_module(module_name):
    """Run a project module and return its combined console output."""

    command = [sys.executable, "-m", module_name]

    result = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=300,
    )

    combined_output = (
        (result.stdout or "")
        + "\n"
        + (result.stderr or "")
    )

    print(combined_output)

    if result.returncode != 0:
        raise RuntimeError(
            f"Module {module_name} failed with exit code "
            f"{result.returncode}."
        )

    return combined_output


def calculate_audio_metrics(original_path, stego_path):
    """Calculate waveform quality metrics using the saved WAV files."""

    original, original_sr = sf.read(
        str(original_path),
        dtype="float64",
        always_2d=True,
    )

    stego, stego_sr = sf.read(
        str(stego_path),
        dtype="float64",
        always_2d=True,
    )

    if original_sr != stego_sr:
        raise ValueError(
            f"Sample-rate mismatch: original={original_sr}, "
            f"stego={stego_sr}"
        )

    if original.shape[1] != stego.shape[1]:
        raise ValueError(
            f"Channel-count mismatch: original={original.shape[1]}, "
            f"stego={stego.shape[1]}"
        )

    if original.shape != stego.shape:
        raise ValueError(
            f"Audio dimensions differ: original={original.shape}, "
            f"stego={stego.shape}. "
            "Metrics were not calculated on mismatched signals."
        )

    error = original - stego

    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))

    signal_power = float(np.mean(original ** 2))
    noise_power = mse

    if noise_power == 0:
        snr_db = float("inf")
    elif signal_power == 0:
        snr_db = float("-inf")
    else:
        snr_db = float(
            10.0 * np.log10(signal_power / noise_power)
        )

    return {
        "sample_rate_hz": original_sr,
        "samples": original.shape[0],
        "duration_seconds": original.shape[0] / original_sr,
        "snr_db": snr_db,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }


def extract_status(extraction_log):
    """Read the actual integrity and sender-match status lines."""

    integrity_match = re.search(
        r"SHA-256 verified\s*:\s*(PASS|FAIL)",
        extraction_log,
        re.IGNORECASE,
    )

    sender_match = re.search(
        r"Sender match\s*:\s*(True|False)",
        extraction_log,
        re.IGNORECASE,
    )

    integrity_status = (
        integrity_match.group(1).upper()
        if integrity_match
        else "UNKNOWN"
    )

    sender_status = (
        sender_match.group(1).capitalize()
        if sender_match
        else "UNKNOWN"
    )

    return integrity_status, sender_status


def csv_value(value):
    """Convert non-finite floats into readable CSV values."""

    if isinstance(value, float) and not math.isfinite(value):
        return str(value)

    return value


# ============================================================
# MAIN BATCH EVALUATION
# ============================================================

def main():
    print("=" * 72)
    print("BATCH AUDIO QUALITY EVALUATION")
    print("=" * 72)

    if not ROOT.exists():
        raise FileNotFoundError(f"Project folder not found: {ROOT}")

    if not DATASET.exists():
        raise FileNotFoundError(
            "VCC2020 dataset folder not found.\n"
            f"Checked: {DATASET}\n"
            "Update DATASET at the top of this script to match "
            "your actual dataset location."
        )

    if not STEGO_FILE.parent.exists():
        raise FileNotFoundError(
            f"Audio directory not found: {STEGO_FILE.parent}"
        )

    if not STEGO_FILE.exists():
        print(
            "Note: stego_audio.wav will be created by the "
            "embedding module during the first test."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    selector_path = find_selector_file()
    print(f"Carrier selector: {selector_path}")
    print(f"Dataset folder  : {DATASET}")
    print(f"Results folder  : {RESULTS_DIR}")

    # Save the original selector contents so they can be restored.
    original_selector_bytes = selector_path.read_bytes()

    # Discover the available WAV files.
    all_wav_files = sorted(
        (
            path
            for path in DATASET.rglob("*")
            if path.is_file()
            and path.suffix.lower() == ".wav"
            and "__MACOSX" not in str(path).upper()
        ),
        key=lambda path: str(path).lower(),
    )

    print(f"\nTotal WAV files scanned: {len(all_wav_files)}")

    # Select eligible audio files based on the embedding capacity.
    eligible_files = []

    for wav_path in all_wav_files:
        try:
            info = sf.info(str(wav_path))

            if (
                info.samplerate == REQUIRED_SAMPLE_RATE
                and info.channels == 1
                and info.frames >= MIN_SAMPLES
            ):
                eligible_files.append(wav_path)

        except Exception as exc:
            print(f"Skipping unreadable WAV: {wav_path.name} ({exc})")

    print(f"Eligible 24 kHz mono carriers: {len(eligible_files)}")

    if len(eligible_files) < TARGET_COUNT:
        raise RuntimeError(
            f"Only {len(eligible_files)} eligible audio files were found; "
            f"{TARGET_COUNT} are required.\n"
            f"Each file must be mono, {REQUIRED_SAMPLE_RATE} Hz, and "
            f"at least {MIN_SAMPLES} samples long.\n"
            "The script will not silently claim to have tested 20 files. "
            "Use a smaller TARGET_COUNT if fewer suitable files exist."
        )

    # Use different files in sorted order, without random selection.
    selected_files = eligible_files[:TARGET_COUNT]

    print(f"\nFiles selected for testing: {len(selected_files)}")
    print("-" * 72)

    csv_path = RESULTS_DIR / "batch_audio_results.csv"
    summary_path = RESULTS_DIR / "batch_summary.txt"
    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    results = []

    try:
        for index, carrier_path in enumerate(selected_files, start=1):
            print(
                f"\n[{index}/{TARGET_COUNT}] "
                f"{carrier_path.name}"
            )

            row = {
                "index": index,
                "carrier_file": carrier_path.name,
                "carrier_path": str(carrier_path),
                "status": "FAILED",
                "sample_rate_hz": "",
                "samples": "",
                "duration_seconds": "",
                "snr_db": "",
                "mse": "",
                "rmse": "",
                "mae": "",
                "sha256_integrity": "UNKNOWN",
                "sender_match": "UNKNOWN",
                "error": "",
            }

            log_parts = []

            try:
                # Set the carrier used by the existing embedder.
                selector_path.write_text(
                    str(carrier_path.resolve()),
                    encoding="utf-8",
                )

                # Run the project's existing waveform embedder.
                embedding_log = run_module(EMBEDDING_MODULE)
                log_parts.append(
                    "========== EMBEDDING LOG ==========\n"
                    + embedding_log
                )

                if not STEGO_FILE.exists():
                    raise FileNotFoundError(
                        "Embedding completed without producing "
                        f"{STEGO_FILE}"
                    )

                # Copy this stego output before the next carrier overwrites it.
                saved_stego_path = (
                    RESULTS_DIR
                    / f"{index:02d}_{carrier_path.stem}_stego.wav"
                )

                shutil.copy2(STEGO_FILE, saved_stego_path)

                # Calculate metrics from the original and saved stego WAV.
                metrics = calculate_audio_metrics(
                    carrier_path,
                    saved_stego_path,
                )
                row.update(metrics)

                # Run the existing extractor on the current stego output.
                extraction_log = run_module(PAYLOAD_EXTRACTOR)
                log_parts.append(
                    "========== EXTRACTION LOG ==========\n"
                    + extraction_log
                )

                integrity, sender_match = extract_status(extraction_log)

                row["sha256_integrity"] = integrity
                row["sender_match"] = sender_match

                if integrity == "PASS" and sender_match == "True":
                    row["status"] = "PASS"
                elif integrity == "FAIL" or sender_match == "False":
                    row["status"] = "PAYLOAD_CHECK_FAILED"
                else:
                    row["status"] = "METRICS_ONLY"

                print("\nAUDIO METRICS")
                print(f"Carrier       : {carrier_path.name}")
                print(f"Duration      : {row['duration_seconds']:.3f} s")
                print(f"SNR           : {row['snr_db']:.4f} dB")
                print(f"MSE           : {row['mse']:.10g}")
                print(f"RMSE          : {row['rmse']:.10g}")
                print(f"MAE           : {row['mae']:.10g}")
                print(f"SHA-256       : {integrity}")
                print(f"Sender match  : {sender_match}")
                print(f"Status        : {row['status']}")

            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["status"] = "FAILED"

                log_parts.append(
                    "\n========== ERROR ==========\n"
                    + traceback.format_exc()
                )

                print(f"FAILED: {row['error']}")
                print("Continuing to the next eligible audio file.")

            finally:
                log_path = log_dir / (
                    f"{index:02d}_{carrier_path.stem}.log"
                )
                log_path.write_text(
                    "\n\n".join(log_parts),
                    encoding="utf-8",
                )

                results.append(row)

                # Save progress after every audio file so partial results
                # are available even if a later test fails.
                fieldnames = list(results[0].keys())

                with csv_path.open(
                    "w",
                    newline="",
                    encoding="utf-8-sig",
                ) as csv_file:
                    writer = csv.DictWriter(
                        csv_file,
                        fieldnames=fieldnames,
                    )
                    writer.writeheader()

                    for result in results:
                        writer.writerow(
                            {
                                key: csv_value(value)
                                for key, value in result.items()
                            }
                        )

    finally:
        # Always restore the original carrier selection.
        selector_path.write_bytes(original_selector_bytes)
        print(f"\nOriginal selector restored: {selector_path}")

    # ========================================================
    # SUMMARY
    # ========================================================

    metric_rows = [
        row for row in results
        if isinstance(row["snr_db"], (int, float))
        and isinstance(row["mse"], (int, float))
        and isinstance(row["rmse"], (int, float))
        and isinstance(row["mae"], (int, float))
    ]

    payload_pass_count = sum(
        row["sha256_integrity"] == "PASS"
        and row["sender_match"] == "True"
        for row in results
    )

    fully_passed = sum(
        row["status"] == "PASS"
        for row in results
    )

    lines = [
        "BATCH AUDIO EVALUATION SUMMARY",
        "=" * 60,
        f"Dataset folder: {DATASET}",
        f"Eligible files: {len(eligible_files)}",
        f"Requested tests: {TARGET_COUNT}",
        f"Tests completed: {len(results)}",
        f"Files with waveform metrics: {len(metric_rows)}",
        f"Payload integrity and sender match passed: {payload_pass_count}",
        f"Fully passed tests: {fully_passed}",
        "",
    ]

    if metric_rows:
        finite_snr = [
            row["snr_db"]
            for row in metric_rows
            if math.isfinite(row["snr_db"])
        ]

        lines.append("AGGREGATE WAVEFORM METRICS")
        lines.append("-" * 60)

        if finite_snr:
            lines.append(
                f"Mean SNR: {np.mean(finite_snr):.4f} dB"
            )
            lines.append(
                f"Minimum SNR: {np.min(finite_snr):.4f} dB"
            )
            lines.append(
                f"Maximum SNR: {np.max(finite_snr):.4f} dB"
            )

        lines.append(
            f"Mean MSE: {np.mean([r['mse'] for r in metric_rows]):.10g}"
        )
        lines.append(
            f"Mean RMSE: {np.mean([r['rmse'] for r in metric_rows]):.10g}"
        )
        lines.append(
            f"Mean MAE: {np.mean([r['mae'] for r in metric_rows]):.10g}"
        )

        lines.extend([
            "",
            "INDIVIDUAL RESULTS",
            "-" * 60,
            (
                "File | Status | SNR (dB) | MSE | RMSE | MAE "
                "| SHA-256 | Sender match"
            ),
        ])

        for row in results:
            lines.append(
                f"{row['carrier_file']} | {row['status']} | "
                f"{row['snr_db']} | {row['mse']} | "
                f"{row['rmse']} | {row['mae']} | "
                f"{row['sha256_integrity']} | {row['sender_match']}"
            )

    failures = [
        row for row in results
        if row["status"] != "PASS"
    ]

    if failures:
        lines.extend(["", "FAILED OR INCOMPLETE TESTS", "-" * 60])

        for row in failures:
            lines.append(
                f"{row['carrier_file']}: {row['status']} - "
                f"{row['error'] or 'Payload checks were not both PASS/True'}"
            )

    summary_text = "\n".join(lines) + "\n"
    summary_path.write_text(summary_text, encoding="utf-8")

    print("\n" + summary_text)
    print(f"CSV results   : {csv_path}")
    print(f"Text summary  : {summary_path}")
    print(f"Per-file logs : {log_dir}")


if __name__ == "__main__":
    main()
