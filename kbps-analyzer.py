# This script analyzes audio files to determine their technical characteristics.
# It calculates a "dominant" cutoff frequency by building a histogram of per-frame
# spectral rolloff values (in mono) and then infers the original bitrate based on that
# cutoff frequency. The final results are printed in a table using Rich.
#
# /// script
# dependencies = [
#   "mutagen",
#   "numpy",
#   "librosa",
#   "matplotlib",
#   "tqdm",
#   "rich"
# ]
# ///

import argparse
import sys
import os
from typing import Tuple, Optional, List
import numpy as np
import librosa
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3, COMM
import re
import asyncio
import concurrent.futures
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

debug: bool = False

def get_cutoff(file_path: str) -> Optional[float]:
    """
    Loads audio in mono, computes spectral rolloff per frame (roll_percent=0.999),
    then builds a histogram of the rolloff values to find the 'mode' bin.
    The mode bin is interpreted as the dominant cutoff frequency, ignoring occasional spikes.
    Returns the cutoff in kHz.
    """
    try:
        # If debugging, try importing matplotlib for plotting
        if debug:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                plt = None
        else:
            plt = None

        y, sr = librosa.load(file_path, sr=None, mono=True)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.999)
        rolloff_values: np.ndarray = rolloff.flatten()

        bin_size_hz: int = 500  # each bin covers 500 Hz
        max_val: float = np.max(rolloff_values)
        bins: np.ndarray = np.arange(0, max_val + bin_size_hz, bin_size_hz)
        hist, bin_edges = np.histogram(rolloff_values, bins=bins)

        max_bin_idx: int = int(np.argmax(hist))
        bin_high: float = bin_edges[max_bin_idx + 1]
        cutoff_hz: float = bin_high

        if plt is not None:
            plt.figure(figsize=(8, 4))
            plt.bar(bin_edges[:-1], hist, width=bin_size_hz, align='edge', alpha=0.7)
            plt.axvline(cutoff_hz, color='red', linestyle='--', label=f"Mode cutoff = {cutoff_hz:.0f} Hz")
            plt.title("Spectral Rolloff Histogram")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Count")
            plt.legend()
            plot_filename: str = os.path.splitext(os.path.basename(file_path))[0] + "_rolloff_hist.png"
            plt.savefig(plot_filename, dpi=120)
            plt.close()

        return cutoff_hz / 1000.0  # convert Hz to kHz

    except Exception as e:
        print("Error analyzing audio for cutoff frequency:", e)
        return None

def infer_bitrate_from_cutoff(file_type: str, avg_cutoff: float) -> Optional[int]:
    """
    Infers the original bitrate based on the dominant cutoff frequency and file type.
    Returns the inferred bitrate (in kbps) or None if it cannot be determined.
    """
    if file_type == "mp3":
        if avg_cutoff < 16:         # ~11 kHz cutoff
            return 64
        elif 16 <= avg_cutoff < 19:   # ~16 kHz cutoff
            return 128
        elif 19 <= avg_cutoff < 20:   # ~19 kHz cutoff
            return 192
        elif 20 <= avg_cutoff:        # ~20 kHz cutoff
            return 320
        else:
            return None
    elif file_type == "m4a":
        if avg_cutoff <= 22:          # ~22 kHz cutoff
            return 500
        else:
            return None
    else:
        return None

def update_id3_comment(file_path: str, inferred_bitrate: int) -> Optional[Tuple[str, str]]:
    """
    Updates the ID3 comment field of an MP3 file by prepending the inferred bitrate in the format:
      '~<inferred_bitrate>kbps ' followed by the original comment (if any).
    Any existing bitrate marker of the form '~<number>kbps' is removed.
    
    Returns a tuple containing the existing (old) comment and the new comment.
    """
    try:
        audio: ID3 = ID3(file_path)
        comments = audio.getall("COMM")
        old_comment: str = ""
        if comments:
            for comm in comments:
                if comm.lang == "eng" and comm.desc == "":
                    old_comment = comm.text[0]
                    break
            else:
                old_comment = comments[0].text[0]
        else:
            old_comment = ""

        cleaned_comment: str = re.sub(r"^~\d+kbps\.?\s*", "", old_comment).strip() if old_comment else ""
        new_comment: str = f"~{inferred_bitrate}kbps " + cleaned_comment if cleaned_comment else f"~{inferred_bitrate}kbps"
        audio.delall("COMM")
        audio.add(COMM(encoding=3, lang='eng', desc='', text=new_comment))
        audio.save()
        return old_comment, new_comment
    except Exception as e:
        print("Error updating ID3 comment:", e)
        return None

def process_file(file_path: str, write_flag: bool = False) -> Tuple[str, str, str, str]:
    """
    Determines the file type by extension, computes the dominant cutoff frequency,
    infers the bitrate, and returns a tuple with:
      (File, Dominant Cutoff, Inferred Bitrate, Comment)
    For MP3 files, if the comment is updated, the returned comment includes Rich markup:
    - If updated: "[green]Updated:[/green] <new_comment> [dim](was <old_comment>)[/dim]"
    - Otherwise, the existing comment (if any) is shown.
    """
    base_name: str = os.path.basename(file_path)
    file_ext: str = os.path.splitext(file_path)[1].lower().strip('.')
    
    if file_ext in ["mp3", "m4a"]:
        avg_cutoff: Optional[float] = get_cutoff(file_path)
        if avg_cutoff is None:
            return (base_name, "Unknown", "Unknown", "Could not determine cutoff")
        else:
            inferred_bitrate: Optional[int] = infer_bitrate_from_cutoff(file_ext, avg_cutoff)
            cutoff_str: str = f"{avg_cutoff:.1f} kHz"
            bitrate_str: str = f"{inferred_bitrate} kbps" if inferred_bitrate is not None else "Unknown"
            comment: str = ""
            if file_ext == "mp3" and write_flag and inferred_bitrate is not None:
                result: Optional[Tuple[str, str]] = update_id3_comment(file_path, inferred_bitrate)
                if result is not None:
                    old_comment, new_comment = result
                    if old_comment != new_comment and old_comment:
                        comment = f"[orange]{new_comment}[/orange] [dim](was {old_comment})[/dim]"
                    else:
                        comment = new_comment or old_comment
            else:
                comment = ""
            return (base_name, cutoff_str, bitrate_str, comment)
    elif file_ext == "flac":
        return (base_name, "Lossless", "N/A", "Graph drawn continuously")
    else:
        return (base_name, "Unsupported", "Unsupported", "")

async def main_async() -> None:
    """
    Asynchronous main function that uses a ProcessPoolExecutor to run file processing concurrently,
    displays a progress bar via tqdm, and finally outputs the results in a table using Rich.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Analyze audio files to determine a dominant cutoff frequency (via histogram) and infer the original bitrate."
    )
    parser.add_argument("path", help="Path to a file or directory")
    parser.add_argument("--write", action="store_true",
                        help="Write the inferred bitrate info to the ID3 tag of MP3 files")
    args = parser.parse_args()

    loop = asyncio.get_running_loop()
    results: List[Tuple[str, str, str, str]] = []

    if os.path.isfile(args.path):
        result: Tuple[str, str, str, str] = await loop.run_in_executor(None, process_file, args.path, args.write)
        results.append(result)
    elif os.path.isdir(args.path):
        file_list: List[str] = [
            os.path.join(args.path, filename)
            for filename in os.listdir(args.path)
            if os.path.isfile(os.path.join(args.path, filename))
        ]
        tasks: List[asyncio.Future] = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for file_path in file_list:
                tasks.append(loop.run_in_executor(executor, process_file, file_path, args.write))
            for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing files"):
                results.append(await future)
    else:
        print("File or directory not found:", args.path)
        sys.exit(1)

    # Create a Rich table for output.
    table = Table(title="Audio Analysis Results")
    table.add_column("💿 File", style="cyan", no_wrap=True)
    table.add_column("🔎 Dominant Cutoff", style="magenta")
    table.add_column("🎚 Inferred Bitrate", style="green", justify="right")
    table.add_column("💬 Comment", style="yellow")

    for row in results:
        file_name, cutoff, bitrate, comment = row
        table.add_row(file_name, cutoff, bitrate, comment)

    console = Console()
    console.print(table)

if __name__ == '__main__':
    asyncio.run(main_async())
