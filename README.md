# KBPS Analyzer

## Overiew

MP3 --> Spectral Analysis --> Cutoff Frequency --> Approximated Real Bitrate.

[Reference Aritcle](https://www.reddit.com/r/hiphopheads/comments/2t88ne/a_quick_guide_to_checking_the_real_bitrate_of/?sort=new)

## Usage

### With `uv` (Recommended)

Install `uv` Python manager by running:

```bash
# Linux / MacOS
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then run:

```bash
uv run --python 3.9 kbps-analyzer.py "path/to/folder/or/file"
```

### With Python 3.9

```bash
pip install mutagen numpy librosa matplotlib tqdm rich
python kbps-analyzer.py "path/to/folder/or/file" --write
```



## Notes

- Passing `--write` will write the analysed birtate to the file's comment tag in the following format:

    Comment tag: `My existing comment` -> `~128kbps My existing comment`

- This method is not 100% accurate, and non-standard encoded comment tags might be overwritten/changed.
