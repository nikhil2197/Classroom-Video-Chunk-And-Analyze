#!/usr/bin/env python3
"""
Split an Eleven Labs STT verbose_json transcript into fixed-interval segments.

Usage:
  python split_to_intervals.py input.json [--interval 10] [--output output.json]

Reads an Eleven Labs STT output JSON (with a "words" array containing word-level timestamps)
and writes a JSON array of segments, each with "start", "end", and concatenated "text"
for each interval (default 10 seconds).
"""
import os
import sys
import json
import argparse

def load_words(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'words' not in data:
        sys.stderr.write(f"Error: no 'words' key in {path}\n")
        sys.exit(1)
    return data['words']

def make_segments(words, interval_sec):
    # Determine total duration
    max_end = max(w.get('end', 0) for w in words)
    # Number of segments (cover until max_end)
    num = int(max_end // interval_sec) + 1
    segments = []
    for i in range(num):
        start_t = i * interval_sec
        end_t = start_t + interval_sec
        # Collect word texts whose start time is within this interval
        texts = []
        for w in words:
            if w.get('type') != 'word':
                continue
            st = w.get('start', 0)
            if st >= start_t and st < end_t:
                texts.append(w.get('text', ''))
        txt = ' '.join(texts).strip()
        segments.append({
            'start': float(start_t),
            'end': float(end_t),
            'text': txt
        })
    return segments

def main():
    parser = argparse.ArgumentParser(description="Split transcript into fixed intervals")
    parser.add_argument('input', help='Path to Eleven Labs transcript JSON')
    parser.add_argument('--interval', '-i', type=float, default=10.0,
                        help='Interval length in seconds (default: 10)')
    parser.add_argument('--output', '-o', help='Output JSON path (default: input_basename_intervals.json)')
    args = parser.parse_args()

    words = load_words(args.input)
    segments = make_segments(words, args.interval)

    # Determine output path; default to same directory as input
    base, _ = os.path.splitext(os.path.basename(args.input))
    suffix = f"_intervals_{int(args.interval)}s.json"
    if args.output:
        out_path = args.output
    else:
        input_dir = os.path.dirname(args.input) or "."
        out_path = os.path.join(input_dir, base + suffix)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(segments)} segments to {out_path}")

if __name__ == '__main__':
    main()