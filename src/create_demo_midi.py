"""Create three small original MIDI melodies for a quick pipeline test."""
from pathlib import Path
import pretty_midi

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "midi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MELODIES = {
    "demo_c_major.mid": [60, 62, 64, 67, 65, 64, 62, 60, 64, 67, 72, 67, 64, 62, 60, 60],
    "demo_g_major.mid": [67, 69, 71, 74, 72, 71, 69, 67, 71, 74, 79, 74, 71, 69, 67, 67],
    "demo_a_minor.mid": [69, 72, 76, 74, 72, 71, 69, 67, 69, 72, 74, 72, 71, 69, 69, 69],
}

for filename, pitches in MELODIES.items():
    midi = pretty_midi.PrettyMIDI(initial_tempo=110)
    instrument = pretty_midi.Instrument(program=0)
    start = 0.0
    for i, pitch in enumerate(pitches):
        duration = 0.38 if i % 4 else 0.55
        instrument.notes.append(
            pretty_midi.Note(velocity=90, pitch=pitch, start=start, end=start + duration)
        )
        start += duration
    midi.instruments.append(instrument)
    midi.write(str(OUT_DIR / filename))
    print(f"Created: {OUT_DIR / filename}")

print("Demo MIDI files created. You can now run src/train_model.py.")
