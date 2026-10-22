"""Train an LSTM next-note predictor from MIDI files."""
from pathlib import Path
import json
import numpy as np
import pretty_midi
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout

ROOT = Path(__file__).resolve().parents[1]
MIDI_DIR = ROOT / "data" / "midi"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
SEQUENCE_LENGTH = 16
EPOCHS = 35
BATCH_SIZE = 32

def extract_pitches(path: Path) -> list[int]:
    """Return note pitches from all non-empty instruments in one MIDI file."""
    try:
        midi = pretty_midi.PrettyMIDI(str(path))
        notes = [note for inst in midi.instruments for note in inst.notes]
        notes.sort(key=lambda note: (note.start, note.pitch))
        return [int(note.pitch) for note in notes]
    except Exception as exc:
        print(f"Skipping unreadable MIDI {path.name}: {exc}")
        return []

def main():
    midi_files = sorted([*MIDI_DIR.glob("*.mid"), *MIDI_DIR.glob("*.midi")])
    if not midi_files:
        raise SystemExit("No MIDI files found. Run: python src/create_demo_midi.py")

    pitch_stream = []
    for path in midi_files:
        pitches = extract_pitches(path)
        if pitches:
            pitch_stream.extend(pitches)
    if len(pitch_stream) <= SEQUENCE_LENGTH:
        raise SystemExit("Not enough notes. Add more MIDI files or create demo MIDI files.")

    # Use a vocabulary of pitches found in the collected music.
    vocabulary = sorted(set(pitch_stream))
    pitch_to_id = {pitch: idx for idx, pitch in enumerate(vocabulary)}
    encoded = [pitch_to_id[p] for p in pitch_stream]

    X, y = [], []
    for i in range(len(encoded) - SEQUENCE_LENGTH):
        X.append(encoded[i:i + SEQUENCE_LENGTH])
        y.append(encoded[i + SEQUENCE_LENGTH])
    X = np.asarray(X, dtype=np.int32)
    y = np.asarray(y, dtype=np.int32)

    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH,)),
        Embedding(input_dim=len(vocabulary), output_dim=64),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(128),
        Dense(len(vocabulary), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=1)

    model.save(MODEL_DIR / "music_lstm.keras")
    with open(MODEL_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "sequence_length": SEQUENCE_LENGTH,
            "vocabulary": vocabulary,
            "note_count": len(pitch_stream),
            "source_files": [p.name for p in midi_files],
        }, f, indent=2)
    print(f"Training complete. Model saved to: {MODEL_DIR / 'music_lstm.keras'}")

if __name__ == "__main__":
    main()
