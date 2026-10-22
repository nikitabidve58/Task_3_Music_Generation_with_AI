"""Generate a new MIDI melody with the trained LSTM model."""
from pathlib import Path
import json
import numpy as np
import pretty_midi
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "music_lstm.keras"
CONFIG_PATH = ROOT / "models" / "config.json"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
GENERATED_NOTES = 80
TEMPERATURE = 0.9

def sample_with_temperature(probabilities, temperature=1.0):
    probabilities = np.asarray(probabilities).astype("float64")
    probabilities = np.log(np.maximum(probabilities, 1e-8)) / max(temperature, 1e-3)
    probabilities = np.exp(probabilities - np.max(probabilities))
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

def main():
    if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
        raise SystemExit("Trained model not found. Run: python src/train_model.py")

    model = tf.keras.models.load_model(MODEL_PATH)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sequence_length = int(config["sequence_length"])
    vocabulary = [int(p) for p in config["vocabulary"]]

    # Start with a random sequence of pitches from the learned vocabulary.
    seed = np.random.randint(0, len(vocabulary), size=sequence_length).tolist()
    generated = []
    for _ in range(GENERATED_NOTES):
        x = np.asarray([seed[-sequence_length:]], dtype=np.int32)
        probabilities = model.predict(x, verbose=0)[0]
        next_id = sample_with_temperature(probabilities, TEMPERATURE)
        generated.append(vocabulary[next_id])
        seed.append(next_id)

    midi = pretty_midi.PrettyMIDI(initial_tempo=110)
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    start = 0.0
    beat = 60.0 / 110.0
    for pitch in generated:
        duration = beat * 0.8
        instrument.notes.append(
            pretty_midi.Note(velocity=85, pitch=int(pitch), start=start, end=start + duration)
        )
        start += beat
    midi.instruments.append(instrument)
    output_path = OUTPUT_DIR / "generated_music.mid"
    midi.write(str(output_path))
    print(f"Generated {len(generated)} notes.")
    print(f"Saved MIDI file: {output_path}")

if __name__ == "__main__":
    main()
