# Task 3: Music Generation with AI

A beginner-friendly AI music-generation project using an LSTM (RNN) model. It reads MIDI files, converts notes into sequences, trains a model to predict the next note, generates new note sequences, and saves the result as a MIDI file.

## Project structure
- `data/midi/` — put your own `.mid` / `.midi` files here.
- `src/create_demo_midi.py` — creates a small demo MIDI dataset so you can test the pipeline.
- `src/train_model.py` — preprocesses MIDI files and trains the LSTM model.
- `src/generate_music.py` — generates new notes and writes a MIDI file.
- `requirements.txt` — Python dependencies.

## 1. Install Python packages
Use Python 3.10 or 3.11 if possible. Open a terminal in this project folder and run:

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow can be large and installation depends on your Python/OS version. If installation fails, use a supported Python version listed in TensorFlow's official installation guide.

## 2. Create demo MIDI files
```bash
python src/create_demo_midi.py
```
This writes a few short, original demo melodies to `data/midi/`. They are only for testing the code; for better results, use a larger, legally obtained MIDI collection (for example, a suitable subset of the MAESTRO dataset).

## 3. Train the model
```bash
python src/train_model.py
```
The model and training configuration are saved in `models/`.

## 4. Generate a new MIDI melody
```bash
python src/generate_music.py
```
The generated file is saved as `output/generated_music.mid`. Open it in a MIDI player or a digital audio workstation (DAW).

## 5. MIDI to audio (optional)
MIDI contains note events, not recorded sound. To listen to it as WAV, install a MIDI synthesizer such as FluidSynth and the optional Python package `pretty_midi`. Then you can use a DAW or a MIDI-to-audio renderer. This project saves MIDI by default so it does not depend on a particular soundfont.

## How it works
1. Collect MIDI files.
2. Read note pitches and durations.
3. Convert notes to integer sequences.
4. Create fixed-length training sequences.
5. Train an LSTM to predict the next note.
6. Sample predictions to create a new sequence.
7. Convert generated notes back into a MIDI file.

## Notes
- This is an educational baseline, not a professional music composer.
- A small demo dataset may produce repetitive music. More diverse MIDI files and longer training improve results.
- Use music you have permission to use.
