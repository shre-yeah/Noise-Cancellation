import torchaudio
torchaudio.set_audio_backend("soundfile")

from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file

model, df_state, _ = init_df()

audio_path = download_file(
    "https://github.com/Rikorose/DeepFilterNet/raw/e031053/assets/noise_freesound_2530.wav",
    download_dir="."
)
audio, _ = load_audio(audio_path, sr=df_state.sr())
enhanced_audio = enhance(model, df_state, audio)
save_audio("enhanced2.wav", enhanced_audio, df_state.sr())

print("✅ Noise reduction complete! Check the file: enhanced.wav")
