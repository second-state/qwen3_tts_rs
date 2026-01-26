# Demo Audio Files

These audio files were generated using the Qwen3 TTS Rust inference engine with the `Qwen3-TTS-12Hz-0.6B-CustomVoice` model.

## Vivian (English)

```bash
cargo run --example tts_demo --release -- \
  models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  "The morning sun cast golden light across the quiet valley, where birds sang melodies that echoed through the mist." \
  Vivian \
  english
mv output.wav demos/vivian_english.wav
```

## Vivian (Chinese)

```bash
cargo run --example tts_demo --release -- \
  models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  "清晨的阳光洒满了宁静的山谷，鸟儿在薄雾中歌唱着悠扬的旋律。" \
  Vivian \
  chinese
mv output.wav demos/vivian_chinese.wav
```

## Ryan (English)

```bash
cargo run --example tts_demo --release -- \
  models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  "Technology continues to reshape how we live and work, bringing both new opportunities and unexpected challenges to communities around the world." \
  Ryan \
  english
mv output.wav demos/ryan_english.wav
```

## Ryan (Chinese)

```bash
cargo run --example tts_demo --release -- \
  models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  "科技不断改变着我们的生活和工作方式，为世界各地的社区带来了新的机遇和意想不到的挑战。" \
  Ryan \
  chinese
mv output.wav demos/ryan_chinese.wav
```
