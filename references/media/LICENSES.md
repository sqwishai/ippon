# Media licences

Every file in `images/` and `audio/`, and every file in `videos/` but one, is in the public domain or dedicated CC0
1.0. The one exception, vllm-sr's own example clip, is Apache-2.0 (below). The licence of each photograph, recording
and clip comes from its Wikimedia Commons file page (its licence template, checked 23 and 29 October 2026). The
requests that use these files, with the sha256 of each file, are in `../requests/media-d1.json`, `../requests/media-d1-audio.json`,
`../requests/media-qwen.json` and `../requests/media-video.json`.

## Photographs and scans (Wikimedia Commons, public domain)

Each file is the Commons thumbnail at the width given, downloaded with curl from `upload.wikimedia.org` as Wikimedia's
thumbnailer serves it (resized from the original and re-encoded as JPEG, with a small embedded sRGB ICC profile on
some). The files were not changed afterwards. A public-domain work may be resized and redistributed freely.

| File | Size | Commons file page | Author | Licence |
|---|---|---|---|---|
| `earth-apollo17.jpg` | 960x961, 267 KB | [The Earth seen from Apollo 17.jpg](https://commons.wikimedia.org/wiki/File:The_Earth_seen_from_Apollo_17.jpg) (original 3000x3002) | NASA, Apollo 17 crew (Harrison Schmitt or Ron Evans), 1972 | Public domain: `PD-USGov-NASA`, a work of NASA |
| `aldrin-apollo11.jpg` | 1280x1288, 448 KB | [Aldrin Apollo 11 original.jpg](https://commons.wikimedia.org/wiki/File:Aldrin_Apollo_11_original.jpg) (original 3912x3936, NASA AS11-40-5903) | Neil Armstrong, NASA, 1969 | Public domain: `PD-USGov-NASA`, a work of NASA |
| `migrant-mother.jpg` | 960x1248 grayscale, 255 KB | [Lange-MigrantMother02.jpg](https://commons.wikimedia.org/wiki/File:Lange-MigrantMother02.jpg) (original 6205x8066) | Dorothea Lange, Farm Security Administration, 1936 | Public domain: `PD-USGov-FSA`, a work of the US federal government (also `PD-old-auto`, author died 1965) |
| `wright-first-flight.jpg` | 1280x830, 180 KB | [First flight2.jpg](https://commons.wikimedia.org/wiki/File:First_flight2.jpg) (original 6603x4280, Library of Congress ppprs.00626) | John T. Daniels, 1903 | Public domain: `PD-US-expired`, published in the United States before 1931 |
| `lincoln-1863.jpg` | 500x644 grayscale, 72 KB | [Abraham Lincoln O-77 matte collodion print.jpg](https://commons.wikimedia.org/wiki/File:Abraham_Lincoln_O-77_matte_collodion_print.jpg) (original 2200x2835) | Alexander Gardner (1821-1882), 1863 | Public domain: `PD-old-auto-1923` (author died 1882) and `PD-scan` |
| `bison.jpg` | 500x326, 63 KB | [American bison k5680-1.jpg](https://commons.wikimedia.org/wiki/File:American_bison_k5680-1.jpg) (original 2700x1761) | Jack Dykinga, USDA Agricultural Research Service (image K5680-1) | Public domain: `PD-USGov-USDA-ARS`, a work of the USDA |
| `vegetables.jpg` | 960x1450, 354 KB | [Vegetables.jpg](https://commons.wikimedia.org/wiki/File:Vegetables.jpg) (original 1787x2700) | Scott Bauer, USDA Agricultural Research Service (image K7246-9) | Public domain: `PD-USGov-USDA-ARS`, a work of the USDA |
| `constitution-page1.jpg` | 1280x1549, 694 KB | [Constitution of the United States - DPLA - 9ca804144bd5965e992ae3528bc3c6a3 (page 1).jpg](https://commons.wikimedia.org/wiki/File:Constitution_of_the_United_States_-_DPLA_-_9ca804144bd5965e992ae3528bc3c6a3_(page_1).jpg) (original 7258x8785, National Archives via DPLA) | Constitutional Convention, 1787; scan by the National Archives | Public domain: `PD-USGov` |

## Synthetic images (this repository, CC0 1.0)

`make_synthetic.py` draws these images with Pillow only, from seed 20261023. With Pillow 12.3.0,
`python -I make_synthetic.py` regenerates them byte for byte. Text is set in Pillow's embedded Aileron Regular (by Sora Sagano,
[dotcolon.net/fonts/aileron](https://dotcolon.net/fonts/aileron), CC0 1.0). The images are dedicated to the public
domain under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

| File | Size | What it covers |
|---|---|---|
| `chart-bars-800x600.png` | 800x600 RGB, 18 KB | a bar chart with text |
| `table-invoice-1024x512.png` | 1024x512 RGB, 40 KB | a table, screenshot-like, with an area of exactly twice d1's per-crop budget |
| `solid-red-1x1.png` | 1x1 RGB | one pixel, below the 64x1024-pixel floor |
| `solid-blue-24x24.png` | 24x24 RGB | a solid colour below the floor |
| `small-circle-100x80.jpg` | 100x80 RGB JPEG | below the floor |
| `gradient-512x384.png` | 512x384 RGB | a gradient, multiples of 32 inside the budget |
| `squares-512x512.png` | 512x512 RGB | exactly the budget |
| `circles-720x720.png` | 720x720 RGB | just under d1's tiling line (720/32 = 22.5 rounds to even) |
| `circles-725x725.png` | 725x725 RGB | just over the tiling line |
| `dashboard-1000x800.jpg` | 1000x800 RGB JPEG | aspect 5:4, a tie between tile grids |
| `shapes-4000x3000.jpg` | 4000x3000 RGB JPEG, 281 KB | 4,000x3,000 (tiled) |
| `receipt-400x3200.png` | 400x3200 RGB, 127 KB | aspect 1:8 |
| `banner-3000x60.png` | 3000x60 RGB | aspect 50:1 |
| `gray-gradient-640x480.png` | 640x480 L | grayscale PNG |
| `palette-stripes-600x400.png` | 600x400 P | palette PNG with a transparent index |
| `rgba-circle-512x384.png` | 512x384 RGBA | RGBA, transparent pixels storing magenta |
| `arrow-exif6.jpg` | 640x480 stored, 480x640 shown | EXIF orientation 6 |
| `cmyk-blocks-640x480.jpg` | 640x480 CMYK JPEG | CMYK |
| `gray16-gradient-512x256.png` | 512x256 I;16 | 16-bit grayscale PNG |

`../requests/make_media_d1.py` writes `../requests/media-d1.json` from these files, with the same output each time.
Run `python3 -I make_media_d1.py > media-d1.json` in `../requests/`.

## Speech clips (Wikimedia Commons, public domain and CC0)

Each clip is an excerpt of a recording on Wikimedia Commons, made by `make_audio.py`. The script takes the file as
Commons serves it (checked by sha256). ffmpeg 9.0.1 decodes it, mixes it down to mono and resamples it to 16 kHz with
its default resampler. The script then cuts it at whole samples and writes it as 16 kHz mono WAV, PCM16 or float32. A public-domain or CC0 recording may be
cut, converted and redistributed freely.

| File | Excerpt | Commons file page | Author | Licence |
|---|---|---|---|---|
| `armstrong-small-step.wav` | all 24.1 s, PCM16 | [Armstrong Small Step.ogg](https://commons.wikimedia.org/wiki/File:Armstrong_Small_Step.ogg) | Neil Armstrong, NASA, 21 July 1969 | Public domain: `PD-USGov-NASA`, a work of NASA |
| `kennedy-rice-30s.wav` | 34 s to 64 s (exactly 30 s), PCM16 | [Jfk rice university we choose to go to the moon.ogg](https://commons.wikimedia.org/wiki/File:Jfk_rice_university_we_choose_to_go_to_the_moon.ogg) | John F. Kennedy, 12 September 1962 (John F. Kennedy Presidential Library and Museum) | Public domain: `PD-USGov`, a work of the US federal government |
| `apollo13-problem-15s.wav` | 0 s to 15 s, float32 | [Apollo13-wehaveaproblem.ogg](https://commons.wikimedia.org/wiki/File:Apollo13-wehaveaproblem.ogg) | Apollo 13 crew and mission control, NASA, 14 April 1970 | Public domain: `PD-USGov-NASA`, a work of NASA |
| `keats-chapman-40s.wav` | 13 s to 53 s (40 s, of which the model reads 30), PCM16 | [On-First-Looking-Into-Chapman's-Homer-LibriVox-Vorbis.ogg](https://commons.wikimedia.org/wiki/File:On-First-Looking-Into-Chapman%E2%80%99s-Homer-LibriVox-Vorbis.ogg) | John Keats (1816), read by Leonard Wilson for LibriVox, 2010 | CC0 1.0: `cc-zero` |
| `poe-bells-5s.wav` | 4 s to 9 s, PCM16 | [LibriVox - The Bells - Alan Davis-Drake.ogg](https://commons.wikimedia.org/wiki/File:LibriVox_-_The_Bells_-_Alan_Davis-Drake.ogg) | Edgar Allan Poe (1849), read by Alan Davis-Drake for LibriVox | Public domain: `LibriVox public domain` (the recording) and `PD-old` (the poem) |
| `poe-bells-0.3s.wav` | 20 s to 20.3 s (the model pads it to 0.5 s), float32 | the same | the same | the same |
| `poe-bells-5s-22k-stereo.wav` | 4 s to 9 s, linearly interpolated to 22,050 Hz, the right channel at half the left's level, PCM16 | the same | the same | the same |

## Synthetic clips (this repository, CC0 1.0)

`make_audio.py` also writes these clips, and they are dedicated to the public domain under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

| File | What it covers |
|---|---|
| `silence-1s.wav` | 1 s of exact zeros, PCM16 |
| `tone-440hz-2s.wav` | 2 s of a 440 Hz sine at a quarter of full scale, float32 |

`../requests/make_media_d1_audio.py` writes `../requests/media-d1-audio.json` from these files, with the same
output each time. Run `python3 -I make_media_d1_audio.py > media-d1-audio.json` in `../requests/`.

## Video clips (Wikimedia Commons, CC0 1.0)

The clip is Commons' own 240p VP9 WebM transcode of the file, downloaded with curl from `upload.wikimedia.org` as
Commons serves it. It was not changed afterwards, and it keeps its Opus audio track, which the models do not read.

| File | Size | Commons file page | Author | Licence |
|---|---|---|---|---|
| `storefront-mazatlan-240p.webm` | 426x240 VP9 at 355/12 fps, 8.9 s, 435 KB (the original is 1280x720) | [Cuidado con el Perro, Gran Plaza Mazatlán, 20 de junio de 2026.webm](https://commons.wikimedia.org/wiki/File:Cuidado_con_el_Perro,_Gran_Plaza_Mazatl%C3%A1n,_20_de_junio_de_2026.webm) | Roedor Pacheco, 20 June 2026 | CC0 1.0: `self\|cc-zero` |

## vllm-sr's example clip (Apache-2.0)

`d3-example-square.mp4` (640x360 MPEG-4 Part 2 at 8 fps, 4 s, 26 KB) is `assets/example-video.mp4` of vllm-sr's d3
packages ([vllm-sr/d3-flash](https://huggingface.co/vllm-sr/d3-flash), revision
13dc1cc8ef23caefa765907e1d764751e5629f7d, the same file in vllm-sr/d3-mini), unchanged. It is the clip that the
video example on their card asks about. It is licensed under the Apache License 2.0 (this repository's `LICENSE`).
The NOTICE of its package reads "d3-flash (Decision 3.0), Copyright 2026 vLLM Semantic Router Team. Built on Qwen/Qwen3.5-9B
(Apache-2.0)."

## Synthetic video clips (this repository, CC0 1.0)

`make_videos.py` draws these clips frame by frame with numpy, and ffmpeg 9.0.1 encodes them from raw RGB frames.
With the same ffmpeg, `python -I make_videos.py` regenerates them byte for byte. They are dedicated to the public
domain under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

| File | What it covers |
|---|---|
| `ball-h264-320x240-24fps-3s.mp4` | H.264 in MP4, 6 of 72 frames read, 240 rounds up to 256 |
| `bar-h264-1280x720-29.97fps-20s.mp4` | 30000/1001 fps, 32 of 600 frames read (2 a second would be 40), frames over the 200,704-pixel budget, read at 576x320 |
| `square-vp9-160x120-10fps-2.5s.webm` | VP9 in WebM, 5 frames read, an odd number, so the last is read twice |
| `flash-h264-64x48-5fps-1s.mp4` | 5 frames, where 2 a second would give 2 and the floor is 4 |
| `tiny-h264-24x16-4fps-3s.mp4` | frames under 32 pixels a side, scaled up before rounding |
| `portrait-mjpeg-180x320-15fps-4s.mov` | Motion JPEG in QuickTime, portrait |
| `rings-mpeg4-200x200-12fps-6s.mkv` | MPEG-4 Part 2 in Matroska |

`../requests/make_media_video.py` writes `../requests/media-video.json` from these files and the images, with the
same output each time. Run `python3 make_media_video.py > media-video.json` in `../requests/`.
