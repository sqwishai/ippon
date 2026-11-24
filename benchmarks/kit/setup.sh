#!/bin/sh
# The benchmark kit's environments, llama.cpp, Ollama and models: the models at the revisions timed
# (benchmarks/README.md), every other runtime at the release the kit pins (llama.cpp v0.8.0, whose
# macOS binary is its build b11960; mlx-vlm 0.7.10; oMLX v0.7.2; Ollama v0.40.13). Needs uv
# (https://docs.astral.sh/uv/) and, for the models, about 250 GB of disk.
#
#   sh setup.sh envs            # one Python environment per runtime, under ../envs
#   sh setup.sh llama           # llama.cpp v0.8.0's official macOS release (b11960), under ../bin
#   sh setup.sh ollama [DIR]    # Ollama v0.40.13's official macOS release, under ../bin, its MLX builds in DIR/ollama
#   sh setup.sh models [DIR]    # every model at the revision timed, under DIR (default ~/models)
#   sh setup.sh rivals [DIR]    # the other runtimes of the same models, at their releases, and their own builds
#   sh setup.sh                 # all five
#
# Each runtime gets its own environment with the versions its creator pins or documents, so none of them runs on
# another's dependencies: ippon (this repository, a regular install), mlx-vlm (the community Clef-Flash build's
# loader runs on it), Liquid's, Cloudflare's and vllm-sr's transformers code, Perplexity's pplx_decider (its
# pyproject's pins) and Torchcast's server (its card's torch and transformers, without its CUDA kernels).
set -eu
KIT=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$KIT/../.." && pwd)
ENVS=$KIT/../envs
BIN=$KIT/../bin
RIVALS=$KIT/../rivals
OMLX_COMMIT=23b671f64f0a64f532e38d692a6c480ea704418d  # jundot/omlx v0.7.2
LLAMA=b11960  # v0.8.0's build, the same commit, 8642541; the first release that reads d1-2B and d1-Omni-2B
LLAMA_SHA256=58e5a5e1d7c2bf66cedd9c29e0323f3e570660a70a1e2156d32eed08f4cc1e3c
OLLAMA=v0.40.13
OLLAMA_SHA256=bae29252232b9e1358cf5801a7d65a07442af18fb5467cb68c7fc3711f59c511  # ollama-darwin.tgz
OLLAMA_PORT=11439  # the pulling server's, on 127.0.0.1 only

env_() {  # name, packages...
  name=$1; shift
  echo "== envs/$name: $*"
  uv venv -q --allow-existing --python 3.12 --python-preference only-managed "$ENVS/$name"
  uv pip install -q --python "$ENVS/$name/bin/python" "$@"
}

envs() {
  command -v uv > /dev/null || { echo "install uv first: https://docs.astral.sh/uv/"; exit 1; }
  command -v cargo > /dev/null || { echo "install Rust's cargo first (ippon builds its native library from source): curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal"; exit 1; }
  env_ harness "huggingface_hub==2.2.0"  # the HTTP clients use the standard library; hf downloads the models
  env_ ippon "ippon @ file://$ROOT"
  env_ mlxvlm "mlx==0.32.4" "transformers==5.19.0" "mlx-vlm==0.7.10"  # it needs mlx>=0.32.4, transformers>=5.14.0
  env_ d1 "torch==2.14.1" "torchvision==0.29.1" "transformers==5.19.0" "pillow==13.0.0" "torchcodec==0.17.0"  # d1-2B's card's packages, and the video decoder its processor uses
  # Cloudflare's and vllm-sr's code read through transformers' Qwen3VLProcessor, which needs torchvision and Pillow
  env_ clef "torch==2.14.1" "torchvision==0.29.1" "transformers==5.17.0" "accelerate==1.15.0" "einops==0.8.2" \
    "huggingface_hub==1.33.0" "pillow==13.0.0"
  env_ pplx "torch==2.14.0" "torchvision==0.29.0" "transformers==5.17.0" "safetensors==0.8.0" "pillow==12.3.0" \
    "huggingface_hub==1.31.0"
  env_ vllmsr "torch==2.14.1" "torchvision==0.29.1" "transformers==5.17.0" "accelerate==1.15.0" "einops==0.8.2" \
    "huggingface_hub==1.33.0" "pillow==13.0.0" "opencv-python-headless==5.0.0.93"  # OpenCV: its video decoder
  # Base Labs' runtime for elephant-1: the checkpoint's constraints.txt pins (torch: the Mac's wheel of its image's)
  env_ elephant "torch==2.8.0" "torchvision==0.23.0" "transformers==5.19.0" "accelerate==1.15.0" \
    "safetensors==0.8.0" "pydantic==2.14.1" "fastapi==0.143.1" "uvicorn==0.54.0" "pillow==11.0.0" "httpx"
  env_ torchcast "torch==2.9.1" "torchvision==0.24.1" "transformers==5.8.1" "accelerate==1.15.0" \
    "safetensors==0.8.0" "pillow==13.0.0"
  # Jeff's server (v1.3.0, installed by `rivals` from its checkout): its requirements.txt pins, torch's Mac wheel
  env_ jeff "torch==2.13.0" "transformers==5.17.0" "safetensors==0.8.0" "huggingface_hub==1.32.0" "accelerate==1.15.0"
  # Fastino's glide server, shipped in each GLiDE checkpoint: its requirements.txt pins, without triton and
  # flash-linear-attention, which need CUDA (the package then takes its PyTorch path)
  env_ glide "torch==2.14.0" "transformers==5.17.0" "tokenizers==0.23.2" "safetensors==0.8.0" \
    "huggingface_hub==1.33.0" "jinja2==3.1.6" "numpy==2.5.3" "pillow==12.3.0" "fastapi==0.142.2" "uvicorn==0.54.0" \
    "anyio==4.15.1"
  "$ENVS/ippon/bin/python" -I -c "import ippon.machine as m; print('ippon', m.revision())"
}

llama() {
  mkdir -p "$BIN"
  archive=$BIN/llama-$LLAMA-bin-macos-arm64.tar.gz
  [ -f "$archive" ] || curl -fL -o "$archive" \
    "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA/llama-$LLAMA-bin-macos-arm64.tar.gz"
  echo "$LLAMA_SHA256  $archive" | shasum -a 256 -c -
  mkdir -p "$BIN/llama-$LLAMA"
  tar -xzf "$archive" -C "$BIN/llama-$LLAMA" --strip-components 1
  xattr -dr com.apple.quarantine "$BIN/llama-$LLAMA" 2> /dev/null || true
  "$BIN/llama-$LLAMA/llama-server" --version
}

ollama() {  # Ollama's release and its own MLX builds, pulled by a server of its own into DIR/ollama (OLLAMA_MODELS,
            # never ~/.ollama), stopped when they are in; a build counts only as the registry digest timed
  MODELS=${1:-$HOME/models}
  store=$MODELS/ollama
  archive=$BIN/ollama-$OLLAMA-darwin.tgz
  mkdir -p "$BIN/ollama-$OLLAMA" "$store"
  [ -f "$archive" ] || curl -fL -o "$archive" \
    "https://github.com/ollama/ollama/releases/download/$OLLAMA/ollama-darwin.tgz"
  echo "$OLLAMA_SHA256  $archive" | shasum -a 256 -c -
  tar -xzf "$archive" -C "$BIN/ollama-$OLLAMA"
  xattr -dr com.apple.quarantine "$BIN/ollama-$OLLAMA" 2> /dev/null || true
  OLLAMA_MODELS=$store OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT OLLAMA_NOPRUNE=1 \
    "$BIN/ollama-$OLLAMA/ollama" serve > "$store/pull.log" 2>&1 &
  server=$!
  trap 'kill $server 2> /dev/null' EXIT
  until curl -fs "http://127.0.0.1:$OLLAMA_PORT/api/version"; do
    kill -0 "$server" 2> /dev/null || { echo "ollama serve exited; see $store/pull.log"; exit 1; }
    sleep 1
  done
  echo
  # each tag's weights (ollama_weights.py): Ollama republishes a tag when only its params change, as it did for d3's
  while read -r tag digest; do  # its MLX packages (default d1 tags resolve to their BF16 packages)
    echo "== ollama $tag"
    OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT "$BIN/ollama-$OLLAMA/ollama" pull "$tag" > /dev/null
    m=$store/manifests/registry.ollama.ai/library/${tag%%:*}/${tag#*:}
    [ "$("$ENVS/harness/bin/python" -I "$KIT/ollama_weights.py" "$m")" = "$digest" ] \
      || { echo "$tag no longer has the weights timed ($digest)"; exit 1; }
  done << EOF
clef-flash:9b-mxfp8 03aa226ac955d0d2fc87d8acec5f81607d2c569dbc0a0570f823b91d26977c24
d3:4b-mxfp8 30a72d50d0b2d538a69c137f8310a1f7cc6abe84da06b9d449168cafd73ff174
d3:9b-mxfp8 2f8fcf58d27996b3f497acf6f9b18399ab162226732ce4353258332d4baeeced
d1:2b 7e85d0650974abc88333fe0c38252face1c1cdaa70f46b3322d92004b1d49d38
d1:2b-mlx-bf16 7e85d0650974abc88333fe0c38252face1c1cdaa70f46b3322d92004b1d49d38
d1-omni:2b 23253d06837142db979b5ae4d86e09dc262bcdce5edf28ae8dbf65566644af22
d1-omni:2b-mlx-bf16 23253d06837142db979b5ae4d86e09dc262bcdce5edf28ae8dbf65566644af22
d1:2b-mxfp8 4f3cc45b893dae1ce66ed2adedf165b7279c2183e7bafd05120cfb1d941f4adb
d1-omni:2b-mxfp8 974c818a89ecdd687e0113e0d3a65231e9dcfa97f126361b9d55e4788f355f46
EOF
  kill "$server"
  wait "$server" 2> /dev/null || true
  trap - EXIT
  "$BIN/ollama-$OLLAMA/ollama" --version 2>&1 | tail -1
}

rivals() {  # other runtimes of the models the kit times with a wide following, each at its release, in an
            # environment of its own (their code read first: it runs with this account's access)
  MODELS=${1:-$HOME/models}
  if xcrun -f metal > /dev/null 2>&1; then  # oMLX builds its kernels with Xcode's Metal toolchain
    if [ ! -d "$RIVALS/omlx/.git" ]; then  # jundot/omlx: an MLX server with decision models, its native kernels built
      mkdir -p "$RIVALS" && git clone -q https://github.com/jundot/omlx.git "$RIVALS/omlx"
    fi
    git -C "$RIVALS/omlx" checkout -q "$OMLX_COMMIT"
    uv venv -q --allow-existing --seed --python 3.12 --python-preference only-managed "$ENVS/omlx"
    make -C "$RIVALS/omlx" install PYTHON="$ENVS/omlx/bin/python"
    mkdir -p "$MODELS/omlx"  # what oMLX serves: the models of `models` it runs, linked (it lists every folder it has)
    for m in clef-flash-mlx-8bit d3-mini d3-flash; do ln -sfn "$MODELS/$m" "$MODELS/omlx/$m"; done
  else
    echo "oMLX skipped: it needs Xcode and its Metal toolchain (xcodebuild -downloadComponent MetalToolchain)"
  fi
  # OneJev's qev, the runtime elephant-1's setup.sh installs: its pinned commit with the checkpoint's serve.patch
  [ -d "$KIT/../makers/onejev/.git" ] || git clone -q https://github.com/OmniJev/OneJev.git "$KIT/../makers/onejev"
  git -C "$KIT/../makers/onejev" checkout -q -f 81ce62f1597c91e46767d4d02d6ac2e18534fe94
  git -C "$KIT/../makers/onejev" clean -fdq
  git -C "$KIT/../makers/onejev" apply "$MODELS/elephant-1/serve.patch"
  # Jeff's serving code at v1.3.0, the release its model card names, into its environment
  [ -d "$RIVALS/jeff-v1.3.0/.git" ] || git clone -q https://github.com/j-geuter/Jeff "$RIVALS/jeff-v1.3.0"
  git -C "$RIVALS/jeff-v1.3.0" checkout -q 2c446ca3249f31c83cc4a230f3aed605858da330
  uv pip install -q --python "$ENVS/jeff/bin/python" --no-deps "$RIVALS/jeff-v1.3.0"
}

get() {  # repo, revision, folder, [files...]
  repo=$1 rev=$2 folder=$3; shift 3
  echo "== $repo@$rev -> $MODELS/$folder"
  "$ENVS/harness/bin/hf" download "$repo" "$@" --revision "$rev" --local-dir "$MODELS/$folder" > /dev/null
}

models() {
  MODELS=${1:-$HOME/models}
  # full precision, as the creators publish them
  get LiquidAI/d1-2B cfe5760ff5d5998f91d9ac56bd4b2dbd96996d41 d1-2B
  get LiquidAI/d1-Omni-2B c8d48c08dada0191c6b362820f055cfb49c99426 d1-Omni-2B
  get Cloudflare/clef-flash 8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3 clef-flash
  get perplexity-ai/pplx-decider-flash-v1.3 d06fb3f2d97e0a52325a0d463c450cf0338f44fe pplx-decider-flash-v1.3 \
    --exclude "locate-span/*"
  get vllm-sr/d3-mini bd0090dd758d32ba59ef5c749910a8e87fc4391f d3-mini  # its model files are c5aa751c's
  get vllm-sr/d3-flash 13dc1cc8ef23caefa765907e1d764751e5629f7d d3-flash
  get baselabs/elephant-1 b3f8a93462fb1051a9de031e94a7ff380e254072 elephant-1
  get Sqwish/Kettei-Core 5a9821633c8bb15b92b677d368083ec04720d164 kettei-core
  get Sqwish/Kettei-Dot a8ef982f8736cbac710899810f87bb53668be6d1 kettei-dot
  get torchcast-ai/torchcast-decision-4b 492bfb19662f3c6e601cc9c4bd2911ac95609ed4 torchcast-decision-4b  # v1.3.1
  get jgeuter/Jeff-1.1-Medium 540c40924dd67c715d3367309ba6744bac754b00 jeff-1.1-medium --exclude "adapter/*"
  get fastino/GLiDE-2B 00f122c120cdeed82284106cb77d9bab5a50ad16 glide-2b
  get fastino/GLiDE-9B ab54df5acba0eebf7df45560ffdaacd64e32df3b glide-9b
  # the community's MLX build of Clef-Flash
  get mlx-community/clef-flash-8bit 5a195dc147926b63b1d139691ed4480ca9f92c0e clef-flash-mlx-8bit
  # the creators' Q8_0 GGUF files, which llama.cpp and ippon both read
  get ggml-org/Clef-Flash-GGUF 8aa35d1cc7e141f3771e564eeb630a501597c681 Clef-Flash-GGUF Clef-Flash-Q8_0.gguf
  get LiquidAI/d1-2B-GGUF e968f6575f0d4fad928312d7af15cad90fd365c0 d1-2B-GGUF d1-2B-Q8_0.gguf mmproj-d1-2B-Q8_0.gguf
  get LiquidAI/d1-Omni-2B-GGUF c53ae0b1f92b17065e9158d34019c9ca1ea5bcf8 d1-Omni-2B-GGUF d1-Omni-2B-Q8_0.gguf \
    mmproj-d1-Omni-2B-Q8_0.gguf
  get h2oai/h2o-lightning-4b-gguf e19edebcafb33e1154e6260d695935ab0984b0e8 h2o-lightning-4b-gguf \
    h2o-lightning-4b-Q8_0.gguf mmproj-lightning-f16.gguf
}

case ${1:-all} in
  envs) envs ;;
  llama) llama ;;
  models) models "${2:-}" ;;
  ollama) ollama "${2:-}" ;;
  rivals) rivals "${2:-}" ;;
  all) envs; llama; ollama; models; rivals ;;
  *) echo "usage: sh setup.sh [envs|llama|ollama [DIR]|models [DIR]|rivals [DIR]]"; exit 2 ;;
esac
