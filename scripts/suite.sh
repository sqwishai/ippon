#!/bin/sh
# ippon's suite on this Mac (docs/suite.md):
#   sh scripts/suite.sh setup            the ippon environment, and mlx-vlm's for the comparison arms
#   sh scripts/suite.sh models DIR       download the suite's Hub models into DIR
#   sh scripts/suite.sh run DIR [...]    run the suite into suite-<chip>/ (extra options go to ippon suite)
set -e
cd "$(dirname "$0")/.."
ARMS="$HOME/.ippon-arms"
MLX_VLM="mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git@4623cf9d2633c5ee2b1ac125eeae51423447c41f"
case "$1" in
setup)
  command -v uv > /dev/null || { echo "uv is needed: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
  uv sync
  [ -x "$ARMS/mlxvlm/bin/python" ] || uv venv -q --python 3.12 "$ARMS/mlxvlm"
  VIRTUAL_ENV="$ARMS/mlxvlm" uv pip install -q "mlx==0.32.4" "$MLX_VLM"
  uv run ippon doctor || true
  ;;
models)
  dir=${2:?the folder to put the models in}
  mkdir -p "$dir"
  uv run python -c "from ippon.suite import SUITE; print('\n'.join(f'{m.hub.replace(\"@\", \" \")} {m.directory}' for m in SUITE if m.hub))" |
    while read -r repo revision folder; do
      echo "== $repo@$revision"
      uv run hf download "$repo" --revision "$revision" --local-dir "$dir/$folder" --exclude "*.gguf"
    done
  ;;
run)
  dir=${2:?the folder holding the models}
  shift 2
  chip=$(sysctl -n machdep.cpu.brand_string | tr 'A-Z ' 'a-z-')
  py="$ARMS/mlxvlm/bin/python"
  arms=""
  [ -x "$py" ] && arms="--arm community-clef=$py"
  # shellcheck disable=SC2086
  uv run ippon suite --models "$dir" --out "suite-$chip" $arms "$@"
  ;;
*)
  sed -n 2,5p "$0"
  exit 1
  ;;
esac
