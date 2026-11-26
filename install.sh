#!/bin/sh
# Install ippon on a Mac with Apple silicon:
#
#   curl -fsSL https://sqwish.ai/ippon/install.sh | sh
#   curl -fsSL https://sqwish.ai/ippon/install.sh | sh -s -- kettei-dot   (that model)
#
# uv installs ippon as a tool of its own (`uv tool install`), so nothing else on the Mac changes. It uses the uv and
# the Python (3.11 or newer) the Mac already has, and installs either only when it is missing or too old. Running it
# again upgrades ippon; `uv tool uninstall ippon` removes it. IPPON_PACKAGE installs something other than PyPI's ippon: "ippon==0.1.0", or a wheel's path or URL.
# When uv's bin directory is not on this terminal's PATH, ippon is linked into one that is and that
# this user can write to (~/.local/bin, ~/bin, /opt/homebrew/bin or /usr/local/bin), so it works at once.
# UV_NO_MODIFY_PATH=1 leaves the shell's PATH and those directories alone, as it does for uv's own installer.
set -eu

UV_VERSION=0.12.12    # the uv installed when the Mac has none, or one older than UV_MIN
UV_MIN=0.5.0          # the oldest uv whose `tool install` takes a Python version range
PYTHONS='>=3.11'  # the Pythons ippon supports: one already on the Mac is used, else uv installs one
package=${IPPON_PACKAGE:-ippon}

fail() {
  printf 'ippon install: %s\n' "$*" >&2
  exit 1
}

[ "$(uname -s)" = Darwin ] || fail "ippon runs on macOS, and this is $(uname -s)."
if [ "$(uname -m)" != arm64 ]; then
  if [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = 1 ]; then
    fail "this terminal runs under Rosetta. Open a terminal that runs natively on Apple silicon and run this again."
  fi
  fail "ippon needs a Mac with Apple silicon (M1 or later), and this Mac has an Intel processor."
fi

# whether version $1 (as `uv --version` prints it) is at least $2
at_least() {
  [ "$(printf '%s\n%s\n' "$2" "$1" | sort -t. -k1,1n -k2,2n -k3,3n | head -n 1)" = "$2" ]
}

# whether the program at $1 runs natively on Apple silicon: a Mac moved over from an Intel one can keep an x86_64 uv
# or Python, which runs under Rosetta and installs an x86_64 ippon that MLX can't load
# (`file`, which every Mac has: `lipo` asks to install the developer tools on a Mac without them)
native() {
  case "$(file -bL "$1" 2>/dev/null)" in
    *arm64*) return 0 ;;
    *Mach-O*) return 1 ;;
    *) return 0 ;; # a script that starts it, or a file `file` can't read: taken as it is
  esac
}

# where uv's installer puts uv: UV_INSTALL_DIR, else XDG_BIN_HOME, else ~/.local/bin, as its installer chooses
uv_installed() {
  for dir in "${UV_INSTALL_DIR:-}" "${XDG_BIN_HOME:-}" "$HOME/.local/bin"; do
    if [ -n "$dir" ] && [ -x "$dir/uv" ]; then
      printf '%s\n' "$dir/uv"
      return
    fi
  done
}

uv=$(command -v uv || true)
if [ -z "$uv" ]; then
  uv=$(uv_installed)
fi
if [ -n "$uv" ]; then
  have=$("$uv" --version 2>/dev/null | awk '{print $2}')
  if [ -n "$have" ] && ! native "$uv"; then
    printf 'uv at %s is built for Intel Macs, so installing uv %s for Apple silicon beside it...\n' "$uv" "$UV_VERSION"
    uv=
  elif [ -n "$have" ] && at_least "$have" "$UV_MIN"; then
    printf 'Using uv %s at %s.\n' "$have" "$uv"
  else
    printf 'uv %s at %s is older than %s, which ippon needs, so installing uv %s beside it...\n' \
      "${have:-(unknown)}" "$uv" "$UV_MIN" "$UV_VERSION"
    uv=
  fi
fi
if [ -z "$uv" ]; then
  printf 'Installing uv %s, which installs ippon...\n' "$UV_VERSION"
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | sh
  uv=$(uv_installed)
  [ -n "$uv" ] || fail "uv did not install where its installer puts it. See https://docs.astral.sh/uv/ to install it, then run this again."
fi

found=$("$uv" python find "$PYTHONS" 2>/dev/null || true)
if [ -n "$found" ] && ! native "$found"; then
  printf 'The Python at %s is built for Intel Macs, so uv installs one for Apple silicon...\n' "$found"
  PYTHONS=cpython-3.12-macos-aarch64
  found=
fi
if [ -n "$found" ]; then
  printf 'Installing %s on the Python already at %s...\n' "$package" "$found"
else
  printf 'Installing %s, with a Python for it (no Python 3.11 or newer on this Mac yet)...\n' "$package"
fi
"$uv" tool install --python "$PYTHONS" --upgrade "$package"

bin=$("$uv" tool dir --bin)
case ":$PATH:" in
  *":$bin:"*) on_path=1 ;;
  *) on_path=0 ;;
esac
# whether $1/ippon is free for a link to uv's, or is that link already
free_for_links() {
  for name in ippon; do
    if [ -L "$1/$name" ]; then
      [ "$(readlink "$1/$name")" = "$bin/$name" ] || return 1
    elif [ -e "$1/$name" ]; then
      return 1
    fi
  done
}

# uv's bin directory is on the PATH of new terminals only. So that ippon works in this one too, its command is linked
# into a directory this terminal's PATH already has and this user can write to without sudo, never replacing a file.
linked=
if [ "$on_path" = 0 ] && [ -z "${UV_NO_MODIFY_PATH:-}" ]; then
  "$uv" tool update-shell >/dev/null 2>&1 || true
  for dir in "$HOME/.local/bin" "$HOME/bin" /opt/homebrew/bin /usr/local/bin; do
    case ":$PATH:" in
      *":$dir:"*) ;;
      *) continue ;;
    esac
    if [ -d "$dir" ] && [ -w "$dir" ] && free_for_links "$dir"; then
      ln -sf "$bin/ippon" "$dir/ippon" && linked=$dir && break
    fi
  done
fi
# uv too, when this terminal can't find it (the uv just installed, beside ippon), so `uv run --with ippon` works here
linked_uv=
if [ -n "$linked" ] && ! command -v uv >/dev/null 2>&1 && [ ! -e "$linked/uv" ]; then
  ln -s "$uv" "$linked/uv" && linked_uv=1
fi

if [ -n "$linked" ]; then
  printf '%s linked into %s, which is on your PATH, so it works in this terminal now.\n' \
    "$([ -n "$linked_uv" ] && echo 'ippon and uv are' || echo 'ippon is')" "$linked"
elif [ "$on_path" = 0 ] && [ -n "${UV_NO_MODIFY_PATH:-}" ]; then
  printf 'UV_NO_MODIFY_PATH is set, so add %s to your PATH to run ippon.\n' "$bin"
elif [ "$on_path" = 0 ]; then
  printf 'Open a new terminal after this, so that %s is on your PATH.\n' "$bin"
fi

# Then `ippon setup`: the model for this Mac, downloaded, and what to run next. It asks on the terminal, which is
# /dev/tty here since this script arrives on standard input. With a model named (sh -s -- kettei-core) it asks no
# question about the model, and options after it (sh -s -- kettei-dot --accept-license) go to setup too. Where nobody can answer (an agent's shell, CI), ippon stays installed and says so.
if [ -n "${1:-}" ] && (: < /dev/tty) 2>/dev/null && [ -t 1 ]; then
  # --yes takes every default, and a model's licence, which has none, is still asked on the terminal
  "$bin/ippon" setup "$@" --yes < /dev/tty || fail "ippon is installed, but setting up $1 failed (above). Run: ippon setup $1"
elif [ -n "${1:-}" ]; then
  "$bin/ippon" setup "$@" --yes || fail "ippon is installed, but setting up $1 failed (above). Run: ippon setup $1 --accept-license, where nobody can answer"
elif (: < /dev/tty) 2>/dev/null && [ -t 1 ]; then
  "$bin/ippon" setup < /dev/tty || fail "ippon is installed, but setup stopped. Run it again any time: ippon setup"
else
  printf '\nippon is installed. To choose and download a decision model for this Mac, run: ippon setup\n'
fi
printf 'Run this installer again to upgrade ippon. To remove it: ippon remove all --models, then uv tool uninstall ippon\n'
