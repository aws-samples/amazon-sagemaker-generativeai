#!/usr/bin/env bash
# Set up the ShoppingBench evaluation harness (idempotent; re-running skips finished steps).
#   - JDK 21 for pyserini/Lucene, matching this machine's OS and CPU (no root needed)
#   - git-lfs (the product catalog is a Git LFS object); a system git-lfs is reused if present
#   - ShoppingBench at a pinned commit, its 1.4 GB catalog, and a Python venv with its requirements
#   - the Lucene index over the ~2.7M-product catalog (a partial index from an interrupted build is rebuilt)
# Usage: [PYTHON=python3.12] setup_shopbench.sh [--code-only] <work_dir>   (creates <work_dir>/{jdk,ShoppingBench,sbenv,corpus,...})
#   --code-only: only the ShoppingBench code and test files at the pinned commit (no catalog, Java or index). Used by
#                the training notebook's data prep; a later full run in the same <work_dir> reuses the checkout.
set -euo pipefail
CODE_ONLY=0; [ "${1:-}" = "--code-only" ] && { CODE_ONLY=1; shift; }
WORK="${1:?usage: setup_shopbench.sh [--code-only] <work_dir>}"; mkdir -p "$WORK"; cd "$WORK"
SB_COMMIT=2f4d132500d4962d7ee3143e103b93499c82aca0
if [ "$CODE_ONLY" = 1 ]; then
  [ -d ShoppingBench ] || GIT_LFS_SKIP_SMUDGE=1 git clone -q https://github.com/yjwjy/ShoppingBench ShoppingBench
  GIT_LFS_SKIP_SMUDGE=1 git -C ShoppingBench checkout -q "$SB_COMMIT"
  echo "ShoppingBench code ready: $WORK/ShoppingBench @ $SB_COMMIT"; exit 0
fi
case "$(uname -s)" in Darwin) OS=mac ;; Linux) OS=linux ;; *) echo "unsupported OS" >&2; exit 1 ;; esac
case "$(uname -m)" in arm64|aarch64) ARCH=aarch64; LFS_ARCH=arm64 ;; x86_64|amd64) ARCH=x64; LFS_ARCH=amd64 ;; *) echo "unsupported CPU" >&2; exit 1 ;; esac

# --- JDK 21
java_home() { [ -d "$WORK/jdk/Contents/Home" ] && echo "$WORK/jdk/Contents/Home" || echo "$WORK/jdk"; }
if [ ! -x "$(java_home)/bin/java" ]; then
  rm -rf jdk && mkdir -p jdk
  curl -sL "https://api.adoptium.net/v3/binary/latest/21/ga/$OS/$ARCH/jdk/hotspot/normal/eclipse" | tar xz -C jdk --strip-components=1
fi
export JAVA_HOME="$(java_home)"; export PATH="$JAVA_HOME/bin:$PATH"

# --- git-lfs
if ! command -v git-lfs >/dev/null; then
  mkdir -p git-lfs
  if [ ! -x git-lfs/git-lfs ]; then
    if [ "$OS" = mac ]; then
      curl -sL -o git-lfs/lfs.zip "https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-darwin-$LFS_ARCH-v3.5.1.zip"
      unzip -jo git-lfs/lfs.zip '*/git-lfs' -d git-lfs >/dev/null
    else
      curl -sL "https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-$LFS_ARCH-v3.5.1.tar.gz" | tar xz -C git-lfs --strip-components=1
    fi
  fi
  export PATH="$WORK/git-lfs:$PATH"
fi

# --- ShoppingBench + catalog
[ -d ShoppingBench ] || git clone -q https://github.com/yjwjy/ShoppingBench ShoppingBench
(cd ShoppingBench && git checkout -q "$SB_COMMIT" && git lfs install --local >/dev/null && git lfs pull)
GZ=ShoppingBench/resources/documents.jsonl.gz
[ "$(wc -c < "$GZ")" -gt 1000000000 ] || { echo "$GZ is still a Git LFS pointer; check git-lfs" >&2; exit 1; }

# --- harness venv (its pinned pyserini / sentence-transformers stay out of the notebook kernel)
if [ ! -x sbenv/bin/python ]; then
  "${PYTHON:-python3}" -m venv sbenv
  sbenv/bin/pip install -q -r ShoppingBench/requirements.txt openai tqdm
fi

# --- Lucene index (a finished index has a segments_N file; an interrupted build leaves *.tmp + write.lock)
IDX=ShoppingBench/indexes
if ! ls "$IDX"/segments_* >/dev/null 2>&1 || ls "$IDX"/*.tmp >/dev/null 2>&1; then
  rm -rf "$IDX"; mkdir -p corpus
  [ -f corpus/documents.jsonl ] || gunzip -c "$GZ" > corpus/documents.jsonl   # own folder: avoids indexing the .gz too
  # pyserini 1.0.0 builds an OpenAI client at import time; current openai releases need some key to be set
  (cd ShoppingBench && OPENAI_API_KEY="${OPENAI_API_KEY:-unused}" ../sbenv/bin/python -m pyserini.index.lucene \
     --collection JsonCollection --input ../corpus --index indexes --generator DefaultLuceneDocumentGenerator \
     --threads 8 --storePositions --storeDocvectors --storeRaw)
fi
echo "ShoppingBench ready: $WORK/ShoppingBench (JAVA_HOME=$JAVA_HOME)"
