#!/usr/bin/env bash
# Rootless OSMesa (software OpenGL) for rendering videos on a headless machine without
# working EGL, e.g. the cluster, where the GPU has no graphics support. Adapted from the
# course lab's setup_mujoco_headless.sh.
#
#   scripts/setup_osmesa.sh
#
# Downloads libosmesa6 and its missing dependencies with apt-get (download only, no sudo,
# nothing installed system-wide) and extracts them under .mujoco-osmesa/. The video renderer
# (render.py) adds the extracted libraries to LD_LIBRARY_PATH when it tries MUJOCO_GL=osmesa.
# Re-running is safe; delete .mujoco-osmesa/ to download again.
set -euo pipefail
cd "$(dirname "$0")/.."

PREFIX=$PWD/.mujoco-osmesa
APTROOT=$PREFIX/apt
ROOTFS=$PREFIX/root

for tool in apt-get dpkg-deb ldd; do
    command -v "$tool" >/dev/null || { echo "OSMesa setup: $tool not found" >&2; exit 1; }
done

OSMESA=$(find "$ROOTFS" -type f -name 'libOSMesa.so.*' 2>/dev/null | head -n 1 || true)
if [ -z "$OSMESA" ]; then
    echo "Downloading OSMesa into $PREFIX (no sudo, nothing installed system-wide)"
    mkdir -p "$APTROOT/lists/partial" "$APTROOT/archives/partial" "$ROOTFS"
    # A private APT state lets an unprivileged user run update and download; reading the
    # system dpkg status means only the dependencies missing from this machine are fetched.
    APT_OPTS=(
        -o "Debug::NoLocking=1"
        -o "APT::Sandbox::User=$(id -un)"
        -o "Dir::State::status=/var/lib/dpkg/status"
        -o "Dir::State::lists=$APTROOT/lists"
        -o "Dir::Cache::archives=$APTROOT/archives"
        -o "APT::Get::List-Cleanup=0"
    )
    apt-get "${APT_OPTS[@]}" -qq update
    apt-get "${APT_OPTS[@]}" -qq -y -d --no-install-recommends install libosmesa6
    if ! compgen -G "$APTROOT/archives/*.deb" >/dev/null; then
        echo "OSMesa is already installed system-wide"
        exit 0
    fi
    for deb in "$APTROOT"/archives/*.deb; do
        dpkg-deb -x "$deb" "$ROOTFS"
    done
    OSMESA=$(find "$ROOTFS" -type f -name 'libOSMesa.so.*' | head -n 1 || true)
    [ -n "$OSMESA" ] || { echo "OSMesa setup: libOSMesa not found after extraction" >&2; exit 1; }
fi
# PyOpenGL loads libOSMesa.so first.
ln -sfn "$(basename "$OSMESA")" "$(dirname "$OSMESA")/libOSMesa.so"

LIBRARY_PATH=$(find "$ROOTFS" -name '*.so*' -printf '%h\n' | sort -u | paste -sd:)
if LD_LIBRARY_PATH="$LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$OSMESA" \
    | grep 'not found' >&2; then
    echo "OSMesa setup: the libraries above are missing" >&2
    exit 1
fi
echo "OSMesa ready: $OSMESA"
