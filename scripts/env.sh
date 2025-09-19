# Common environment setup across build*.sh scripts

# Derive version metadata from Cargo and Git so that Rust builds embed
# consistent release information. This function populates VERSION, VERSION_SEMVER,
# and VERSION_METADATA if they are not already provided by the caller.
derive_version_metadata() {
    # Determine the Cargo package that produces the primary binary.
    cargo_package=${CARGO_PACKAGE_NAME:-"oxi-llama"}

    cargo_version=""
    if command -v cargo >/dev/null 2>&1; then
        cargo_version=$(cargo pkgid -p "$cargo_package" 2>/dev/null | awk -F'#' 'NF > 1 { print $2 }')
    fi

    git_describe=""
    if command -v git >/dev/null 2>&1 && git rev-parse --git-dir >/dev/null 2>&1; then
        git_describe=$(git describe --tags --first-parent --abbrev=7 --long --dirty --always 2>/dev/null | sed -e 's/^v//g')
    fi

    metadata=""
    if [ -n "$cargo_version" ] && [ -n "$git_describe" ]; then
        case "$git_describe" in
            ${cargo_version}-*)
                metadata=${git_describe#${cargo_version}-}
                metadata=$(printf '%s' "$metadata" | sed 's/-/./g')
                ;;
            *)
                metadata=$(printf '%s' "$git_describe" | sed 's/-/./g')
                ;;
        esac
    elif [ -n "$git_describe" ]; then
        metadata=$(printf '%s' "$git_describe" | sed 's/-/./g')
    fi

    if [ -z "${VERSION:-}" ]; then
        if [ -n "$cargo_version" ]; then
            if [ -n "$metadata" ]; then
                VERSION="${cargo_version}+${metadata}"
            else
                VERSION="$cargo_version"
            fi
        elif [ -n "$git_describe" ]; then
            VERSION="$git_describe"
        else
            VERSION="0.0.0"
        fi
    fi

    if [ -z "${VERSION_SEMVER:-}" ]; then
        if [ -n "$cargo_version" ]; then
            VERSION_SEMVER="$cargo_version"
        else
            VERSION_SEMVER="${VERSION%%+*}"
        fi
    fi

    if [ -z "${VERSION_METADATA:-}" ]; then
        if [ -n "$metadata" ]; then
            VERSION_METADATA="$metadata"
        elif [ -n "$git_describe" ]; then
            VERSION_METADATA=$(printf '%s' "$git_describe" | sed 's/-/./g')
        else
            VERSION_METADATA="${VERSION}"
        fi
    fi

    export VERSION VERSION_SEMVER VERSION_METADATA

    unset cargo_package cargo_version git_describe metadata
}

derive_version_metadata

# TODO - consider `docker buildx ls --format=json` to autodiscover platform capability
PLATFORM=${PLATFORM:-"linux/arm64,linux/amd64"}
DOCKER_ORG=${DOCKER_ORG:-"ollama"}
FINAL_IMAGE_REPO=${FINAL_IMAGE_REPO:-"${DOCKER_ORG}/ollama"}
CARGO_FEATURES=${CARGO_FEATURES:-""}
RUST_TARGETS=${RUST_TARGETS:-""}
OLLAMA_COMMON_BUILD_ARGS="--build-arg=VERSION --build-arg=CARGO_FEATURES=${CARGO_FEATURES} --build-arg=RUST_TARGETS=${RUST_TARGETS}"

if [ -z "${OLLAMA_ENV_QUIET:-}" ]; then
    echo "Building Ollama"
    echo "VERSION=$VERSION"
    echo "PLATFORM=$PLATFORM"
    if [ -n "$CARGO_FEATURES" ]; then
        echo "CARGO_FEATURES=$CARGO_FEATURES"
    fi
    if [ -n "$RUST_TARGETS" ]; then
        echo "RUST_TARGETS=$RUST_TARGETS"
    fi
    if [ -n "$VERSION_METADATA" ] && [ "$VERSION_METADATA" != "$VERSION" ]; then
        echo "VERSION_METADATA=$VERSION_METADATA"
    fi
fi
