# SochDB Python SDK Release Process

## Overview

The SochDB Python SDK wraps pre-compiled binaries from the **main SochDB repository** ([sochdb/sochdb](https://github.com/sochdb/sochdb)) and packages them into platform-specific Python wheels for distribution on PyPI.

## How It Works

### 1. Binary Source
- **All binaries come from** the main [sochdb/sochdb](https://github.com/sochdb/sochdb) repository
- The Python SDK does NOT compile binaries itself
- Each release pulls pre-built binaries from a specific sochdb/sochdb release

### 2. What Gets Packaged
Each platform-specific wheel contains:
- **Python SDK code** (`sochdb` package)
- **Executables** (in `_bin/<platform>/`):
  - `sochdb-bulk` - CLI tool for bulk operations
  - `sochdb-server` - Standalone database server
  - `sochdb-grpc-server` - gRPC server implementation
- **Native Libraries** (in `lib/<platform>/`):
  - `libsochdb_storage.*` - Storage engine FFI library
  - `libsochdb_index.*` - Indexing engine FFI library

### 3. Platform Support
The workflow builds wheels for:
- **Linux x86_64** (`manylinux_2_17_x86_64`)
- **macOS ARM64** (`macosx_11_0_arm64`) - Apple Silicon
- **macOS x86_64** (`macosx_10_15_x86_64`) - Intel
- **Windows x64** (`win_amd64`)

Unsupported platforms should not fall back to a PyPI source distribution, because that can produce a misleading install with the wrong bundled native binaries.

### 4. Python Version Support
Wheels are compatible with:
- Python 3.9
- Python 3.10
- Python 3.11
- Python 3.12
- Python 3.13

## Release Workflow

### Prerequisites
1. Ensure the desired version exists as a release in [sochdb/sochdb](https://github.com/sochdb/sochdb/releases)
2. The release must have platform-specific archives:
   - `sochdb-{version}-x86_64-unknown-linux-gnu.tar.gz`
   - `sochdb-{version}-aarch64-apple-darwin.tar.gz`
   - `sochdb-{version}-x86_64-apple-darwin.tar.gz`
   - `sochdb-{version}-x86_64-pc-windows-msvc.zip`

### Running a Release

1. **Go to Actions tab** in GitHub
2. **Select "Release" workflow**
3. **Click "Run workflow"**
4. **Enter parameters:**
   - `version`: The version for the Python SDK (e.g., `0.3.2`)
   - `sochdb_version`: (Optional) If different from `version`, specify the sochdb/sochdb release to pull binaries from
   - `dry_run`: Check this to validate without publishing

### What Happens During Release

1. **Build Wheels** (parallel):
   - Downloads binaries from `sochdb/sochdb` release
   - Creates platform-specific wheels
   - Each wheel is self-contained with binaries for that platform

2. **Build Source Distribution**:
   - Creates `.tar.gz` with Python source code
   - Does NOT include binaries

3. **Create GitHub Release**:
   - Creates a new release tag (e.g., `v0.3.2`)
   - Attaches all wheels and source distribution
   - Generates release notes with installation instructions

4. **Publish to PyPI**:
   - Uploads wheels only
   - Keeps the source distribution off PyPI so unsupported platforms do not fall back to a broken binary mix
   - Uses OIDC Trusted Publisher (no token needed)

5. **Summary**:
   - Shows comprehensive build status
   - Links to PyPI package and GitHub release
   - Shows source binary repository

## Example Release

```bash
# In sochdb/sochdb - first create a release there
git tag v0.3.2
git push origin v0.3.2
# Build and upload platform binaries to GitHub release

# Then in sochdb-python-sdk - run the workflow
# Go to: https://github.com/sochdb/sochdb-python-sdk/actions
# Run workflow with:
#   version: 0.3.2
#   sochdb_version: 0.3.2 (or leave blank to use same version)
#   dry_run: false
```

## Versioning Strategy

### Option 1: Same Version (Recommended)
- Python SDK version matches SochDB core version
- Example: Both are `0.3.2`
- Simplest for users to understand

### Option 2: Independent Versions
- Python SDK has its own versioning
- Specify `sochdb_version` to pull specific binaries
- Example: SDK is `1.0.0`, pulls binaries from SochDB `0.3.2`

## Troubleshooting

### "Release not found" Error
- Ensure the sochdb/sochdb release exists with the correct tag format (`v0.3.2`)
- Check that platform-specific archives are attached to the release

### "No packages showing" in GitHub
- This is now fixed! The workflow creates a GitHub release with all packages attached
- Check the [Releases page](https://github.com/sochdb/sochdb-python-sdk/releases)

### Wheel Platform Tag Issues
- Wheels are built as `py3-none-{platform}` (platform-specific, not pure Python)
- The workflow automatically renames wheels with correct platform tags

### Missing Binaries in Wheel
- Check the workflow logs under "Copy binaries and libraries to SDK"
- Verify the sochdb/sochdb release has the expected binary files in the archive

## Manual Testing

To test a release locally before publishing:

```bash
# Download a wheel from the workflow artifacts or GitHub release
pip install sochdb_client-0.3.2-py3-none-macosx_11_0_arm64.whl

# Verify binaries are included
python -c "import sochdb; print(sochdb.__file__)"
ls -la /path/to/site-packages/sochdb/_bin/
ls -la /path/to/site-packages/sochdb/lib/

# Test basic functionality
python -c "from sochdb import Database; db = Database.open(':memory:'); print('OK')"

# Test CLI tools
sochdb-bulk --help
sochdb-server --help
```

## Publishing Checklist

- [ ] Verify sochdb/sochdb release exists with all platform binaries
- [ ] Update `version` in `pyproject.toml` if needed
- [ ] Run workflow with `dry_run: true` first
- [ ] Review workflow artifacts and logs
- [ ] Run workflow with `dry_run: false` for production
- [ ] Verify GitHub release is created
- [ ] Verify only wheel packages appear on PyPI (no sdist upload)
- [ ] Test installation: `pip install sochdb-client==X.Y.Z`
- [ ] Update CHANGELOG.md with release notes

## Links

- **PyPI Package**: https://pypi.org/project/sochdb-client/
- **GitHub Releases**: https://github.com/sochdb/sochdb-python-sdk/releases
- **Main SochDB Repo**: https://github.com/sochdb/sochdb
- **SochDB Releases**: https://github.com/sochdb/sochdb/releases
