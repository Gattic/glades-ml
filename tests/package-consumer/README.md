# Public CPU package / deployment regression

After building the CPU library in `build/`, run:

```sh
bash unit-tests/test.sh package-consumer
```

The consumer uses only `find_package(glades CONFIG REQUIRED)` and the public
`glades` target: no source include patches. It compiles/links against the build
export and a relocated install export (`cmake --install --prefix` into an isolated
build-owned prefix; the user's installed library and trainer are not changed).
Evidence, models and logs stay in `build/package-consumer-*`.

`NNetwork::loadModelDirectory(absoluteDirectory, forShape)` reads a v3 model
package at an explicit location, independent of its directory basename. No
`glades::init()` or CWD change is needed for direct network evaluation with
explicit callbacks. DFF per-step progress logging belongs to training, not eval;
callbacks own evaluation observability. The directory loader neither
creates directories nor writes the package. It requires integrity metadata,
verifies the existing weight checksum/size and architecture size regardless of
`GLADES_MODEL_VERIFY_FILES`, and rejects DFF geometry inconsistent with the saved
architecture or caller's input shape. Subsequent evaluations must retain that
shape. Other model families retain their existing codec checks.

Use trusted, immutable artifacts. These checks are not authentication or a
concurrent file-replacement sandbox. In particular the v3 format has no
architecture-content hash, so deployment applications should authenticate/hash
the complete package separately. Discard the network after a failed load.
Named `saveModel`/`loadModel` and training/checkpoint APIs keep their current
contracts and share the same package codec; no research/trainer APIs change.

Coverage includes a real small DFF training/save/read/prediction round trip,
repeat inference and unchanged epoch/seed, renamed paths containing spaces,
read-only model and empty read-only CWD, relocation, missing/corrupt/truncated
files, absent integrity metadata, symlink files/roots, null and mismatched input
shape. The existing `save-load` selector covers named-package regression.

Language policy: the existing build selects C++23 and public dependencies use
modern headers. This change does not alter that setting; new library and C++
regression code uses C++98-compatible constructs as required by `AGENTS.md`.
This is not a claim that the complete current library builds under C++98.
