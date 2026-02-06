// Property-based + fuzz-like (randomized) tests for Glades ML.
//
// These tests are intentionally opt-in because they can be slower than the core unit suite.
// Run via:
//   ./build/glades-unit-tests prop-fuzz
//
// NOTE: This is NOT libFuzzer. See fuzz harnesses in this directory for libFuzzer entrypoints.
#pragma once

void PropFuzzUnitTest();

