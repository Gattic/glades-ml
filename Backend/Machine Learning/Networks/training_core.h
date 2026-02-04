// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// TrainingCore: side-effect-free training/inference core.
//
// This module owns the "run loop" for the neural network engine, but it does NOT:
// - print to stdout/stderr
// - send GUI/network messages
// - instantiate any default callbacks
//
// Observability is optional and provided by pluggable callbacks passed in by the caller.
//
#ifndef _GLADES_TRAINING_CORE_H_
#define _GLADES_TRAINING_CORE_H_

namespace glades {

class NNetwork;
class DataInput;
class ITrainingCallbacks;
struct NNetworkStatus;

class TrainingCore
{
public:
	// Run a single train/test loop using the provided callbacks (may be NULL).
	// This function is synchronous and does not retain the callbacks pointer.
	//
	// NOTE: TrainingCore is a legacy/stable API surface. The implementation delegates to `Trainer`
	// to keep training orchestration separate from model representation.
	static NNetworkStatus run(NNetwork& net, const DataInput* data, int runType, ITrainingCallbacks* callbacks);
};

} // namespace glades

#endif

