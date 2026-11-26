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
#ifndef _GQUATERNIONINPUT
#define _GQUATERNIONINPUT

#include "NumberInput.h"
#include "../GMath/gmath.h"
#include "../GMath/quaternion.h"
#include <string>
#include <vector>

namespace glades {

// Quaternion-aware tabular DataInput for Quaternion Neural Networks (TYPE_QNN).
//
// Inherits from NumberInput so it plugs directly into NNetwork::train/test and
// model persistence while also materializing explicit Quaternion input and target
// representations using the same boundary codec as TYPE_QNN:
//   - Input packing:
//       QUAT_PACK_GROUPS_OF_4: packs 4 real features into (r, i, j, k) per quaternion
//         (e.g., Iris 4 features [SL, SW, PL, PW] -> 1 input quaternion).
//       QUAT_ONE_PER_FEATURE: embeds each real feature f -> (x_f, 0, 0, 0).
//   - Target/output mapping:
//       QUAT_READOUT_REAL_PART: one quaternion per scalar target/class [(y_c, 0, 0, 0)]
//         (e.g., Iris 3 classes -> 3 output quaternions).
//       QUAT_READOUT_SPLIT_COMPONENTS: packs up to 4 targets into (r, i, j, k) lanes
//         (e.g., Iris 3 classes -> 1 output quaternion (y_0, y_1, y_2, 0)).
class QuaternionInput : public NumberInput
{
public:
	QuaternionInput();
	virtual ~QuaternionInput();

	// Load any GTable with output column(s) already toggled, performing deterministic
	// train/test splitting and train-only standardization, then materializing the
	// quaternion input and target buffers.
	bool loadTabularWithSplit(const shmea::GTable& rawTable,
	                          const TrainTestSplitConfig& splitCfg,
	                          int standardizeFlag = GMath::ZSCORE,
	                          QuaternionInputPackingMode packing = QUAT_PACK_GROUPS_OF_4,
	                          QuaternionOutputReadoutMode readout = QUAT_READOUT_REAL_PART);

	// Dedicated Iris dataset loader (4 numeric features + 3 species classes).
	// Maps the 4 standardized features (SL, SW, PL, PW) into quaternion inputs and the
	// 3 one-hot species classes into quaternion targets according to (packing, readout).
	bool loadIris(const std::string& filePath,
	              const TrainTestSplitConfig& splitCfg,
	              int standardizeFlag = GMath::ZSCORE,
	              QuaternionInputPackingMode packing = QUAT_PACK_GROUPS_OF_4,
	              QuaternionOutputReadoutMode readout = QUAT_READOUT_REAL_PART);

	// Dedicated XOR gate dataset loader (2 binary features + 1 binary target).
	bool loadXOR(const std::string& filePath,
	             QuaternionInputPackingMode packing = QUAT_PACK_GROUPS_OF_4,
	             QuaternionOutputReadoutMode readout = QUAT_READOUT_REAL_PART);

	// Materialize/re-encode quaternion input & target buffers from the currently
	// loaded NumberInput train/test matrices.
	bool rebuildQuaternionBuffers(QuaternionInputPackingMode packing = QUAT_PACK_GROUPS_OF_4,
	                              QuaternionOutputReadoutMode readout = QUAT_READOUT_REAL_PART);

	// Zero-copy quaternion row views (row-major flat storage).
	bool getTrainQuatInputRow(unsigned int index, const Quaternion*& outData, unsigned int& outCount) const;
	bool getTrainQuatTargetRow(unsigned int index, const Quaternion*& outData, unsigned int& outCount) const;
	bool getTestQuatInputRow(unsigned int index, const Quaternion*& outData, unsigned int& outCount) const;
	bool getTestQuatTargetRow(unsigned int index, const Quaternion*& outData, unsigned int& outCount) const;

	unsigned int getQuatInputCount() const { return quatInputCount; }
	unsigned int getQuatOutputCount() const { return quatOutputCount; }
	unsigned int getRealOutputCount() const { return realOutputCount; }
	QuaternionInputPackingMode getInputPackingMode() const { return inputPackingMode; }
	QuaternionOutputReadoutMode getOutputReadoutMode() const { return outputReadoutMode; }
	const std::vector<std::string>& getClassLabels() const { return classLabels; }

	virtual NNetworkStatus getLastStatus() const { return lastStatus; }

private:
	QuaternionInputPackingMode inputPackingMode;
	QuaternionOutputReadoutMode outputReadoutMode;
	unsigned int quatInputCount;
	unsigned int quatOutputCount;
	unsigned int realOutputCount;

	// Contiguous row-major [numRows][quatInputCount / quatOutputCount] buffers.
	std::vector<Quaternion> trainQuatInputs;
	std::vector<Quaternion> trainQuatTargets;
	std::vector<Quaternion> testQuatInputs;
	std::vector<Quaternion> testQuatTargets;
	std::vector<std::string> classLabels;
	NNetworkStatus lastStatus;

	void clearQuaternionBuffers();
};

} // namespace glades

#endif
