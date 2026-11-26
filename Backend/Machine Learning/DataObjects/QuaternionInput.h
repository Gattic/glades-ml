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

	// Load any binary truth-table CSV with N input columns + 1 output column (e.g., xorparity3.csv).
	bool loadParityFile(const std::string& filePath,
	                    QuaternionInputPackingMode packing = QUAT_PACK_GROUPS_OF_4,
	                    QuaternionOutputReadoutMode readout = QUAT_READOUT_REAL_PART);

	// Generate complete 2^nBits odd-parity truth table (1 <= nBits <= 8).
	bool loadBinaryParity(unsigned int nBits,
	                      QuaternionInputPackingMode packing = QUAT_PACK_GROUPS_OF_4,
	                      QuaternionOutputReadoutMode readout = QUAT_READOUT_REAL_PART);

	// Dedicated Wine dataset loader (13 numeric features + 3 wine cultivar classes).
	bool loadWine(const std::string& filePath,
	              const TrainTestSplitConfig& splitCfg,
	              int standardizeFlag = GMath::ZSCORE,
	              QuaternionInputPackingMode packing = QUAT_PACK_GROUPS_OF_4,
	              QuaternionOutputReadoutMode readout = QUAT_READOUT_REAL_PART);

	// Dedicated Dow Jones weekly OHLC 4-tuple regression loader:
	// Packs train-only Z-scored (open, high, low, close) into 1 input quaternion and
	// predicts next_weeks_close standardized by train close mean/std.
	bool loadDowJonesOHLC(const std::string& trainPath,
	                      const std::string& testPath,
	                      float* outTargetMean = NULL,
	                      float* outTargetStd = NULL,
	                      QuaternionInputPackingMode packing = QUAT_PACK_GROUPS_OF_4,
	                      QuaternionOutputReadoutMode readout = QUAT_READOUT_REAL_PART);

	// Synthetic SO(3) 3D vector rotation regression:
	//   Input  (8 reals -> 2 quats): (0, vx, vy, vz), (qr, qi, qj, qk) with ||q|| = 1
	//   Target (4 reals -> 1 quat) : (0, vx', vy', vz') = q (x) (0, v) (x) q*
	// When oodHemisphereAndAngle is true:
	//   Train: vz > 0.25, rotation angle in [0, pi/2]
	//   Test : vz < -0.25, rotation angle in [pi/2, pi]
	bool loadSO3Rotation(unsigned int nTrain,
	                     unsigned int nTest,
	                     unsigned int seed,
	                     bool oodHemisphereAndAngle = false,
	                     QuaternionInputPackingMode packing = QUAT_PACK_GROUPS_OF_4,
	                     QuaternionOutputReadoutMode readout = QUAT_READOUT_SPLIT_COMPONENTS);

	// Synthetic SO(3) quaternion composition regression:
	//   Input  (8 reals -> 2 quats): unit quaternions q1, q2
	//   Target (4 reals -> 1 quat) : q1 (x) q2
	// When oodRotatedFrame is true:
	//   Train: q1.k > 0.15 and q2.k > 0.15
	//   Test : q1.k < -0.15 and q2.k < -0.15
	bool loadSO3Composition(unsigned int nTrain,
	                        unsigned int nTest,
	                        unsigned int seed,
	                        bool oodRotatedFrame = false,
	                        QuaternionInputPackingMode packing = QUAT_PACK_GROUPS_OF_4,
	                        QuaternionOutputReadoutMode readout = QUAT_READOUT_SPLIT_COMPONENTS);

	// Tier 2A: Triatomic H2O-like Potential Energy Surface & Exact Central-Atom Force:
	//   Input  (8 reals -> 2 quats): (0, r1x, r1y, r1z), (0, r2x, r2y, r2z)
	//   Target (4 reals -> 1 quat) : (E, Fx, Fy, Fz) where E is SO(3)-invariant Morse+bend
	//     potential energy and F = -dE/dx_O = dE/dr1 + dE/dr2 is the SO(3)-equivariant force.
	// When oodOrientation is true:
	//   Train: molecules rotated within [0, 30 deg] of canonical frame
	//   Test : molecules rotated by unseen [60, 180 deg] rotations in SO(3)
	bool loadMolecularPES(unsigned int nTrain,
	                      unsigned int nTest,
	                      unsigned int seed,
	                      bool oodOrientation = false,
	                      QuaternionInputPackingMode packing = QUAT_PACK_GROUPS_OF_4,
	                      QuaternionOutputReadoutMode readout = QUAT_READOUT_SPLIT_COMPONENTS);

	// Tier 2B: Chiral Molecular Center (C-L1L2L3) 3-Class Classification:
	//   Input  (12 reals -> 3 quats): (c1, r1), (c2, r2), (c3, r3) with distinct ligand scalars
	//   Target (3 one-hot classes)  : 0=Planar, 1=R-Enantiomer (tau > 0), 2=S-Enantiomer (tau < 0)
	//   Strictly SO(3)-invariant and parity-sensitive (mirror reflection swaps R <-> S).
	// When oodOrientation is true:
	//   Train: orientations within [0, 35 deg] of canonical frame
	//   Test : unseen orientations in [60, 180 deg]
	bool loadMolecularChirality(unsigned int nTrain,
	                            unsigned int nTest,
	                            unsigned int seed,
	                            bool oodOrientation = false,
	                            QuaternionInputPackingMode packing = QUAT_PACK_GROUPS_OF_4,
	                            QuaternionOutputReadoutMode readout = QUAT_READOUT_REAL_PART);

	// Tier 2C: Quantum Slater-Koster p-p Molecular Orbital Overlap Integral:
	//   Input  (12 reals -> 3 quats): (0, r), (0, p1), (0, p2)
	//   Target (1 scalar -> 1 quat) : S_pp(r, p1, p2) (strictly SO(3)-invariant)
	// When oodOrientation is true:
	//   Train: bond direction r_hat.z > 0.65 (upper polar cone)
	//   Test : bond direction r_hat.z < -0.35 (lower hemisphere)
	bool loadSlaterKosterOrbital(unsigned int nTrain,
	                             unsigned int nTest,
	                             unsigned int seed,
	                             bool oodOrientation = false,
	                             QuaternionInputPackingMode packing = QUAT_PACK_GROUPS_OF_4,
	                             QuaternionOutputReadoutMode readout = QUAT_READOUT_REAL_PART);

	// Tier 2D: 3D Keplerian Orbital Mechanics (mu = 1):
	//   Input  (8 reals -> 2 quats): (0, rx, ry, rz), (0, vx, vy, vz)
	//   Target (4 reals -> 1 quat) : (eps, ex, ey, ez) where eps = 0.5*||v||^2 - 1/||r|| is the
	//     SO(3)-invariant specific orbital energy and e = v x (r x v) - r/||r|| is the
	//     SO(3)-equivariant Laplace-Runge-Lenz eccentricity vector.
	// When oodOrbitalPlane is true:
	//   Train: near-equatorial prograde orbits (inclination i in [0, 25 deg])
	//   Test : steeply inclined / polar / retrograde orbits (inclination i in [65, 180 deg])
	bool loadKeplerOrbit(unsigned int nTrain,
	                     unsigned int nTest,
	                     unsigned int seed,
	                     bool oodOrbitalPlane = false,
	                     QuaternionInputPackingMode packing = QUAT_PACK_GROUPS_OF_4,
	                     QuaternionOutputReadoutMode readout = QUAT_READOUT_SPLIT_COMPONENTS);

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
