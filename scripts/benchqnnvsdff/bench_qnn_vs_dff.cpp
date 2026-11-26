// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
// Isolated QNN (TYPE_QNN) vs. DFNN (TYPE_DFF) benchmark worker.
// Designed to be invoked once per (task, model) in a fresh OS process by
// scripts/benchqnnvsdff.sh so peak RSS (ru_maxrss) and live heap delta
// (mallinfo2) reflect each model configuration in isolation.

#include "Backend/Machine Learning/main.h"
#include "Backend/Machine Learning/rng.h"
#include "Backend/Machine Learning/DataObjects/NumberInput.h"
#include "Backend/Machine Learning/DataObjects/QuaternionInput.h"
#include "Backend/Machine Learning/GMath/gmath.h"
#include "Backend/Machine Learning/GMath/quaternion.h"
#include "Backend/Machine Learning/Networks/network.h"
#include "Backend/Machine Learning/Networks/training_callbacks.h"
#include "Backend/Machine Learning/State/Terminator.h"
#include "Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "Backend/Machine Learning/Structure/nninfo.h"
#include "Backend/Machine Learning/Structure/outputlayerinfo.h"
#include "Backend/Database/GLogger.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <malloc.h>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace {

struct ModelSpec
{
	const char* id;
	const char* label;
	const char* task; // xor, iris80, iris40, tier1 tasks, tier2 tasks (mol_pes_*, mol_chiral_*, orbital_slater_*, orbital_kepler_*)
	int netType;      // NNetwork::TYPE_QNN or NNetwork::TYPE_DFF
	unsigned int hiddenSize;
	unsigned int hiddenSize2; // 0 for 1-hidden-layer; >0 for 2-hidden-layer
	glades::QuaternionConfig::InputPackingMode qnnPacking;
	glades::QuaternionConfig::OutputReadoutMode qnnReadout;
	glades::QuaternionConfig::LayerMode qnnLayerMode;
	glades::QuaternionConfig::ActivationMode qnnActMode;
	glades::OptimizerConfig::Type optimizer;
	float lr;
	float momentum;
	float wd2;
	unsigned int epochs;
	unsigned int taskParam; // e.g., N_train for so3_rot_curve / tier2 tasks
};

// Shorthand constants for concise table declarations (C++98 compatible)
#define PK4  glades::QuaternionConfig::PACK_GROUPS_OF_4
#define PK1  glades::QuaternionConfig::ONE_QUAT_PER_FEATURE
#define RDR  glades::QuaternionConfig::READOUT_REAL_PART
#define RDS  glades::QuaternionConfig::READOUT_SPLIT_COMPONENTS
#define LSTD glades::QuaternionConfig::LAYER_STANDARD
#define LEQV glades::QuaternionConfig::LAYER_SO3_EQUIVARIANT
#define ASPL glades::QuaternionConfig::ACT_SPLIT
#define APOL glades::QuaternionConfig::ACT_POLAR_TANH
#define SGDM glades::OptimizerConfig::SGD_MOMENTUM
#define ADMW glades::OptimizerConfig::ADAMW
#define TQNN glades::NNetwork::TYPE_QNN
#define TDFF glades::NNetwork::TYPE_DFF

#define SO3_ROT_CURVE_BLOCK(N) \
	{ "qnn_equiv_polar_adamw_n" #N, "QNN-SO3-Equiv 2->8->1 Polar (AdamW) [N=" #N "]", "so3_rot_curve", TQNN,  8u, 0u, PK4, RDS, LEQV, APOL, ADMW, 0.02f, 0.0f, 1e-5f, 400u, N##u }, \
	{ "qnn_equiv_polar_sgdm_n"  #N, "QNN-SO3-Equiv 2->8->1 Polar (SGD+M) [N=" #N "]", "so3_rot_curve", TQNN,  8u, 0u, PK4, RDS, LEQV, APOL, SGDM, 0.05f, 0.9f, 1e-5f, 400u, N##u }, \
	{ "qnn_std_polar_adamw_n"   #N, "QNN-Std 2->8->1 Polar (AdamW) [N=" #N "]",       "so3_rot_curve", TQNN,  8u, 0u, PK4, RDS, LSTD, APOL, ADMW, 0.02f, 0.0f, 1e-5f, 400u, N##u }, \
	{ "qnn_std_split_adamw_n"   #N, "QNN-Std 2->8->1 Split (AdamW) [N=" #N "]",       "so3_rot_curve", TQNN,  8u, 0u, PK4, RDS, LSTD, ASPL, ADMW, 0.02f, 0.0f, 1e-5f, 400u, N##u }, \
	{ "qnn_std_split_sgdm_n"    #N, "QNN-Std 2->8->1 Split (SGD+M) [N=" #N "]",       "so3_rot_curve", TQNN,  8u, 0u, PK4, RDS, LSTD, ASPL, SGDM, 0.05f, 0.9f, 1e-5f, 400u, N##u }, \
	{ "dff_8_32_4_sgdm_n"       #N, "DFF 8->32->4 [width=] (SGD+M) [N=" #N "]",       "so3_rot_curve", TDFF, 32u, 0u, PK4, RDS, LSTD, ASPL, SGDM, 0.05f, 0.9f, 1e-5f, 400u, N##u }, \
	{ "dff_8_10_4_sgdm_n"       #N, "DFF 8->10->4 [param=] (SGD+M) [N=" #N "]",       "so3_rot_curve", TDFF, 10u, 0u, PK4, RDS, LSTD, ASPL, SGDM, 0.05f, 0.9f, 1e-5f, 400u, N##u }

#define TIER2_JOINT4_BLOCK(TASK_ID, LR_EQ_SGDM, LR_STD_SGDM) \
	{ "qnn_equiv_polar_adamw", "QNN-SO3-Equiv 2->8->6->1 Polar (AdamW)", TASK_ID, TQNN,  8u,  6u, PK4, RDS, LEQV, APOL, ADMW, 0.015f,     0.0f, 1e-5f, 400u, 192u }, \
	{ "qnn_equiv_polar_sgdm",  "QNN-SO3-Equiv 2->8->6->1 Polar (SGD+M)", TASK_ID, TQNN,  8u,  6u, PK4, RDS, LEQV, APOL, SGDM, LR_EQ_SGDM, 0.9f, 1e-5f, 400u, 192u }, \
	{ "qnn_std_polar_adamw",   "QNN-Std 2->8->6->1 Polar (AdamW)",       TASK_ID, TQNN,  8u,  6u, PK4, RDS, LSTD, APOL, ADMW, 0.015f,     0.0f, 1e-5f, 400u, 192u }, \
	{ "qnn_std_split_adamw",   "QNN-Std 2->8->6->1 Split (AdamW)",       TASK_ID, TQNN,  8u,  6u, PK4, RDS, LSTD, ASPL, ADMW, 0.015f,     0.0f, 1e-5f, 400u, 192u }, \
	{ "qnn_std_split_sgdm",    "QNN-Std 2->8->6->1 Split (SGD+M)",       TASK_ID, TQNN,  8u,  6u, PK4, RDS, LSTD, ASPL, SGDM, LR_STD_SGDM,0.9f, 1e-5f, 400u, 192u }, \
	{ "dff_8_32_24_4_sgdm",    "DFF 8->32->24->4 [width=] (SGD+M)",      TASK_ID, TDFF, 32u, 24u, PK4, RDS, LSTD, ASPL, SGDM, LR_STD_SGDM,0.9f, 1e-5f, 400u, 192u }, \
	{ "dff_8_14_11_4_sgdm",    "DFF 8->14->11->4 [param=] (SGD+M)",      TASK_ID, TDFF, 14u, 11u, PK4, RDS, LSTD, ASPL, SGDM, LR_STD_SGDM,0.9f, 1e-5f, 400u, 192u }

#define TIER2_CHIRAL_BLOCK(TASK_ID) \
	{ "qnn_equiv_polar_adamw", "QNN-SO3-Equiv 3->8->3 Polar (AdamW)",    TASK_ID, TQNN,  8u,  0u, PK4, RDR, LEQV, APOL, ADMW, 0.02f, 0.0f, 1e-4f, 300u, 180u }, \
	{ "qnn_equiv_polar_sgdm",  "QNN-SO3-Equiv 3->8->3 Polar (SGD+M)",    TASK_ID, TQNN,  8u,  0u, PK4, RDR, LEQV, APOL, SGDM, 0.05f, 0.9f, 1e-4f, 300u, 180u }, \
	{ "qnn_std_polar_adamw",   "QNN-Std 3->8->3 Polar (AdamW)",          TASK_ID, TQNN,  8u,  0u, PK4, RDR, LSTD, APOL, ADMW, 0.02f, 0.0f, 1e-4f, 300u, 180u }, \
	{ "qnn_std_split_adamw",   "QNN-Std 3->8->3 Split (AdamW)",          TASK_ID, TQNN,  8u,  0u, PK4, RDR, LSTD, ASPL, ADMW, 0.02f, 0.0f, 1e-4f, 300u, 180u }, \
	{ "qnn_std_split_sgdm",    "QNN-Std 3->8->3 Split (SGD+M)",          TASK_ID, TQNN,  8u,  0u, PK4, RDR, LSTD, ASPL, SGDM, 0.05f, 0.9f, 1e-4f, 300u, 180u }, \
	{ "dff_12_32_3_sgdm",      "DFF 12->32->3 [width=] (SGD+M)",         TASK_ID, TDFF, 32u,  0u, PK4, RDR, LSTD, ASPL, SGDM, 0.05f, 0.9f, 1e-4f, 300u, 180u }, \
	{ "dff_12_14_3_sgdm",      "DFF 12->14->3 [param=] (SGD+M)",         TASK_ID, TDFF, 14u,  0u, PK4, RDR, LSTD, ASPL, SGDM, 0.05f, 0.9f, 1e-4f, 300u, 180u }

#define TIER2_SLATER_BLOCK(TASK_ID) \
	{ "qnn_equiv_polar_adamw", "QNN-SO3-Equiv 3->8->6->1 Polar (AdamW)", TASK_ID, TQNN,  8u,  6u, PK4, RDR, LEQV, APOL, ADMW, 0.015f, 0.0f, 1e-5f, 400u, 192u }, \
	{ "qnn_equiv_polar_sgdm",  "QNN-SO3-Equiv 3->8->6->1 Polar (SGD+M)", TASK_ID, TQNN,  8u,  6u, PK4, RDR, LEQV, APOL, SGDM, 0.020f, 0.9f, 1e-5f, 400u, 192u }, \
	{ "qnn_std_polar_adamw",   "QNN-Std 3->8->6->1 Polar (AdamW)",       TASK_ID, TQNN,  8u,  6u, PK4, RDR, LSTD, APOL, ADMW, 0.015f, 0.0f, 1e-5f, 400u, 192u }, \
	{ "qnn_std_split_adamw",   "QNN-Std 3->8->6->1 Split (AdamW)",       TASK_ID, TQNN,  8u,  6u, PK4, RDR, LSTD, ASPL, ADMW, 0.015f, 0.0f, 1e-5f, 400u, 192u }, \
	{ "qnn_std_split_sgdm",    "QNN-Std 3->8->6->1 Split (SGD+M)",       TASK_ID, TQNN,  8u,  6u, PK4, RDR, LSTD, ASPL, SGDM, 0.030f, 0.9f, 1e-5f, 400u, 192u }, \
	{ "dff_12_32_24_1_sgdm",   "DFF 12->32->24->1 [width=] (SGD+M)",     TASK_ID, TDFF, 32u, 24u, PK4, RDR, LSTD, ASPL, SGDM, 0.030f, 0.9f, 1e-5f, 400u, 192u }, \
	{ "dff_12_14_11_1_sgdm",   "DFF 12->14->11->1 [param=] (SGD+M)",     TASK_ID, TDFF, 14u, 11u, PK4, RDR, LSTD, ASPL, SGDM, 0.030f, 0.9f, 1e-5f, 400u, 192u }

static const ModelSpec kSpecs[] = {
	// === XOR Gate (4 samples, batch=4, TANH -> SIGMOID, 20 seeds) ===
	{ "qnn_1_4_1_sgdm",       "QNN 1->4->1 (SGD+M)",             "xor",    TQNN,  4u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.20f, 0.9f, 0.0f,  800u, 0u },
	{ "dff_2_16_1_sgdm",      "DFF 2->16->1 [width=] (SGD+M)",   "xor",    TDFF, 16u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.20f, 0.9f, 0.0f,  800u, 0u },
	{ "dff_2_12_1_sgdm",      "DFF 2->12->1 [param=] (SGD+M)",   "xor",    TDFF, 12u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.20f, 0.9f, 0.0f,  800u, 0u },
	{ "qnn_1_4_1_adamw",      "QNN 1->4->1 (AdamW)",             "xor",    TQNN,  4u, 0u, PK4, RDR, LSTD, ASPL, ADMW, 0.05f, 0.0f, 0.0f,  800u, 0u },
	{ "qnn_1_1_1_sgdm",       "QNN 1->1->1 (SGD+M)",             "xor",    TQNN,  1u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.20f, 0.9f, 0.0f, 1500u, 0u },
	{ "dff_2_4_1_sgdm",       "DFF 2->4->1 [width=] (SGD+M)",    "xor",    TDFF,  4u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.20f, 0.9f, 0.0f, 1500u, 0u },
	{ "dff_2_3_1_sgdm",       "DFF 2->3->1 [param=] (SGD+M)",    "xor",    TDFF,  3u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.20f, 0.9f, 0.0f, 1500u, 0u },
	{ "dff_2_2_1_sgdm",       "DFF 2->2->1 [min] (SGD+M)",       "xor",    TDFF,  2u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.20f, 0.9f, 0.0f, 1500u, 0u },
	{ "qnn_1_1_1_adamw",      "QNN 1->1->1 (AdamW)",             "xor",    TQNN,  1u, 0u, PK4, RDR, LSTD, ASPL, ADMW, 0.05f, 0.0f, 0.0f, 1500u, 0u },

	// === Iris 80/20 (120 train / 30 test, batch=16, 5 splits x 5 seeds) ===
	{ "qnn_1_6_3_real_sgdm",  "QNN 1->6->3 Real (SGD+M)",        "iris80", TQNN,  6u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.10f, 0.9f, 1e-4f, 300u, 0u },
	{ "qnn_1_6_1_split_sgdm", "QNN 1->6->1 Split (SGD+M)",       "iris80", TQNN,  6u, 0u, PK4, RDS, LSTD, ASPL, SGDM, 0.10f, 0.9f, 1e-4f, 300u, 0u },
	{ "dff_4_24_3_sgdm",      "DFF 4->24->3 [width=] (SGD+M)",   "iris80", TDFF, 24u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.10f, 0.9f, 1e-4f, 300u, 0u },
	{ "dff_4_15_3_sgdm",      "DFF 4->15->3 [param=R] (SGD+M)",  "iris80", TDFF, 15u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.10f, 0.9f, 1e-4f, 300u, 0u },
	{ "dff_4_9_3_sgdm",       "DFF 4->9->3 [param=S] (SGD+M)",   "iris80", TDFF,  9u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.10f, 0.9f, 1e-4f, 300u, 0u },
	{ "dff_4_6_3_sgdm",       "DFF 4->6->3 [nodes=] (SGD+M)",    "iris80", TDFF,  6u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.10f, 0.9f, 1e-4f, 300u, 0u },
	{ "qnn_1_6_3_real_adamw", "QNN 1->6->3 Real (AdamW)",        "iris80", TQNN,  6u, 0u, PK4, RDR, LSTD, ASPL, ADMW, 0.02f, 0.0f, 1e-4f, 300u, 0u },
	{ "qnn_1_6_1_split_adamw","QNN 1->6->1 Split (AdamW)",       "iris80", TQNN,  6u, 0u, PK4, RDS, LSTD, ASPL, ADMW, 0.02f, 0.0f, 1e-4f, 300u, 0u },
	{ "dff_4_15_3_sgdm_lr003","DFF 4->15->3 (SGD+M lr=0.03)",    "iris80", TDFF, 15u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.03f, 0.9f, 1e-4f, 300u, 0u },

	// === Iris 40/60 (60 train / 90 test, batch=16, 5 splits x 5 seeds) ===
	{ "qnn_1_6_3_real_sgdm",  "QNN 1->6->3 Real (SGD+M)",        "iris40", TQNN,  6u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.10f, 0.9f, 1e-4f, 300u, 0u },
	{ "qnn_1_6_1_split_sgdm", "QNN 1->6->1 Split (SGD+M)",       "iris40", TQNN,  6u, 0u, PK4, RDS, LSTD, ASPL, SGDM, 0.10f, 0.9f, 1e-4f, 300u, 0u },
	{ "dff_4_24_3_sgdm",      "DFF 4->24->3 [width=] (SGD+M)",   "iris40", TDFF, 24u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.10f, 0.9f, 1e-4f, 300u, 0u },
	{ "dff_4_15_3_sgdm",      "DFF 4->15->3 [param=R] (SGD+M)",  "iris40", TDFF, 15u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.10f, 0.9f, 1e-4f, 300u, 0u },
	{ "dff_4_9_3_sgdm",       "DFF 4->9->3 [param=S] (SGD+M)",   "iris40", TDFF,  9u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.10f, 0.9f, 1e-4f, 300u, 0u },
	{ "dff_4_6_3_sgdm",       "DFF 4->6->3 [nodes=] (SGD+M)",    "iris40", TDFF,  6u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.10f, 0.9f, 1e-4f, 300u, 0u },
	{ "qnn_1_6_3_real_adamw", "QNN 1->6->3 Real (AdamW)",        "iris40", TQNN,  6u, 0u, PK4, RDR, LSTD, ASPL, ADMW, 0.02f, 0.0f, 1e-4f, 300u, 0u },
	{ "qnn_1_6_1_split_adamw","QNN 1->6->1 Split (AdamW)",       "iris40", TQNN,  6u, 0u, PK4, RDS, LSTD, ASPL, ADMW, 0.02f, 0.0f, 1e-4f, 300u, 0u },
	{ "dff_4_15_3_sgdm_lr003","DFF 4->15->3 (SGD+M lr=0.03)",    "iris40", TDFF, 15u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.03f, 0.9f, 1e-4f, 300u, 0u },

	// === Tier 1A: SO(3) 3D Vector Rotation Sample-Efficiency Curve (N in {32,64,128,256,512}) ===
	SO3_ROT_CURVE_BLOCK(32),
	SO3_ROT_CURVE_BLOCK(64),
	SO3_ROT_CURVE_BLOCK(128),
	SO3_ROT_CURVE_BLOCK(256),
	SO3_ROT_CURVE_BLOCK(512),

	// === Tier 1B: SO(3) 3D Vector Rotation OOD (Opposite Hemisphere + Unseen Angle [pi/2, pi], N=128) ===
	{ "qnn_equiv_polar_adamw", "QNN-SO3-Equiv 2->8->1 Polar (AdamW)", "so3_rot_ood", TQNN,  8u, 0u, PK4, RDS, LEQV, APOL, ADMW, 0.02f, 0.0f, 1e-5f, 400u, 128u },
	{ "qnn_equiv_polar_sgdm",  "QNN-SO3-Equiv 2->8->1 Polar (SGD+M)", "so3_rot_ood", TQNN,  8u, 0u, PK4, RDS, LEQV, APOL, SGDM, 0.05f, 0.9f, 1e-5f, 400u, 128u },
	{ "qnn_std_polar_adamw",   "QNN-Std 2->8->1 Polar (AdamW)",       "so3_rot_ood", TQNN,  8u, 0u, PK4, RDS, LSTD, APOL, ADMW, 0.02f, 0.0f, 1e-5f, 400u, 128u },
	{ "qnn_std_split_adamw",   "QNN-Std 2->8->1 Split (AdamW)",       "so3_rot_ood", TQNN,  8u, 0u, PK4, RDS, LSTD, ASPL, ADMW, 0.02f, 0.0f, 1e-5f, 400u, 128u },
	{ "qnn_std_split_sgdm",    "QNN-Std 2->8->1 Split (SGD+M)",       "so3_rot_ood", TQNN,  8u, 0u, PK4, RDS, LSTD, ASPL, SGDM, 0.05f, 0.9f, 1e-5f, 400u, 128u },
	{ "dff_8_32_4_sgdm",       "DFF 8->32->4 [width=] (SGD+M)",       "so3_rot_ood", TDFF, 32u, 0u, PK4, RDS, LSTD, ASPL, SGDM, 0.05f, 0.9f, 1e-5f, 400u, 128u },
	{ "dff_8_10_4_sgdm",       "DFF 8->10->4 [param=] (SGD+M)",       "so3_rot_ood", TDFF, 10u, 0u, PK4, RDS, LSTD, ASPL, SGDM, 0.05f, 0.9f, 1e-5f, 400u, 128u },

	// === Tier 1C: SO(3) Quaternion Composition q1 (x) q2 (In-Dist & OOD Rotated Frame, N=128) ===
	{ "qnn_equiv_polar_adamw", "QNN-SO3-Equiv 2->6->1 Polar (AdamW)", "so3_comp_id",  TQNN,  6u, 0u, PK4, RDS, LEQV, APOL, ADMW, 0.02f, 0.0f, 1e-5f, 400u, 128u },
	{ "qnn_equiv_polar_sgdm",  "QNN-SO3-Equiv 2->6->1 Polar (SGD+M)", "so3_comp_id",  TQNN,  6u, 0u, PK4, RDS, LEQV, APOL, SGDM, 0.05f, 0.9f, 1e-5f, 400u, 128u },
	{ "qnn_std_polar_adamw",   "QNN-Std 2->6->1 Polar (AdamW)",       "so3_comp_id",  TQNN,  6u, 0u, PK4, RDS, LSTD, APOL, ADMW, 0.02f, 0.0f, 1e-5f, 400u, 128u },
	{ "qnn_std_split_adamw",   "QNN-Std 2->6->1 Split (AdamW)",       "so3_comp_id",  TQNN,  6u, 0u, PK4, RDS, LSTD, ASPL, ADMW, 0.02f, 0.0f, 1e-5f, 400u, 128u },
	{ "qnn_std_split_sgdm",    "QNN-Std 2->6->1 Split (SGD+M)",       "so3_comp_id",  TQNN,  6u, 0u, PK4, RDS, LSTD, ASPL, SGDM, 0.05f, 0.9f, 1e-5f, 400u, 128u },
	{ "dff_8_24_4_sgdm",       "DFF 8->24->4 [width=] (SGD+M)",       "so3_comp_id",  TDFF, 24u, 0u, PK4, RDS, LSTD, ASPL, SGDM, 0.05f, 0.9f, 1e-5f, 400u, 128u },
	{ "dff_8_8_4_sgdm",        "DFF 8->8->4 [param=] (SGD+M)",        "so3_comp_id",  TDFF,  8u, 0u, PK4, RDS, LSTD, ASPL, SGDM, 0.05f, 0.9f, 1e-5f, 400u, 128u },

	{ "qnn_equiv_polar_adamw", "QNN-SO3-Equiv 2->6->1 Polar (AdamW)", "so3_comp_ood", TQNN,  6u, 0u, PK4, RDS, LEQV, APOL, ADMW, 0.02f, 0.0f, 1e-5f, 400u, 128u },
	{ "qnn_equiv_polar_sgdm",  "QNN-SO3-Equiv 2->6->1 Polar (SGD+M)", "so3_comp_ood", TQNN,  6u, 0u, PK4, RDS, LEQV, APOL, SGDM, 0.05f, 0.9f, 1e-5f, 400u, 128u },
	{ "qnn_std_polar_adamw",   "QNN-Std 2->6->1 Polar (AdamW)",       "so3_comp_ood", TQNN,  6u, 0u, PK4, RDS, LSTD, APOL, ADMW, 0.02f, 0.0f, 1e-5f, 400u, 128u },
	{ "qnn_std_split_adamw",   "QNN-Std 2->6->1 Split (AdamW)",       "so3_comp_ood", TQNN,  6u, 0u, PK4, RDS, LSTD, ASPL, ADMW, 0.02f, 0.0f, 1e-5f, 400u, 128u },
	{ "qnn_std_split_sgdm",    "QNN-Std 2->6->1 Split (SGD+M)",       "so3_comp_ood", TQNN,  6u, 0u, PK4, RDS, LSTD, ASPL, SGDM, 0.05f, 0.9f, 1e-5f, 400u, 128u },
	{ "dff_8_24_4_sgdm",       "DFF 8->24->4 [width=] (SGD+M)",       "so3_comp_ood", TDFF, 24u, 0u, PK4, RDS, LSTD, ASPL, SGDM, 0.05f, 0.9f, 1e-5f, 400u, 128u },
	{ "dff_8_8_4_sgdm",        "DFF 8->8->4 [param=] (SGD+M)",        "so3_comp_ood", TDFF,  8u, 0u, PK4, RDS, LSTD, ASPL, SGDM, 0.05f, 0.9f, 1e-5f, 400u, 128u },

	// === Tier 1D: In-repo datasets/xorparity3.csv (8 rows, exact-one-bit sum==1, 20 seeds) ===
	{ "qnn_1_4_1_split_sgdm",  "QNN 1->4->1 Split (SGD+M)",           "xorparity3",   TQNN,  4u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.20f, 0.9f, 0.0f, 1000u, 0u },
	{ "qnn_1_4_1_polar_sgdm",  "QNN 1->4->1 Polar (SGD+M)",           "xorparity3",   TQNN,  4u, 0u, PK4, RDR, LSTD, APOL, SGDM, 0.20f, 0.9f, 0.0f, 1000u, 0u },
	{ "qnn_1_4_1_split_adamw", "QNN 1->4->1 Split (AdamW)",           "xorparity3",   TQNN,  4u, 0u, PK4, RDR, LSTD, ASPL, ADMW, 0.03f, 0.0f, 0.0f, 1000u, 0u },
	{ "dff_3_16_1_sgdm",       "DFF 3->16->1 [width=] (SGD+M)",       "xorparity3",   TDFF, 16u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.20f, 0.9f, 0.0f, 1000u, 0u },
	{ "dff_3_10_1_sgdm",       "DFF 3->10->1 [param=] (SGD+M)",       "xorparity3",   TDFF, 10u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.20f, 0.9f, 0.0f, 1000u, 0u },
	{ "qnn_1_2_1_split_sgdm",  "QNN 1->2->1 Split (SGD+M)",           "xorparity3",   TQNN,  2u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.20f, 0.9f, 0.0f, 1500u, 0u },
	{ "dff_3_5_1_sgdm",        "DFF 3->5->1 [param=] (SGD+M)",        "xorparity3",   TDFF,  5u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.20f, 0.9f, 0.0f, 1500u, 0u },

	// === Tier 1E: Synthetic 3-bit Odd Parity (8 rows, 20 seeds) ===
	{ "qnn_1_4_1_split_sgdm",  "QNN 1->4->1 Split (SGD+M)",           "parity3_odd",  TQNN,  4u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.20f, 0.9f, 0.0f, 1000u, 3u },
	{ "qnn_1_4_1_polar_sgdm",  "QNN 1->4->1 Polar (SGD+M)",           "parity3_odd",  TQNN,  4u, 0u, PK4, RDR, LSTD, APOL, SGDM, 0.20f, 0.9f, 0.0f, 1000u, 3u },
	{ "qnn_1_4_1_split_adamw", "QNN 1->4->1 Split (AdamW)",           "parity3_odd",  TQNN,  4u, 0u, PK4, RDR, LSTD, ASPL, ADMW, 0.03f, 0.0f, 0.0f, 1000u, 3u },
	{ "dff_3_16_1_sgdm",       "DFF 3->16->1 [width=] (SGD+M)",       "parity3_odd",  TDFF, 16u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.20f, 0.9f, 0.0f, 1000u, 3u },
	{ "dff_3_10_1_sgdm",       "DFF 3->10->1 [param=] (SGD+M)",       "parity3_odd",  TDFF, 10u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.20f, 0.9f, 0.0f, 1000u, 3u },
	{ "qnn_1_2_1_split_sgdm",  "QNN 1->2->1 Split (SGD+M)",           "parity3_odd",  TQNN,  2u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.20f, 0.9f, 0.0f, 1500u, 3u },
	{ "dff_3_5_1_sgdm",        "DFF 3->5->1 [param=] (SGD+M)",        "parity3_odd",  TDFF,  5u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.20f, 0.9f, 0.0f, 1500u, 3u },

	// === Tier 1F: Synthetic 4-bit Odd Parity (16 rows, 20 seeds) ===
	{ "qnn_1_6_1_split_sgdm",  "QNN 1->6->1 Split (SGD+M)",           "parity4",      TQNN,  6u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.20f, 0.9f, 0.0f, 1200u, 4u },
	{ "qnn_1_6_1_polar_sgdm",  "QNN 1->6->1 Polar (SGD+M)",           "parity4",      TQNN,  6u, 0u, PK4, RDR, LSTD, APOL, SGDM, 0.20f, 0.9f, 0.0f, 1200u, 4u },
	{ "qnn_1_6_1_split_adamw", "QNN 1->6->1 Split (AdamW)",           "parity4",      TQNN,  6u, 0u, PK4, RDR, LSTD, ASPL, ADMW, 0.03f, 0.0f, 0.0f, 1200u, 4u },
	{ "dff_4_24_1_sgdm",       "DFF 4->24->1 [width=] (SGD+M)",       "parity4",      TDFF, 24u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.20f, 0.9f, 0.0f, 1200u, 4u },
	{ "dff_4_12_1_sgdm",       "DFF 4->12->1 [param=] (SGD+M)",       "parity4",      TDFF, 12u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.20f, 0.9f, 0.0f, 1200u, 4u },
	{ "qnn_1_3_1_split_sgdm",  "QNN 1->3->1 Split (SGD+M)",           "parity4",      TQNN,  3u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.20f, 0.9f, 0.0f, 1500u, 4u },
	{ "dff_4_6_1_sgdm",        "DFF 4->6->1 [param=] (SGD+M)",        "parity4",      TDFF,  6u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.20f, 0.9f, 0.0f, 1500u, 4u },

	// === Tier 1G: In-repo Dow Jones OHLC (360 Q1 train -> 390 Q2 test, 5 seeds) ===
	{ "qnn_1_4_1_split_sgdm",  "QNN 1->4->1 Split (SGD+M)",           "dow_ohlc",     TQNN,  4u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.05f, 0.9f, 1e-4f, 200u, 0u },
	{ "qnn_1_4_1_polar_sgdm",  "QNN 1->4->1 Polar (SGD+M)",           "dow_ohlc",     TQNN,  4u, 0u, PK4, RDR, LSTD, APOL, SGDM, 0.05f, 0.9f, 1e-4f, 200u, 0u },
	{ "qnn_1_4_1_equiv_sgdm",  "QNN-SO3-Equiv 1->4->1 Polar (SGD+M lr=0.01)", "dow_ohlc", TQNN, 4u, 0u, PK4, RDR, LEQV, APOL, SGDM, 0.01f, 0.9f, 1e-4f, 200u, 0u },
	{ "qnn_1_4_1_split_adamw", "QNN 1->4->1 Split (AdamW)",           "dow_ohlc",     TQNN,  4u, 0u, PK4, RDR, LSTD, ASPL, ADMW, 0.01f, 0.0f, 1e-4f, 200u, 0u },
	{ "qnn_1_4_1_polar_adamw", "QNN 1->4->1 Polar (AdamW)",           "dow_ohlc",     TQNN,  4u, 0u, PK4, RDR, LSTD, APOL, ADMW, 0.01f, 0.0f, 1e-4f, 200u, 0u },
	{ "qnn_1_4_1_equiv_adamw", "QNN-SO3-Equiv 1->4->1 Polar (AdamW)", "dow_ohlc",     TQNN,  4u, 0u, PK4, RDR, LEQV, APOL, ADMW, 0.01f, 0.0f, 1e-4f, 200u, 0u },
	{ "dff_4_16_1_sgdm",       "DFF 4->16->1 [width=] (SGD+M)",       "dow_ohlc",     TDFF, 16u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.05f, 0.9f, 1e-4f, 200u, 0u },
	{ "dff_4_8_1_sgdm",        "DFF 4->8->1 [param=] (SGD+M)",        "dow_ohlc",     TDFF,  8u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.05f, 0.9f, 1e-4f, 200u, 0u },
	{ "dff_4_16_1_sgdm_lr001", "DFF 4->16->1 [width=] (SGD+M lr=0.01)","dow_ohlc",    TDFF, 16u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.01f, 0.9f, 1e-4f, 200u, 0u },
	{ "dff_4_8_1_sgdm_lr001",  "DFF 4->8->1 [param=] (SGD+M lr=0.01)", "dow_ohlc",    TDFF,  8u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.01f, 0.9f, 1e-4f, 200u, 0u },

	// === Tier 1H: In-repo Wine (178 rows, 13 feats, 3 classes, 5 splits x 5 seeds = 25 runs) ===
	{ "qnn_4_6_3_pack4_sgdm",       "QNN 4->6->3 PACK4 Real (SGD+M, wd=1e-4)",  "wine", TQNN,  6u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.10f, 0.9f, 1e-4f, 300u, 0u },
	{ "qnn_4_6_3_pack4_sgdm_wd1e2", "QNN 4->6->3 PACK4 Real (SGD+M, wd=1e-2)",  "wine", TQNN,  6u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.10f, 0.9f, 1e-2f, 300u, 0u },
	{ "qnn_4_6_3_pack4_sgdm_ep60",  "QNN 4->6->3 PACK4 Real (SGD+M, 60 ep)",    "wine", TQNN,  6u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.10f, 0.9f, 1e-4f,  60u, 0u },
	{ "qnn_4_6_1_pack4_split_wd1e2","QNN 4->6->1 PACK4 Split (SGD+M, wd=1e-2)", "wine", TQNN,  6u, 0u, PK4, RDS, LSTD, ASPL, SGDM, 0.10f, 0.9f, 1e-2f, 300u, 0u },
	{ "qnn_4_6_3_pack4_adamw_wd1e2","QNN 4->6->3 PACK4 Real (AdamW, wd=1e-2)",  "wine", TQNN,  6u, 0u, PK4, RDR, LSTD, ASPL, ADMW, 0.02f, 0.0f, 1e-2f, 300u, 0u },
	{ "qnn_13_6_3_1pf_sgdm",        "QNN 13->6->3 1-Per-Feat (SGD+M, wd=1e-4)", "wine", TQNN,  6u, 0u, PK1, RDR, LSTD, ASPL, SGDM, 0.10f, 0.9f, 1e-4f, 300u, 0u },
	{ "qnn_13_6_3_1pf_sgdm_wd1e2",  "QNN 13->6->3 1-Per-Feat (SGD+M, wd=1e-2)", "wine", TQNN,  6u, 0u, PK1, RDR, LSTD, ASPL, SGDM, 0.10f, 0.9f, 1e-2f, 300u, 0u },
	{ "dff_13_24_3_sgdm",           "DFF 13->24->3 [width=] (SGD+M, wd=1e-4)",  "wine", TDFF, 24u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.10f, 0.9f, 1e-4f, 300u, 0u },
	{ "dff_13_12_3_sgdm",           "DFF 13->12->3 [param=] (SGD+M, wd=1e-4)",  "wine", TDFF, 12u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.10f, 0.9f, 1e-4f, 300u, 0u },
	{ "dff_13_12_3_sgdm_wd1e2",     "DFF 13->12->3 [param=] (SGD+M, wd=1e-2)",  "wine", TDFF, 12u, 0u, PK4, RDR, LSTD, ASPL, SGDM, 0.10f, 0.9f, 1e-2f, 300u, 0u },

	// === Tier 2A: Triatomic H2O Molecular PES + Central-Atom Force (ID & OOD arbitrary 3D rotations) ===
	TIER2_JOINT4_BLOCK("mol_pes_id",  0.015f, 0.03f),
	TIER2_JOINT4_BLOCK("mol_pes_ood", 0.015f, 0.03f),

	// === Tier 2B: Chiral Molecular Center (Planar / R / S Enantiomer 3-class, ID & OOD) ===
	TIER2_CHIRAL_BLOCK("mol_chiral_id"),
	TIER2_CHIRAL_BLOCK("mol_chiral_ood"),

	// === Tier 2C: Quantum Slater-Koster p-p Molecular Orbital Overlap (ID & OOD hemisphere) ===
	TIER2_SLATER_BLOCK("orbital_slater_id"),
	TIER2_SLATER_BLOCK("orbital_slater_ood"),

	// === Tier 2D: 3D Keplerian Orbital Energy + Laplace-Runge-Lenz Eccentricity Vector (ID & OOD plane) ===
	TIER2_JOINT4_BLOCK("orbital_kepler_id",  0.010f, 0.02f),
	TIER2_JOINT4_BLOCK("orbital_kepler_ood", 0.010f, 0.02f)
};

#undef PK4
#undef PK1
#undef RDR
#undef RDS
#undef LSTD
#undef LEQV
#undef ASPL
#undef APOL
#undef SGDM
#undef ADMW
#undef TQNN
#undef TDFF
#undef SO3_ROT_CURVE_BLOCK
#undef TIER2_JOINT4_BLOCK
#undef TIER2_CHIRAL_BLOCK
#undef TIER2_SLATER_BLOCK

static const size_t kNumSpecs = sizeof(kSpecs) / sizeof(kSpecs[0]);

static double now_seconds()
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

static size_t current_heap_bytes()
{
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33))
	struct mallinfo2 mi = mallinfo2();
	return mi.uordblks;
#else
	struct mallinfo mi = mallinfo();
	return (mi.uordblks > 0) ? static_cast<size_t>(mi.uordblks) : 0u;
#endif
}

static long peak_rss_kib()
{
	struct rusage ru;
	if (getrusage(RUSAGE_SELF, &ru) == 0)
		return ru.ru_maxrss;
	return 0L;
}

static void update_heap_delta(size_t baseline, size_t& maxDelta)
{
	const size_t cur = current_heap_bytes();
	if (cur > baseline)
	{
		const size_t d = cur - baseline;
		if (d > maxDelta)
			maxDelta = d;
	}
}

static double vec_mean(const std::vector<double>& v)
{
	if (v.empty())
		return 0.0;
	double s = 0.0;
	for (size_t i = 0u; i < v.size(); ++i)
		s += v[i];
	return s / static_cast<double>(v.size());
}

static double vec_std(const std::vector<double>& v)
{
	if (v.empty())
		return 0.0;
	const double m = vec_mean(v);
	double s2 = 0.0;
	for (size_t i = 0u; i < v.size(); ++i)
	{
		const double d = v[i] - m;
		s2 += d * d;
	}
	return std::sqrt(s2 / static_cast<double>(v.size()));
}

static double vec_median(std::vector<double> v)
{
	if (v.empty())
		return 0.0;
	std::sort(v.begin(), v.end());
	const size_t n = v.size();
	if ((n & 1u) == 1u)
		return v[n / 2u];
	return 0.5 * (v[n / 2u - 1u] + v[n / 2u]);
}

static glades::NNInfo* make_net_info(const char* name,
                                     int batchSize,
                                     float lr,
                                     float momentum,
                                     float wd2,
                                     unsigned int hiddenSize,
                                     unsigned int hiddenSize2,
                                     int hiddenAct,
                                     int outSize,
                                     int outType,
                                     int outAct)
{
	shmea::GPointer<glades::InputLayerInfo> in(new glades::InputLayerInfo(
	    batchSize, lr, momentum, 0.0f, wd2, 0.0f, hiddenAct, 1.0f));
	std::vector< shmea::GPointer<glades::HiddenLayerInfo> > hidden;
	if (hiddenSize2 > 0u)
	{
		hidden.push_back(shmea::GPointer<glades::HiddenLayerInfo>(new glades::HiddenLayerInfo(
		    static_cast<int>(hiddenSize), lr, momentum, 0.0f, wd2, 0.0f, hiddenAct, 1.0f)));
		hidden.push_back(shmea::GPointer<glades::HiddenLayerInfo>(new glades::HiddenLayerInfo(
		    static_cast<int>(hiddenSize2), lr, momentum, 0.0f, wd2, 0.0f, outAct, 1.0f)));
	}
	else
	{
		hidden.push_back(shmea::GPointer<glades::HiddenLayerInfo>(new glades::HiddenLayerInfo(
		    static_cast<int>(hiddenSize), lr, momentum, 0.0f, wd2, 0.0f, outAct, 1.0f)));
	}
	shmea::GPointer<glades::OutputLayerInfo> out(new glades::OutputLayerInfo(
	    outSize, outType));
	return new glades::NNInfo(shmea::GString(name), in, hidden, out);
}

static void compute_param_and_state_sizes(const ModelSpec& spec,
                                          unsigned int realIn,
                                          unsigned int realOut,
                                          unsigned int& outActiveParams,
                                          unsigned int& outStoredParams,
                                          unsigned int& outCoreStateBytes)
{
	if (spec.netType == glades::NNetwork::TYPE_QNN)
	{
		const glades::QuaternionInputPackingMode pMode =
		    (spec.qnnPacking == glades::QuaternionConfig::ONE_QUAT_PER_FEATURE)
		        ? glades::QUAT_ONE_PER_FEATURE
		        : glades::QUAT_PACK_GROUPS_OF_4;
		const unsigned int qIn = glades::quaternionInputCount(realIn, pMode);
		const glades::QuaternionOutputReadoutMode rMode =
		    (spec.qnnReadout == glades::QuaternionConfig::READOUT_SPLIT_COMPONENTS)
		        ? glades::QUAT_READOUT_SPLIT_COMPONENTS
		        : glades::QUAT_READOUT_REAL_PART;
		const unsigned int qOut = glades::quaternionOutputCount(realOut, rMode);
		const bool isEquiv = (spec.qnnLayerMode == glades::QuaternionConfig::LAYER_SO3_EQUIVARIANT);

		std::vector<unsigned int> layers;
		layers.push_back(qIn);
		layers.push_back(spec.hiddenSize);
		if (spec.hiddenSize2 > 0u)
			layers.push_back(spec.hiddenSize2);
		layers.push_back(qOut);

		unsigned int stored = 0u;
		unsigned int active = 0u;
		unsigned int transBytes = 0u;
		unsigned int allSum = layers[0];
		unsigned int nonInputSum = 0u;

		for (size_t l = 0u; l + 1u < layers.size(); ++l)
		{
			const unsigned int inW = layers[l];
			const unsigned int outW = layers[l + 1u];
			const bool isOutLayer = (l + 2u == layers.size());
			const unsigned int w = outW * inW * 4u;
			const unsigned int bStored = outW * 4u;
			const unsigned int bActive = isEquiv ? (outW * 3u) : (isOutLayer ? realOut : bStored);

			stored += w + bStored;
			active += w + bActive;
			transBytes += 4u * (outW * inW + outW) * 16u;
			allSum += outW;
			nonInputSum += outW;
		}

		outStoredParams = stored;
		outActiveParams = active;

		// TensorQNNState: per transition W,vW,v2W,gW + bias,vBias,v2Bias,gBias (16 B/quat)
		// plus scratch z (non-input layers), a (all layers), delta (all layers), outReadout (4 B/float),
		// plus eqL, eqR for LAYER_SO3_EQUIVARIANT.
		const unsigned int eqScratchBytes = isEquiv ? (2u * nonInputSum * 16u) : 0u;
		const unsigned int scratchBytes =
		    nonInputSum * 16u +
		    2u * allSum * 16u +
		    realOut * 4u +
		    eqScratchBytes;
		outCoreStateBytes = transBytes + scratchBytes;
	}
	else
	{
		std::vector<unsigned int> layers;
		layers.push_back(realIn);
		layers.push_back(spec.hiddenSize);
		if (spec.hiddenSize2 > 0u)
			layers.push_back(spec.hiddenSize2);
		layers.push_back(realOut);

		unsigned int totalParams = 0u;
		unsigned int transBytes = 0u;
		unsigned int allSum = layers[0];
		unsigned int nonInputSum = 0u;

		for (size_t l = 0u; l + 1u < layers.size(); ++l)
		{
			const unsigned int inW = layers[l];
			const unsigned int outW = layers[l + 1u];
			const unsigned int w = outW * inW;
			const unsigned int b = outW;
			totalParams += w + b;
			transBytes += (3u * w + 2u * b) * 4u;
			allSum += outW;
			nonInputSum += outW;
		}

		outStoredParams = totalParams;
		outActiveParams = totalParams;

		// TensorDFFState core (excluding helm/aster): per transition W,vW,gW + bias,gBias (4 B/float)
		// plus scratch a (all layers) and delta (non-input layers).
		const unsigned int scratchBytes = (allSum + nonInputSum) * 4u;
		outCoreStateBytes = transBytes + scratchBytes;
	}
}

static size_t measure_weights_bin_bytes(glades::NNetwork& net, glades::DataInput* di)
{
	char nameBuf[128];
	std::sprintf(nameBuf, "bench_qnn_tmp_%ld", static_cast<long>(getpid()));
	const std::string modelName(nameBuf);
	const std::string dir = "database/models/" + modelName;

	(void)::mkdir("database", 0755);
	(void)::mkdir("database/models", 0755);

	glades::NNetworkStatus st = net.saveModel(modelName, di);
	size_t bytes = 0u;
	if (st.ok())
	{
		struct stat sb;
		const std::string wpath = dir + "/weights.bin";
		if (::stat(wpath.c_str(), &sb) == 0 && sb.st_size > 0)
			bytes = static_cast<size_t>(sb.st_size);
	}

	(void)::remove((dir + "/manifest.txt").c_str());
	(void)::remove((dir + "/nninfo.csv").c_str());
	(void)::remove((dir + "/weights.bin").c_str());
	(void)::rmdir(dir.c_str());
	return bytes;
}

struct EpochTracker : public glades::ITrainingCallbacks
{
	unsigned int numClasses;
	bool isBinaryTruthTable;
	bool isRegression;
	size_t heapBaseline;
	size_t* maxHeapDelta;

	unsigned int epBce017;
	unsigned int epTrain95;
	unsigned int epTrain98;
	double lastError;

	EpochTracker(unsigned int nCls, bool binTruth, bool regTask, size_t base, size_t* maxDelta)
	    : numClasses(nCls),
	      isBinaryTruthTable(binTruth),
	      isRegression(regTask),
	      heapBaseline(base),
	      maxHeapDelta(maxDelta),
	      epBce017(0u),
	      epTrain95(0u),
	      epTrain98(0u),
	      lastError(0.0)
	{
	}

	virtual void onRunStart(const glades::NNetwork&, int)
	{
		if (maxHeapDelta)
			update_heap_delta(heapBaseline, *maxHeapDelta);
	}

	virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
	{
		lastError = static_cast<double>(m.totalError);
		const unsigned int ep = static_cast<unsigned int>(m.epoch);
		if (isBinaryTruthTable)
		{
			if (epBce017 == 0u && m.totalError <= 0.17f)
				epBce017 = ep;
		}
		else if (!isRegression)
		{
			const double top1 = 100.0 - 0.5 * static_cast<double>(numClasses) * (100.0 - static_cast<double>(m.totalAccuracy));
			if (epTrain95 == 0u && top1 >= 95.0)
				epTrain95 = ep;
			if (epTrain98 == 0u && top1 >= 98.0)
				epTrain98 = ep;
		}
		if (maxHeapDelta && (ep == 1u || ep == 10u || (ep % 100u) == 0u))
			update_heap_delta(heapBaseline, *maxHeapDelta);
		return false;
	}

	virtual void onRunEnd(const glades::NNetwork&, int)
	{
		if (maxHeapDelta)
			update_heap_delta(heapBaseline, *maxHeapDelta);
	}
};

struct EvalStats
{
	double accPct;
	double loss;
	double brier;
	double f1[3];
	double macroF1;
	unsigned int correct;
	unsigned int total;

	EvalStats() : accPct(0.0), loss(0.0), brier(0.0), macroF1(0.0), correct(0u), total(0u)
	{
		f1[0] = f1[1] = f1[2] = 0.0;
	}
};

static bool eval_split_exact(glades::NNetwork& net,
                             const glades::NumberInput& di,
                             bool useTrainSplit,
                             const unsigned int* classOrderMap,
                             EvalStats& out)
{
	const unsigned int nRows = useTrainSplit ? di.getTrainSize() : di.getTestSize();
	const unsigned int outSize = net.getNNInfo()->getOutputLayerSize();
	const unsigned int nClasses = (outSize > 1u) ? outSize : 2u;
	if (nRows == 0u || nClasses > 3u)
		return false;

	unsigned int cm[3][3];
	for (unsigned int i = 0u; i < 3u; ++i)
		for (unsigned int j = 0u; j < 3u; ++j)
			cm[i][j] = 0u;

	unsigned int correct = 0u;
	double lossSum = 0.0;
	double brierSum = 0.0;
	const double eps = 1e-7;

	glades::NumberInput single;
	single.testMatrix = shmea::GMatrix(1, shmea::GVector<float>(di.getFeatureCount(), 0.0f));
	single.testExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(outSize, 0.0f));
	single.trainMatrix = single.testMatrix;
	single.trainExpectedMatrix = single.testExpectedMatrix;
	single.loaded = true;

	for (unsigned int r = 0u; r < nRows; ++r)
	{
		const float* x = NULL;
		unsigned int xN = 0u;
		const float* y = NULL;
		unsigned int yN = 0u;
		if (useTrainSplit)
		{
			di.getTrainRowView(r, x, xN);
			di.getTrainExpectedRowView(r, y, yN);
		}
		else
		{
			di.getTestRowView(r, x, xN);
			di.getTestExpectedRowView(r, y, yN);
		}
		for (unsigned int c = 0u; c < xN; ++c)
		{
			single.testMatrix[0][c] = x[c];
			single.trainMatrix[0][c] = x[c];
		}
		for (unsigned int c = 0u; c < yN; ++c)
		{
			single.testExpectedMatrix[0][c] = y[c];
			single.trainExpectedMatrix[0][c] = y[c];
		}

		glades::NNetworkStatus st = net.test(&single);
		if (!st.ok())
			return false;
		shmea::GList res = net.getResults();
		if (res.size() != outSize * 2u)
			return false;

		double probs[3];
		for (unsigned int k = 0u; k < outSize; ++k)
		{
			const double p = static_cast<double>(res.getFloat(k * 2u + 1u));
			if (!std::isfinite(p))
				return false;
			probs[k] = p;
		}

		unsigned int expRaw = 0u;
		unsigned int predRaw = 0u;
		if (outSize == 1u)
		{
			const double p1 = std::min(1.0 - eps, std::max(eps, probs[0]));
			const double yt = (y[0] >= 0.5f) ? 1.0 : 0.0;
			expRaw = (yt >= 0.5) ? 1u : 0u;
			predRaw = (probs[0] >= 0.5) ? 1u : 0u;
			lossSum += -(yt * std::log(p1) + (1.0 - yt) * std::log(1.0 - p1));
			const double diff = probs[0] - yt;
			brierSum += diff * diff;
		}
		else
		{
			expRaw = static_cast<unsigned int>(glades::GMath::argmax(y, outSize));
			predRaw = 0u;
			for (unsigned int k = 1u; k < outSize; ++k)
			{
				if (probs[k] > probs[predRaw])
					predRaw = k;
			}
			const double pTrue = std::min(1.0 - eps, std::max(eps, probs[expRaw]));
			lossSum += -std::log(pTrue);
			for (unsigned int k = 0u; k < outSize; ++k)
			{
				const double yt = (k == expRaw) ? 1.0 : 0.0;
				const double diff = probs[k] - yt;
				brierSum += diff * diff;
			}
		}

		if (expRaw == predRaw)
			++correct;
		if (expRaw < nClasses && predRaw < nClasses)
			++cm[expRaw][predRaw];
	}

	out.correct = correct;
	out.total = nRows;
	out.accPct = (100.0 * static_cast<double>(correct)) / static_cast<double>(nRows);
	out.loss = lossSum / static_cast<double>(nRows);
	out.brier = brierSum / static_cast<double>(nRows);

	double f1Sum = 0.0;
	for (unsigned int cCanon = 0u; cCanon < nClasses; ++cCanon)
	{
		const unsigned int cRaw = classOrderMap ? classOrderMap[cCanon] : cCanon;
		const unsigned int tp = cm[cRaw][cRaw];
		unsigned int fp = 0u;
		unsigned int fn = 0u;
		for (unsigned int j = 0u; j < nClasses; ++j)
		{
			if (j != cRaw)
			{
				fp += cm[j][cRaw];
				fn += cm[cRaw][j];
			}
		}
		const unsigned int denom = 2u * tp + fp + fn;
		const double f1Val = (denom > 0u) ? (100.0 * 2.0 * static_cast<double>(tp) / static_cast<double>(denom)) : 0.0;
		out.f1[cCanon] = f1Val;
		f1Sum += f1Val;
	}
	out.macroF1 = f1Sum / static_cast<double>(nClasses);
	return true;
}

static bool eval_single_row_regression(glades::NNetwork& net,
                                       const glades::NumberInput& di,
                                       bool useTrainSplit,
                                       std::vector< std::vector<float> >& outPreds)
{
	const unsigned int nRows = useTrainSplit ? di.getTrainSize() : di.getTestSize();
	const unsigned int outSize = net.getNNInfo()->getOutputLayerSize();
	outPreds.assign(nRows, std::vector<float>(outSize, 0.0f));

	glades::NumberInput single;
	single.testMatrix = shmea::GMatrix(1, shmea::GVector<float>(di.getFeatureCount(), 0.0f));
	single.testExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(outSize, 0.0f));
	single.trainMatrix = single.testMatrix;
	single.trainExpectedMatrix = single.testExpectedMatrix;
	single.loaded = true;

	for (unsigned int r = 0u; r < nRows; ++r)
	{
		const float* x = NULL;
		unsigned int xN = 0u;
		const float* y = NULL;
		unsigned int yN = 0u;
		if (useTrainSplit)
		{
			di.getTrainRowView(r, x, xN);
			di.getTrainExpectedRowView(r, y, yN);
		}
		else
		{
			di.getTestRowView(r, x, xN);
			di.getTestExpectedRowView(r, y, yN);
		}
		for (unsigned int c = 0u; c < xN; ++c)
		{
			single.testMatrix[0][c] = x[c];
			single.trainMatrix[0][c] = x[c];
		}
		for (unsigned int c = 0u; c < yN; ++c)
		{
			single.testExpectedMatrix[0][c] = y[c];
			single.trainExpectedMatrix[0][c] = y[c];
		}

		glades::NNetworkStatus st = net.test(&single);
		if (!st.ok())
			return false;
		shmea::GList res = net.getResults();
		if (res.size() != outSize * 2u)
			return false;
		for (unsigned int k = 0u; k < outSize; ++k)
		{
			const float p = res.getFloat(k * 2u + 1u);
			if (!std::isfinite(p))
				return false;
			outPreds[r][k] = p;
		}
	}
	return true;
}

static bool eval_multi_quat_net(glades::NNetwork& net,
                                const std::vector<glades::Quaternion>& qIn,
                                unsigned int realOut,
                                std::vector<float>& outVals)
{
	const unsigned int realIn = static_cast<unsigned int>(qIn.size()) * 4u;
	glades::NumberInput single;
	single.testMatrix = shmea::GMatrix(1, shmea::GVector<float>(realIn, 0.0f));
	single.testExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(realOut, 0.0f));
	for (size_t q = 0u; q < qIn.size(); ++q)
	{
		single.testMatrix[0][q * 4u + 0u] = qIn[q].r;
		single.testMatrix[0][q * 4u + 1u] = qIn[q].i;
		single.testMatrix[0][q * 4u + 2u] = qIn[q].j;
		single.testMatrix[0][q * 4u + 3u] = qIn[q].k;
	}
	single.trainMatrix = single.testMatrix;
	single.trainExpectedMatrix = single.testExpectedMatrix;
	single.loaded = true;

	if (!net.test(&single).ok())
		return false;
	shmea::GList res = net.getResults();
	if (res.size() != realOut * 2u)
		return false;
	outVals.assign(realOut, 0.0f);
	for (unsigned int k = 0u; k < realOut; ++k)
	{
		const float v = res.getFloat(k * 2u + 1u);
		if (!std::isfinite(v))
			return false;
		outVals[k] = v;
	}
	return true;
}

static double measure_so3_symmetry_error(glades::NNetwork& net,
                                         unsigned int qInCount,
                                         unsigned int realOut,
                                         bool invariantOutput,
                                         uint64_t seed)
{
	glades::rng::Engine eng;
	glades::rng::seed_engine(eng, seed);
	double maxErr = 0.0;
	std::vector<glades::Quaternion> baseIn(qInCount);
	std::vector<glades::Quaternion> rotIn(qInCount);
	std::vector<float> baseOut;
	std::vector<float> rotInOut;

	for (unsigned int t = 0u; t < 32u; ++t)
	{
		for (unsigned int q = 0u; q < qInCount; ++q)
		{
			baseIn[q] = glades::Quaternion(
			    glades::rng::uniform_float(eng, -1.0f, 1.0f),
			    glades::rng::uniform_float(eng, -1.0f, 1.0f),
			    glades::rng::uniform_float(eng, -1.0f, 1.0f),
			    glades::rng::uniform_float(eng, -1.0f, 1.0f));
		}
		glades::Quaternion u(
		    glades::rng::standard_normal(eng),
		    glades::rng::standard_normal(eng),
		    glades::rng::standard_normal(eng),
		    glades::rng::standard_normal(eng));
		u = u.normalized();

		for (unsigned int q = 0u; q < qInCount; ++q)
			rotIn[q] = glades::Quaternion::conjugateAction(u, baseIn[q]);

		if (!eval_multi_quat_net(net, baseIn, realOut, baseOut) ||
		    !eval_multi_quat_net(net, rotIn, realOut, rotInOut))
			return 1e9;

		double err = 0.0;
		if (invariantOutput || realOut != 4u)
		{
			double sq = 0.0;
			for (unsigned int k = 0u; k < realOut; ++k)
			{
				const double d = static_cast<double>(rotInOut[k]) - static_cast<double>(baseOut[k]);
				sq += d * d;
			}
			err = std::sqrt(sq);
		}
		else
		{
			const glades::Quaternion qBase(baseOut[0], baseOut[1], baseOut[2], baseOut[3]);
			const glades::Quaternion qRotIn(rotInOut[0], rotInOut[1], rotInOut[2], rotInOut[3]);
			const glades::Quaternion qRotBase = glades::Quaternion::conjugateAction(u, qBase);
			err = static_cast<double>((qRotIn - qRotBase).norm());
		}
		if (err > maxErr)
			maxErr = err;
	}
	return maxErr;
}

static double measure_so3_equivariance_error(glades::NNetwork& net, uint64_t seed)
{
	return measure_so3_symmetry_error(net, 2u, 4u, false, seed);
}

static bool eval_joint_scalar_vec3_split(glades::NNetwork& net,
                                         const glades::NumberInput& di,
                                         bool useTrainSplit,
                                         double& outRmse4D,
                                         double& outScalarRmse,
                                         double& outVecAngleDeg,
                                         double& outR2,
                                         double& outMse4D)
{
	std::vector< std::vector<float> > preds;
	if (!eval_single_row_regression(net, di, useTrainSplit, preds))
		return false;

	const unsigned int n = useTrainSplit ? di.getTrainSize() : di.getTestSize();
	double meanY[4] = { 0.0, 0.0, 0.0, 0.0 };
	for (unsigned int r = 0u; r < n; ++r)
	{
		const float* y = NULL;
		unsigned int yN = 0u;
		if (useTrainSplit)
			di.getTrainExpectedRowView(r, y, yN);
		else
			di.getTestExpectedRowView(r, y, yN);
		for (unsigned int c = 0u; c < 4u; ++c)
			meanY[c] += static_cast<double>(y[c]);
	}
	for (unsigned int c = 0u; c < 4u; ++c)
		meanY[c] /= static_cast<double>(n);

	double sse4 = 0.0;
	double sst4 = 0.0;
	double sseScalar = 0.0;
	double sumAngleDeg = 0.0;
	unsigned int angleCount = 0u;
	static const double kRadToDeg = 180.0 / 3.14159265358979323846;

	for (unsigned int r = 0u; r < n; ++r)
	{
		const float* y = NULL;
		unsigned int yN = 0u;
		if (useTrainSplit)
			di.getTrainExpectedRowView(r, y, yN);
		else
			di.getTestExpectedRowView(r, y, yN);

		const double ds = static_cast<double>(preds[r][0]) - static_cast<double>(y[0]);
		sseScalar += ds * ds;

		double dotv = 0.0, np2 = 0.0, ny2 = 0.0;
		for (unsigned int c = 0u; c < 4u; ++c)
		{
			const double p = static_cast<double>(preds[r][c]);
			const double t = static_cast<double>(y[c]);
			const double d = p - t;
			sse4 += d * d;
			const double dy = t - meanY[c];
			sst4 += dy * dy;
			if (c >= 1u)
			{
				dotv += p * t;
				np2 += p * p;
				ny2 += t * t;
			}
		}
		if (np2 > 1e-10 && ny2 > 1e-10)
		{
			double cosA = dotv / std::sqrt(np2 * ny2);
			if (cosA > 1.0) cosA = 1.0;
			if (cosA < -1.0) cosA = -1.0;
			sumAngleDeg += std::acos(cosA) * kRadToDeg;
			++angleCount;
		}
	}

	outMse4D = sse4 / static_cast<double>(n * 4u);
	outRmse4D = std::sqrt(outMse4D);
	outScalarRmse = std::sqrt(sseScalar / static_cast<double>(n));
	outVecAngleDeg = (angleCount > 0u) ? (sumAngleDeg / static_cast<double>(angleCount)) : 0.0;
	outR2 = (sst4 > 1e-12) ? (1.0 - (sse4 / sst4)) : 0.0;
	return true;
}

static bool eval_scalar_reg_split(glades::NNetwork& net,
                                  const glades::NumberInput& di,
                                  bool useTrainSplit,
                                  double& outRmse,
                                  double& outMae,
                                  double& outR2,
                                  double& outMse)
{
	std::vector< std::vector<float> > preds;
	if (!eval_single_row_regression(net, di, useTrainSplit, preds))
		return false;

	const unsigned int n = useTrainSplit ? di.getTrainSize() : di.getTestSize();
	double meanY = 0.0;
	for (unsigned int r = 0u; r < n; ++r)
	{
		const float* y = NULL;
		unsigned int yN = 0u;
		if (useTrainSplit)
			di.getTrainExpectedRowView(r, y, yN);
		else
			di.getTestExpectedRowView(r, y, yN);
		meanY += static_cast<double>(y[0]);
	}
	meanY /= static_cast<double>(n);

	double sse = 0.0;
	double sst = 0.0;
	double sae = 0.0;
	for (unsigned int r = 0u; r < n; ++r)
	{
		const float* y = NULL;
		unsigned int yN = 0u;
		if (useTrainSplit)
			di.getTrainExpectedRowView(r, y, yN);
		else
			di.getTestExpectedRowView(r, y, yN);
		const double p = static_cast<double>(preds[r][0]);
		const double t = static_cast<double>(y[0]);
		const double d = p - t;
		sse += d * d;
		sae += std::fabs(d);
		const double dy = t - meanY;
		sst += dy * dy;
	}
	outMse = sse / static_cast<double>(n);
	outRmse = std::sqrt(outMse);
	outMae = sae / static_cast<double>(n);
	outR2 = (sst > 1e-12) ? (1.0 - (sse / sst)) : 0.0;
	return true;
}

static bool eval_so3_split(glades::NNetwork& net,
                           const glades::NumberInput& di,
                           bool useTrainSplit,
                           bool vector3Only,
                           double& outRmse,
                           double& outAngleDeg,
                           double& outR2,
                           double& outMse)
{
	std::vector< std::vector<float> > preds;
	if (!eval_single_row_regression(net, di, useTrainSplit, preds))
		return false;

	const unsigned int n = useTrainSplit ? di.getTrainSize() : di.getTestSize();
	const unsigned int startLane = vector3Only ? 1u : 0u;
	const unsigned int dims = vector3Only ? 3u : 4u;

	double meanY[4] = { 0.0, 0.0, 0.0, 0.0 };
	for (unsigned int r = 0u; r < n; ++r)
	{
		const float* y = NULL;
		unsigned int yN = 0u;
		if (useTrainSplit)
			di.getTrainExpectedRowView(r, y, yN);
		else
			di.getTestExpectedRowView(r, y, yN);
		for (unsigned int c = startLane; c < 4u; ++c)
			meanY[c] += static_cast<double>(y[c]);
	}
	for (unsigned int c = startLane; c < 4u; ++c)
		meanY[c] /= static_cast<double>(n);

	double sse = 0.0;
	double sst = 0.0;
	double sumAngleDeg = 0.0;
	static const double kRadToDeg = 180.0 / 3.14159265358979323846;

	for (unsigned int r = 0u; r < n; ++r)
	{
		const float* y = NULL;
		unsigned int yN = 0u;
		if (useTrainSplit)
			di.getTrainExpectedRowView(r, y, yN);
		else
			di.getTestExpectedRowView(r, y, yN);

		double dotv = 0.0, np2 = 0.0, ny2 = 0.0;
		for (unsigned int c = startLane; c < 4u; ++c)
		{
			const double p = static_cast<double>(preds[r][c]);
			const double t = static_cast<double>(y[c]);
			const double d = p - t;
			sse += d * d;
			const double dy = t - meanY[c];
			sst += dy * dy;
			dotv += p * t;
			np2 += p * p;
			ny2 += t * t;
		}
		if (np2 > 1e-12 && ny2 > 1e-12)
		{
			double cosA = dotv / std::sqrt(np2 * ny2);
			if (!vector3Only)
				cosA = std::fabs(cosA);
			if (cosA > 1.0) cosA = 1.0;
			if (cosA < -1.0) cosA = -1.0;
			const double rad = vector3Only ? std::acos(cosA) : (2.0 * std::acos(cosA));
			sumAngleDeg += rad * kRadToDeg;
		}
	}

	outMse = sse / static_cast<double>(n * dims);
	outRmse = std::sqrt(outMse);
	outAngleDeg = sumAngleDeg / static_cast<double>(n);
	outR2 = (sst > 1e-12) ? (1.0 - (sse / sst)) : 0.0;
	return true;
}

static bool eval_dow_split(glades::NNetwork& net,
                           const glades::NumberInput& di,
                           bool useTrainSplit,
                           double targetStd,
                           double& outRmseStd,
                           double& outRmseUsd,
                           double& outMaeStd,
                           double& outR2,
                           double& outDirAccPct)
{
	std::vector< std::vector<float> > preds;
	if (!eval_single_row_regression(net, di, useTrainSplit, preds))
		return false;

	const unsigned int n = useTrainSplit ? di.getTrainSize() : di.getTestSize();
	double meanY = 0.0;
	for (unsigned int r = 0u; r < n; ++r)
	{
		const float* y = NULL;
		unsigned int yN = 0u;
		if (useTrainSplit)
			di.getTrainExpectedRowView(r, y, yN);
		else
			di.getTestExpectedRowView(r, y, yN);
		meanY += static_cast<double>(y[0]);
	}
	meanY /= static_cast<double>(n);

	double sse = 0.0;
	double sst = 0.0;
	double sae = 0.0;
	unsigned int dirCorrect = 0u;
	for (unsigned int r = 0u; r < n; ++r)
	{
		const float* x = NULL;
		unsigned int xN = 0u;
		const float* y = NULL;
		unsigned int yN = 0u;
		if (useTrainSplit)
		{
			di.getTrainRowView(r, x, xN);
			di.getTrainExpectedRowView(r, y, yN);
		}
		else
		{
			di.getTestRowView(r, x, xN);
			di.getTestExpectedRowView(r, y, yN);
		}
		const double p = static_cast<double>(preds[r][0]);
		const double t = static_cast<double>(y[0]);
		const double closeCur = static_cast<double>(x[3]);
		const double d = p - t;
		sse += d * d;
		sae += std::fabs(d);
		const double dy = t - meanY;
		sst += dy * dy;

		const bool predUp = (p >= closeCur);
		const bool actUp = (t >= closeCur);
		if (predUp == actUp)
			++dirCorrect;
	}

	outRmseStd = std::sqrt(sse / static_cast<double>(n));
	outRmseUsd = outRmseStd * targetStd;
	outMaeStd = sae / static_cast<double>(n);
	outR2 = (sst > 1e-12) ? (1.0 - (sse / sst)) : 0.0;
	outDirAccPct = (100.0 * static_cast<double>(dirCorrect)) / static_cast<double>(n);
	return true;
}

static void resolve_class_map(const glades::QuaternionInput& di,
                              const char* const canonical[3],
                              unsigned int outMap[3])
{
	outMap[0] = 0u;
	outMap[1] = 1u;
	outMap[2] = 2u;
	const std::vector<std::string>& labels = di.getClassLabels();
	for (unsigned int c = 0u; c < 3u; ++c)
	{
		for (size_t i = 0u; i < labels.size(); ++i)
		{
			if (labels[i] == canonical[c])
			{
				outMap[c] = static_cast<unsigned int>(i);
				break;
			}
		}
	}
}

static void print_tsv_header()
{
	std::printf(
	    "task\tmodel_id\tlabel\tactive_params\tstored_params\tbudget_epochs\t"
	    "runs\tconverged_runs\ttrain_acc_mean\ttrain_acc_std\ttest_acc_mean\ttest_acc_std\tcanon_test_acc\t"
	    "ep_metric1_mean\tep_metric1_med\tep_metric2_mean\t"
	    "train_loss_mean\ttest_loss_mean\ttest_loss_std\ttest_brier_mean\t"
	    "f1_c0\tf1_c1\tf1_c2\tmacro_f1_mean\tmacro_f1_std\t"
	    "train_ms_mean\ttrain_ms_std\tpredict_split_us\tpredict_sample_us\t"
	    "peak_heap_delta_kib\tcore_state_bytes\tweights_bin_bytes\tpeak_rss_kib\t"
	    "reg_rmse_mean\treg_rmse_std\treg_aux_mean\treg_aux_std\treg_r2_mean\tequiv_err_mean\n");
}

static void print_list()
{
	for (size_t i = 0u; i < kNumSpecs; ++i)
		std::printf("%s\t%s\t%s\n", kSpecs[i].task, kSpecs[i].id, kSpecs[i].label);
}

static void configure_net(glades::NNetwork& net,
                          const ModelSpec& spec,
                          uint64_t seed,
                          shmea::GLogger* logger)
{
	net.setLogger(logger);
	net.setSeed(seed);
	glades::TrainingConfig cfg = net.getTrainingConfig();
	cfg.optimizer.type = spec.optimizer;
	cfg.qnn.inputPacking = spec.qnnPacking;
	cfg.qnn.outputReadout = spec.qnnReadout;
	cfg.qnn.layerMode = spec.qnnLayerMode;
	cfg.qnn.activationMode = spec.qnnActMode;
	net.setTrainingConfig(cfg);
}

static int run_worker(const ModelSpec& spec,
                      const std::string& dataDir,
                      const std::string& repoDataDir,
                      unsigned int predictRepeats,
                      bool jsonOut)
{
	shmea::GLogger quietLogger(shmea::GLogger::LOG_FATAL);
	quietLogger.setPrintToConsole(false);

	const std::string task(spec.task);
	const bool isXor = (task == "xor");
	const bool isXorParity3 = (task == "xorparity3");
	const bool isParity3Odd = (task == "parity3_odd");
	const bool isParity4 = (task == "parity4");
	const bool isBinaryTruth = (isXor || isXorParity3 || isParity3Odd || isParity4);
	const bool isIris80 = (task == "iris80");
	const bool isIris40 = (task == "iris40");
	const bool isWine = (task == "wine");
	const bool isTabularCls = (isIris80 || isIris40 || isWine);
	const bool isSO3RotCurve = (task == "so3_rot_curve");
	const bool isSO3RotOod = (task == "so3_rot_ood");
	const bool isSO3CompId = (task == "so3_comp_id");
	const bool isSO3CompOod = (task == "so3_comp_ood");
	const bool isSO3 = (isSO3RotCurve || isSO3RotOod || isSO3CompId || isSO3CompOod);
	const bool isDow = (task == "dow_ohlc");
	const bool isMolPesId = (task == "mol_pes_id");
	const bool isMolPesOod = (task == "mol_pes_ood");
	const bool isMolChiralId = (task == "mol_chiral_id");
	const bool isMolChiralOod = (task == "mol_chiral_ood");
	const bool isSlaterId = (task == "orbital_slater_id");
	const bool isSlaterOod = (task == "orbital_slater_ood");
	const bool isKeplerId = (task == "orbital_kepler_id");
	const bool isKeplerOod = (task == "orbital_kepler_ood");
	const bool isTier2Joint4 = (isMolPesId || isMolPesOod || isKeplerId || isKeplerOod);
	const bool isTier2Chiral = (isMolChiralId || isMolChiralOod);
	const bool isTier2Slater = (isSlaterId || isSlaterOod);

	unsigned int realIn = 4u;
	unsigned int realOut = 3u;
	int batchSize = 16;
	int outType = glades::OutputLayerInfo::CLASSIFICATION;
	int outAct = glades::GMath::LINEAR;

	if (isXor)
	{
		realIn = 2u; realOut = 1u; batchSize = 4;
		outType = glades::OutputLayerInfo::CLASSIFICATION; outAct = glades::GMath::SIGMOID;
	}
	else if (isXorParity3 || isParity3Odd)
	{
		realIn = 3u; realOut = 1u; batchSize = 8;
		outType = glades::OutputLayerInfo::CLASSIFICATION; outAct = glades::GMath::SIGMOID;
	}
	else if (isParity4)
	{
		realIn = 4u; realOut = 1u; batchSize = 16;
		outType = glades::OutputLayerInfo::CLASSIFICATION; outAct = glades::GMath::SIGMOID;
	}
	else if (isWine)
	{
		realIn = 13u; realOut = 3u; batchSize = 16;
		outType = glades::OutputLayerInfo::CLASSIFICATION; outAct = glades::GMath::LINEAR;
	}
	else if (isSO3 || isTier2Joint4)
	{
		realIn = 8u; realOut = 4u; batchSize = 16;
		outType = glades::OutputLayerInfo::REGRESSION; outAct = glades::GMath::LINEAR;
	}
	else if (isTier2Chiral)
	{
		realIn = 12u; realOut = 3u; batchSize = 16;
		outType = glades::OutputLayerInfo::CLASSIFICATION; outAct = glades::GMath::LINEAR;
	}
	else if (isTier2Slater)
	{
		realIn = 12u; realOut = 1u; batchSize = 16;
		outType = glades::OutputLayerInfo::REGRESSION; outAct = glades::GMath::LINEAR;
	}
	else if (isDow)
	{
		realIn = 4u; realOut = 1u; batchSize = 32;
		outType = glades::OutputLayerInfo::REGRESSION; outAct = glades::GMath::LINEAR;
	}

	unsigned int activeParams = 0u;
	unsigned int storedParams = 0u;
	unsigned int coreStateBytes = 0u;
	compute_param_and_state_sizes(spec, realIn, realOut, activeParams, storedParams, coreStateBytes);

	std::vector<double> trainAccs;
	std::vector<double> testAccs;
	std::vector<double> canonTestAccs;
	std::vector<double> epM1;
	std::vector<double> epM2;
	std::vector<double> trainLosses;
	std::vector<double> testLosses;
	std::vector<double> testBriers;
	std::vector<double> f1C0;
	std::vector<double> f1C1;
	std::vector<double> f1C2;
	std::vector<double> macroF1s;
	std::vector<double> trainMs;
	std::vector<double> predictSplitUs;
	std::vector<double> regRmses;
	std::vector<double> regAuxs;
	std::vector<double> regR2s;
	std::vector<double> equivErrs;

	unsigned int convergedRuns = 0u;
	size_t maxHeapDeltaBytes = 0u;
	size_t weightsBinBytes = 0u;
	unsigned int splitTestRows = 0u;
	unsigned int totalRuns = 0u;

	const glades::QuaternionInputPackingMode pMode =
	    (spec.qnnPacking == glades::QuaternionConfig::ONE_QUAT_PER_FEATURE)
	        ? glades::QUAT_ONE_PER_FEATURE
	        : glades::QUAT_PACK_GROUPS_OF_4;
	const glades::QuaternionOutputReadoutMode rMode =
	    (spec.qnnReadout == glades::QuaternionConfig::READOUT_SPLIT_COMPONENTS)
	        ? glades::QUAT_READOUT_SPLIT_COMPONENTS
	        : glades::QUAT_READOUT_REAL_PART;

	if (isBinaryTruth)
	{
		glades::QuaternionInput di;
		bool loadedOk = false;
		if (isXor)
			loadedOk = di.loadXOR(dataDir + "/xorgate.csv", pMode, rMode);
		else if (isXorParity3)
			loadedOk = di.loadParityFile(repoDataDir + "/xorparity3.csv", pMode, rMode);
		else if (isParity3Odd)
			loadedOk = di.loadBinaryParity(3u, pMode, rMode);
		else if (isParity4)
			loadedOk = di.loadBinaryParity(4u, pMode, rMode);

		if (!loadedOk)
		{
			std::fprintf(stderr, "ERROR: failed to load binary truth dataset for task %s\n", spec.task);
			return 2;
		}
		splitTestRows = di.getTestSize();

		for (unsigned int s = 0u; s < 20u; ++s)
		{
			const uint64_t seed = 100ULL + static_cast<uint64_t>(s);
			const size_t heapBaseline = current_heap_bytes();
			glades::NNInfo* info = make_net_info(
			    spec.id, batchSize, spec.lr, spec.momentum, spec.wd2,
			    spec.hiddenSize, spec.hiddenSize2, glades::GMath::TANH, static_cast<int>(realOut), outType, outAct);
			glades::NNetwork net(info, spec.netType);
			delete info;
			update_heap_delta(heapBaseline, maxHeapDeltaBytes);

			configure_net(net, spec, seed, &quietLogger);
			glades::Terminator term;
			term.setEpoch(static_cast<int>(spec.epochs));
			if (!net.setTerminator(term).ok())
				return 3;

			EpochTracker cb(2u, true, false, heapBaseline, &maxHeapDeltaBytes);
			const double t0 = now_seconds();
			glades::NNetworkStatus st = net.train(&di, &cb);
			const double t1 = now_seconds();
			if (!st.ok())
			{
				std::fprintf(stderr, "ERROR: train failed for %s seed=%llu: %s\n",
				             spec.id, static_cast<unsigned long long>(seed), st.message.c_str());
				return 4;
			}
			trainMs.push_back((t1 - t0) * 1000.0);
			update_heap_delta(heapBaseline, maxHeapDeltaBytes);

			const double p0 = now_seconds();
			for (unsigned int rep = 0u; rep < predictRepeats; ++rep)
			{
				st = net.test(&di);
				if (!st.ok())
					return 5;
			}
			const double p1 = now_seconds();
			predictSplitUs.push_back(((p1 - p0) * 1e6) / static_cast<double>(predictRepeats));
			update_heap_delta(heapBaseline, maxHeapDeltaBytes);

			EvalStats ev;
			if (!eval_split_exact(net, di, false, NULL, ev))
				return 6;

			trainAccs.push_back(ev.accPct);
			testAccs.push_back(ev.accPct);
			canonTestAccs.push_back(ev.accPct);
			trainLosses.push_back(ev.loss);
			testLosses.push_back(ev.loss);
			testBriers.push_back(ev.brier);
			f1C0.push_back(ev.f1[0]);
			f1C1.push_back(ev.f1[1]);
			f1C2.push_back(0.0);
			macroF1s.push_back(ev.macroF1);
			if (ev.correct == ev.total)
				++convergedRuns;
			if (cb.epBce017 > 0u)
				epM1.push_back(static_cast<double>(cb.epBce017));

			if (s == 19u)
				weightsBinBytes = measure_weights_bin_bytes(net, &di);
		}
		totalRuns = static_cast<unsigned int>(testAccs.size());
	}
	else if (isTabularCls)
	{
		const uint64_t splitSeeds[5] = { 42ULL, 1337ULL, 2024ULL, 7777ULL, 9001ULL };
		const uint64_t trainSeeds[5] = { 42ULL, 1337ULL, 2024ULL, 7777ULL, 9001ULL };
		const float testFrac = isIris40 ? 0.6f : 0.2f;
		const std::string path = isWine ? (repoDataDir + "/wine.data") : (dataDir + "/iris.data");
		const char* const irisCanon[3] = { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };
		const char* const wineCanon[3] = { "1", "2", "3" };

		for (unsigned int sp = 0u; sp < 5u; ++sp)
		{
			glades::NumberInput::TrainTestSplitConfig splitCfg;
			splitCfg.testFraction = testFrac;
			splitCfg.shuffle = true;
			splitCfg.stratify = true;
			splitCfg.seed = splitSeeds[sp];

			glades::QuaternionInput di;
			const bool ok = isWine
			                    ? di.loadWine(path, splitCfg, glades::GMath::ZSCORE, pMode, rMode)
			                    : di.loadIris(path, splitCfg, glades::GMath::ZSCORE, pMode, rMode);
			if (!ok)
			{
				std::fprintf(stderr, "ERROR: failed to load %s\n", path.c_str());
				return 2;
			}
			splitTestRows = di.getTestSize();
			unsigned int classMap[3];
			resolve_class_map(di, isWine ? wineCanon : irisCanon, classMap);

			for (unsigned int ts = 0u; ts < 5u; ++ts)
			{
				const size_t heapBaseline = current_heap_bytes();
				glades::NNInfo* info = make_net_info(
				    spec.id, batchSize, spec.lr, spec.momentum, spec.wd2,
				    spec.hiddenSize, spec.hiddenSize2, glades::GMath::TANH, static_cast<int>(realOut), outType, outAct);
				glades::NNetwork net(info, spec.netType);
				delete info;
				update_heap_delta(heapBaseline, maxHeapDeltaBytes);

				configure_net(net, spec, trainSeeds[ts], &quietLogger);
				glades::Terminator term;
				term.setEpoch(static_cast<int>(spec.epochs));
				if (!net.setTerminator(term).ok())
					return 3;

				EpochTracker cb(3u, false, false, heapBaseline, &maxHeapDeltaBytes);
				const double t0 = now_seconds();
				glades::NNetworkStatus st = net.train(&di, &cb);
				const double t1 = now_seconds();
				if (!st.ok())
				{
					std::fprintf(stderr, "ERROR: train failed for %s split=%u seed=%u: %s\n",
					             spec.id, sp, ts, st.message.c_str());
					return 4;
				}
				trainMs.push_back((t1 - t0) * 1000.0);
				update_heap_delta(heapBaseline, maxHeapDeltaBytes);

				const double p0 = now_seconds();
				for (unsigned int rep = 0u; rep < predictRepeats; ++rep)
				{
					st = net.test(&di);
					if (!st.ok())
						return 5;
				}
				const double p1 = now_seconds();
				predictSplitUs.push_back(((p1 - p0) * 1e6) / static_cast<double>(predictRepeats));
				update_heap_delta(heapBaseline, maxHeapDeltaBytes);

				EvalStats trainEv;
				EvalStats testEv;
				if (!eval_split_exact(net, di, true, classMap, trainEv))
					return 6;
				if (!eval_split_exact(net, di, false, classMap, testEv))
					return 6;

				trainAccs.push_back(trainEv.accPct);
				testAccs.push_back(testEv.accPct);
				if (sp == 0u)
					canonTestAccs.push_back(testEv.accPct);
				trainLosses.push_back(trainEv.loss);
				testLosses.push_back(testEv.loss);
				testBriers.push_back(testEv.brier);
				f1C0.push_back(testEv.f1[0]);
				f1C1.push_back(testEv.f1[1]);
				f1C2.push_back(testEv.f1[2]);
				macroF1s.push_back(testEv.macroF1);
				if (testEv.accPct >= 93.0)
					++convergedRuns;

				epM1.push_back(cb.epTrain95 > 0u ? static_cast<double>(cb.epTrain95) : static_cast<double>(spec.epochs));
				epM2.push_back(cb.epTrain98 > 0u ? static_cast<double>(cb.epTrain98) : static_cast<double>(spec.epochs));

				if (sp == 4u && ts == 4u)
					weightsBinBytes = measure_weights_bin_bytes(net, &di);
			}
		}
		totalRuns = static_cast<unsigned int>(testAccs.size());
	}
	else if (isSO3)
	{
		const uint64_t seeds[5] = { 42ULL, 1337ULL, 2024ULL, 7777ULL, 9001ULL };
		const unsigned int nTrain = (spec.taskParam > 0u) ? spec.taskParam : 128u;
		const unsigned int nTest = 256u;
		const bool isRot = (isSO3RotCurve || isSO3RotOod);
		const bool isOod = (isSO3RotOod || isSO3CompOod);

		for (unsigned int s = 0u; s < 5u; ++s)
		{
			glades::QuaternionInput di;
			const bool ok = isRot
			                    ? di.loadSO3Rotation(nTrain, nTest, static_cast<unsigned int>(seeds[s]), isOod, pMode, rMode)
			                    : di.loadSO3Composition(nTrain, nTest, static_cast<unsigned int>(seeds[s]), isOod, pMode, rMode);
			if (!ok)
			{
				std::fprintf(stderr, "ERROR: failed to generate SO(3) dataset for %s\n", spec.task);
				return 2;
			}
			splitTestRows = di.getTestSize();

			const size_t heapBaseline = current_heap_bytes();
			glades::NNInfo* info = make_net_info(
			    spec.id, batchSize, spec.lr, spec.momentum, spec.wd2,
			    spec.hiddenSize, spec.hiddenSize2, glades::GMath::TANH, static_cast<int>(realOut), outType, outAct);
			glades::NNetwork net(info, spec.netType);
			delete info;
			update_heap_delta(heapBaseline, maxHeapDeltaBytes);

			configure_net(net, spec, seeds[s], &quietLogger);
			glades::Terminator term;
			term.setEpoch(static_cast<int>(spec.epochs));
			if (!net.setTerminator(term).ok())
				return 3;

			EpochTracker cb(1u, false, true, heapBaseline, &maxHeapDeltaBytes);
			const double t0 = now_seconds();
			glades::NNetworkStatus st = net.train(&di, &cb);
			const double t1 = now_seconds();
			if (!st.ok())
			{
				std::fprintf(stderr, "ERROR: train failed for %s seed=%u: %s\n", spec.id, s, st.message.c_str());
				return 4;
			}
			trainMs.push_back((t1 - t0) * 1000.0);
			update_heap_delta(heapBaseline, maxHeapDeltaBytes);

			const double p0 = now_seconds();
			for (unsigned int rep = 0u; rep < predictRepeats; ++rep)
			{
				st = net.test(&di);
				if (!st.ok())
					return 5;
			}
			const double p1 = now_seconds();
			predictSplitUs.push_back(((p1 - p0) * 1e6) / static_cast<double>(predictRepeats));
			update_heap_delta(heapBaseline, maxHeapDeltaBytes);

			double trRmse = 0.0, trAng = 0.0, trR2 = 0.0, trMse = 0.0;
			double teRmse = 0.0, teAng = 0.0, teR2 = 0.0, teMse = 0.0;
			if (!eval_so3_split(net, di, true, isRot, trRmse, trAng, trR2, trMse))
				return 6;
			if (!eval_so3_split(net, di, false, isRot, teRmse, teAng, teR2, teMse))
				return 6;
			const double eqErr = measure_so3_equivariance_error(net, 50000ULL + seeds[s]);

			trainLosses.push_back(trMse);
			testLosses.push_back(teMse);
			regRmses.push_back(teRmse);
			regAuxs.push_back(teAng);
			regR2s.push_back(teR2);
			equivErrs.push_back(eqErr);
			trainAccs.push_back(trR2 * 100.0);
			testAccs.push_back(teR2 * 100.0);
			if (teR2 >= 0.95)
				++convergedRuns;

			if (s == 4u)
				weightsBinBytes = measure_weights_bin_bytes(net, &di);
		}
		totalRuns = static_cast<unsigned int>(regRmses.size());
	}
	else if (isTier2Joint4)
	{
		const uint64_t seeds[5] = { 42ULL, 1337ULL, 2024ULL, 7777ULL, 9001ULL };
		const unsigned int nTrain = (spec.taskParam > 0u) ? spec.taskParam : 192u;
		const unsigned int nTest = 256u;
		const bool isPes = (isMolPesId || isMolPesOod);
		const bool isOod = (isMolPesOod || isKeplerOod);

		for (unsigned int s = 0u; s < 5u; ++s)
		{
			glades::QuaternionInput di;
			const bool ok = isPes
			                    ? di.loadMolecularPES(nTrain, nTest, static_cast<unsigned int>(seeds[s]), isOod, pMode, rMode)
			                    : di.loadKeplerOrbit(nTrain, nTest, static_cast<unsigned int>(seeds[s]), isOod, pMode, rMode);
			if (!ok)
			{
				std::fprintf(stderr, "ERROR: failed to generate Tier 2 joint-4D dataset for %s\n", spec.task);
				return 2;
			}
			splitTestRows = di.getTestSize();

			const size_t heapBaseline = current_heap_bytes();
			glades::NNInfo* info = make_net_info(
			    spec.id, batchSize, spec.lr, spec.momentum, spec.wd2,
			    spec.hiddenSize, spec.hiddenSize2, glades::GMath::TANH, static_cast<int>(realOut), outType, outAct);
			glades::NNetwork net(info, spec.netType);
			delete info;
			update_heap_delta(heapBaseline, maxHeapDeltaBytes);

			configure_net(net, spec, seeds[s], &quietLogger);
			glades::Terminator term;
			term.setEpoch(static_cast<int>(spec.epochs));
			if (!net.setTerminator(term).ok())
				return 3;

			EpochTracker cb(1u, false, true, heapBaseline, &maxHeapDeltaBytes);
			const double t0 = now_seconds();
			glades::NNetworkStatus st = net.train(&di, &cb);
			const double t1 = now_seconds();
			if (!st.ok())
			{
				std::fprintf(stderr, "ERROR: train failed for %s seed=%u: %s\n", spec.id, s, st.message.c_str());
				return 4;
			}
			trainMs.push_back((t1 - t0) * 1000.0);
			update_heap_delta(heapBaseline, maxHeapDeltaBytes);

			const double p0 = now_seconds();
			for (unsigned int rep = 0u; rep < predictRepeats; ++rep)
			{
				st = net.test(&di);
				if (!st.ok())
					return 5;
			}
			const double p1 = now_seconds();
			predictSplitUs.push_back(((p1 - p0) * 1e6) / static_cast<double>(predictRepeats));
			update_heap_delta(heapBaseline, maxHeapDeltaBytes);

			double trRmse4 = 0.0, trScalarRmse = 0.0, trAng = 0.0, trR2 = 0.0, trMse4 = 0.0;
			double teRmse4 = 0.0, teScalarRmse = 0.0, teAng = 0.0, teR2 = 0.0, teMse4 = 0.0;
			if (!eval_joint_scalar_vec3_split(net, di, true, trRmse4, trScalarRmse, trAng, trR2, trMse4))
				return 6;
			if (!eval_joint_scalar_vec3_split(net, di, false, teRmse4, teScalarRmse, teAng, teR2, teMse4))
				return 6;
			const double eqErr = measure_so3_symmetry_error(net, 2u, 4u, false, 60000ULL + seeds[s]);

			trainLosses.push_back(trMse4);
			testLosses.push_back(teMse4);
			testBriers.push_back(teScalarRmse); // Scalar (Energy) RMSE in $20
			regRmses.push_back(teRmse4);
			regAuxs.push_back(teAng);           // Vector (Force / LRL) angle error in deg
			regR2s.push_back(teR2);
			equivErrs.push_back(eqErr);
			trainAccs.push_back(trR2 * 100.0);
			testAccs.push_back(teR2 * 100.0);
			if (teR2 >= 0.95)
				++convergedRuns;

			if (s == 4u)
				weightsBinBytes = measure_weights_bin_bytes(net, &di);
		}
		totalRuns = static_cast<unsigned int>(regRmses.size());
	}
	else if (isTier2Slater)
	{
		const uint64_t seeds[5] = { 42ULL, 1337ULL, 2024ULL, 7777ULL, 9001ULL };
		const unsigned int nTrain = (spec.taskParam > 0u) ? spec.taskParam : 192u;
		const unsigned int nTest = 256u;
		const bool isOod = isSlaterOod;

		for (unsigned int s = 0u; s < 5u; ++s)
		{
			glades::QuaternionInput di;
			if (!di.loadSlaterKosterOrbital(nTrain, nTest, static_cast<unsigned int>(seeds[s]), isOod, pMode, rMode))
			{
				std::fprintf(stderr, "ERROR: failed to generate Slater-Koster dataset for %s\n", spec.task);
				return 2;
			}
			splitTestRows = di.getTestSize();

			const size_t heapBaseline = current_heap_bytes();
			glades::NNInfo* info = make_net_info(
			    spec.id, batchSize, spec.lr, spec.momentum, spec.wd2,
			    spec.hiddenSize, spec.hiddenSize2, glades::GMath::TANH, static_cast<int>(realOut), outType, outAct);
			glades::NNetwork net(info, spec.netType);
			delete info;
			update_heap_delta(heapBaseline, maxHeapDeltaBytes);

			configure_net(net, spec, seeds[s], &quietLogger);
			glades::Terminator term;
			term.setEpoch(static_cast<int>(spec.epochs));
			if (!net.setTerminator(term).ok())
				return 3;

			EpochTracker cb(1u, false, true, heapBaseline, &maxHeapDeltaBytes);
			const double t0 = now_seconds();
			glades::NNetworkStatus st = net.train(&di, &cb);
			const double t1 = now_seconds();
			if (!st.ok())
			{
				std::fprintf(stderr, "ERROR: train failed for %s seed=%u: %s\n", spec.id, s, st.message.c_str());
				return 4;
			}
			trainMs.push_back((t1 - t0) * 1000.0);
			update_heap_delta(heapBaseline, maxHeapDeltaBytes);

			const double p0 = now_seconds();
			for (unsigned int rep = 0u; rep < predictRepeats; ++rep)
			{
				st = net.test(&di);
				if (!st.ok())
					return 5;
			}
			const double p1 = now_seconds();
			predictSplitUs.push_back(((p1 - p0) * 1e6) / static_cast<double>(predictRepeats));
			update_heap_delta(heapBaseline, maxHeapDeltaBytes);

			double trRmse = 0.0, trMae = 0.0, trR2 = 0.0, trMse = 0.0;
			double teRmse = 0.0, teMae = 0.0, teR2 = 0.0, teMse = 0.0;
			if (!eval_scalar_reg_split(net, di, true, trRmse, trMae, trR2, trMse))
				return 6;
			if (!eval_scalar_reg_split(net, di, false, teRmse, teMae, teR2, teMse))
				return 6;
			const double invErr = measure_so3_symmetry_error(net, 3u, 1u, true, 61000ULL + seeds[s]);

			trainLosses.push_back(trMse);
			testLosses.push_back(teMse);
			regRmses.push_back(teRmse);
			regAuxs.push_back(teMae);
			regR2s.push_back(teR2);
			equivErrs.push_back(invErr);
			trainAccs.push_back(trR2 * 100.0);
			testAccs.push_back(teR2 * 100.0);
			if (teR2 >= 0.90)
				++convergedRuns;

			if (s == 4u)
				weightsBinBytes = measure_weights_bin_bytes(net, &di);
		}
		totalRuns = static_cast<unsigned int>(regRmses.size());
	}
	else if (isTier2Chiral)
	{
		const uint64_t seeds[5] = { 42ULL, 1337ULL, 2024ULL, 7777ULL, 9001ULL };
		const unsigned int nTrain = (spec.taskParam > 0u) ? spec.taskParam : 180u;
		const unsigned int nTest = 240u;
		const bool isOod = isMolChiralOod;

		for (unsigned int s = 0u; s < 5u; ++s)
		{
			glades::QuaternionInput di;
			if (!di.loadMolecularChirality(nTrain, nTest, static_cast<unsigned int>(seeds[s]), isOod, pMode, rMode))
			{
				std::fprintf(stderr, "ERROR: failed to generate molecular chirality dataset for %s\n", spec.task);
				return 2;
			}
			splitTestRows = di.getTestSize();

			const size_t heapBaseline = current_heap_bytes();
			glades::NNInfo* info = make_net_info(
			    spec.id, batchSize, spec.lr, spec.momentum, spec.wd2,
			    spec.hiddenSize, spec.hiddenSize2, glades::GMath::TANH, static_cast<int>(realOut), outType, outAct);
			glades::NNetwork net(info, spec.netType);
			delete info;
			update_heap_delta(heapBaseline, maxHeapDeltaBytes);

			configure_net(net, spec, seeds[s], &quietLogger);
			glades::Terminator term;
			term.setEpoch(static_cast<int>(spec.epochs));
			if (!net.setTerminator(term).ok())
				return 3;

			EpochTracker cb(3u, false, false, heapBaseline, &maxHeapDeltaBytes);
			const double t0 = now_seconds();
			glades::NNetworkStatus st = net.train(&di, &cb);
			const double t1 = now_seconds();
			if (!st.ok())
			{
				std::fprintf(stderr, "ERROR: train failed for %s seed=%u: %s\n", spec.id, s, st.message.c_str());
				return 4;
			}
			trainMs.push_back((t1 - t0) * 1000.0);
			update_heap_delta(heapBaseline, maxHeapDeltaBytes);

			const double p0 = now_seconds();
			for (unsigned int rep = 0u; rep < predictRepeats; ++rep)
			{
				st = net.test(&di);
				if (!st.ok())
					return 5;
			}
			const double p1 = now_seconds();
			predictSplitUs.push_back(((p1 - p0) * 1e6) / static_cast<double>(predictRepeats));
			update_heap_delta(heapBaseline, maxHeapDeltaBytes);

			EvalStats trainEv;
			EvalStats testEv;
			if (!eval_split_exact(net, di, true, NULL, trainEv))
				return 6;
			if (!eval_split_exact(net, di, false, NULL, testEv))
				return 6;
			const double invErr = measure_so3_symmetry_error(net, 3u, 3u, true, 62000ULL + seeds[s]);

			trainAccs.push_back(trainEv.accPct);
			testAccs.push_back(testEv.accPct);
			if (s == 0u)
				canonTestAccs.push_back(testEv.accPct);
			trainLosses.push_back(trainEv.loss);
			testLosses.push_back(testEv.loss);
			testBriers.push_back(testEv.brier);
			f1C0.push_back(testEv.f1[0]);
			f1C1.push_back(testEv.f1[1]);
			f1C2.push_back(testEv.f1[2]);
			macroF1s.push_back(testEv.macroF1);
			equivErrs.push_back(invErr);
			if (testEv.accPct >= 95.0)
				++convergedRuns;

			epM1.push_back(cb.epTrain95 > 0u ? static_cast<double>(cb.epTrain95) : static_cast<double>(spec.epochs));
			epM2.push_back(cb.epTrain98 > 0u ? static_cast<double>(cb.epTrain98) : static_cast<double>(spec.epochs));

			if (s == 4u)
				weightsBinBytes = measure_weights_bin_bytes(net, &di);
		}
		totalRuns = static_cast<unsigned int>(testAccs.size());
	}
	else if (isDow)
	{
		const uint64_t seeds[5] = { 42ULL, 1337ULL, 2024ULL, 7777ULL, 9001ULL };
		float tMean = 0.0f, tStd = 1.0f;
		glades::QuaternionInput di;
		if (!di.loadDowJonesOHLC(repoDataDir + "/dow_jones_index_train.csv",
		                         repoDataDir + "/dow_jones_index_test.csv",
		                         &tMean, &tStd, pMode, rMode))
		{
			std::fprintf(stderr, "ERROR: failed to load Dow Jones OHLC from %s\n", repoDataDir.c_str());
			return 2;
		}
		splitTestRows = di.getTestSize();

		for (unsigned int s = 0u; s < 5u; ++s)
		{
			const size_t heapBaseline = current_heap_bytes();
			glades::NNInfo* info = make_net_info(
			    spec.id, batchSize, spec.lr, spec.momentum, spec.wd2,
			    spec.hiddenSize, spec.hiddenSize2, glades::GMath::TANH, static_cast<int>(realOut), outType, outAct);
			glades::NNetwork net(info, spec.netType);
			delete info;
			update_heap_delta(heapBaseline, maxHeapDeltaBytes);

			configure_net(net, spec, seeds[s], &quietLogger);
			glades::Terminator term;
			term.setEpoch(static_cast<int>(spec.epochs));
			if (!net.setTerminator(term).ok())
				return 3;

			EpochTracker cb(1u, false, true, heapBaseline, &maxHeapDeltaBytes);
			const double t0 = now_seconds();
			glades::NNetworkStatus st = net.train(&di, &cb);
			const double t1 = now_seconds();
			if (!st.ok())
			{
				std::fprintf(stderr, "ERROR: train failed for %s seed=%u: %s\n", spec.id, s, st.message.c_str());
				return 4;
			}
			trainMs.push_back((t1 - t0) * 1000.0);
			update_heap_delta(heapBaseline, maxHeapDeltaBytes);

			const double p0 = now_seconds();
			for (unsigned int rep = 0u; rep < predictRepeats; ++rep)
			{
				st = net.test(&di);
				if (!st.ok())
					return 5;
			}
			const double p1 = now_seconds();
			predictSplitUs.push_back(((p1 - p0) * 1e6) / static_cast<double>(predictRepeats));
			update_heap_delta(heapBaseline, maxHeapDeltaBytes);

			double trRmseS = 0.0, trRmseU = 0.0, trMae = 0.0, trR2 = 0.0, trDir = 0.0;
			double teRmseS = 0.0, teRmseU = 0.0, teMae = 0.0, teR2 = 0.0, teDir = 0.0;
			if (!eval_dow_split(net, di, true, static_cast<double>(tStd), trRmseS, trRmseU, trMae, trR2, trDir))
				return 6;
			if (!eval_dow_split(net, di, false, static_cast<double>(tStd), teRmseS, teRmseU, teMae, teR2, teDir))
				return 6;

			trainAccs.push_back(trDir);
			testAccs.push_back(teDir);
			trainLosses.push_back(trRmseS * trRmseS);
			testLosses.push_back(teRmseS * teRmseS);
			testBriers.push_back(teMae); // MAE in sigma units for dow_ohlc
			regRmses.push_back(teRmseS);
			regAuxs.push_back(teRmseU);  // USD RMSE
			regR2s.push_back(teR2);
			equivErrs.push_back(0.0);
			if (teR2 >= 0.95)
				++convergedRuns;

			if (s == 4u)
				weightsBinBytes = measure_weights_bin_bytes(net, &di);
		}
		totalRuns = static_cast<unsigned int>(regRmses.size());
	}

	const double meanPredSplitUs = vec_mean(predictSplitUs);
	const double meanPredSampleUs = (splitTestRows > 0u) ? (meanPredSplitUs / static_cast<double>(splitTestRows)) : 0.0;
	const double peakHeapKib = static_cast<double>(maxHeapDeltaBytes) / 1024.0;
	const long rssKib = peak_rss_kib();

	if (jsonOut)
	{
		std::printf(
		    "{\"task\": \"%s\", \"model_id\": \"%s\", \"label\": \"%s\", "
		    "\"active_params\": %u, \"stored_params\": %u, \"budget_epochs\": %u, "
		    "\"runs\": %u, \"converged_runs\": %u, "
		    "\"train_acc_mean\": %.10g, \"train_acc_std\": %.10g, "
		    "\"test_acc_mean\": %.10g, \"test_acc_std\": %.10g, \"canon_test_acc\": %.10g, "
		    "\"ep_metric1_mean\": %.10g, \"ep_metric1_med\": %.10g, \"ep_metric2_mean\": %.10g, "
		    "\"train_loss_mean\": %.10g, \"test_loss_mean\": %.10g, \"test_loss_std\": %.10g, "
		    "\"test_brier_mean\": %.10g, "
		    "\"f1_c0\": %.10g, \"f1_c1\": %.10g, \"f1_c2\": %.10g, "
		    "\"macro_f1_mean\": %.10g, \"macro_f1_std\": %.10g, "
		    "\"train_ms_mean\": %.10g, \"train_ms_std\": %.10g, "
		    "\"predict_split_us\": %.10g, \"predict_sample_us\": %.10g, "
		    "\"peak_heap_delta_kib\": %.10g, \"core_state_bytes\": %u, "
		    "\"weights_bin_bytes\": %lu, \"peak_rss_kib\": %ld, "
		    "\"reg_rmse_mean\": %.10g, \"reg_rmse_std\": %.10g, "
		    "\"reg_aux_mean\": %.10g, \"reg_aux_std\": %.10g, "
		    "\"reg_r2_mean\": %.10g, \"equiv_err_mean\": %.10g}\n",
		    spec.task,
		    spec.id,
		    spec.label,
		    activeParams,
		    storedParams,
		    spec.epochs,
		    totalRuns,
		    convergedRuns,
		    vec_mean(trainAccs),
		    vec_std(trainAccs),
		    vec_mean(testAccs),
		    vec_std(testAccs),
		    vec_mean(canonTestAccs),
		    vec_mean(epM1),
		    vec_median(epM1),
		    vec_mean(epM2),
		    vec_mean(trainLosses),
		    vec_mean(testLosses),
		    vec_std(testLosses),
		    vec_mean(testBriers),
		    vec_mean(f1C0),
		    vec_mean(f1C1),
		    vec_mean(f1C2),
		    vec_mean(macroF1s),
		    vec_std(macroF1s),
		    vec_mean(trainMs),
		    vec_std(trainMs),
		    meanPredSplitUs,
		    meanPredSampleUs,
		    peakHeapKib,
		    coreStateBytes,
		    static_cast<unsigned long>(weightsBinBytes),
		    rssKib,
		    vec_mean(regRmses),
		    vec_std(regRmses),
		    vec_mean(regAuxs),
		    vec_std(regAuxs),
		    vec_mean(regR2s),
		    vec_mean(equivErrs));
		return 0;
	}

	std::printf(
	    "%s\t%s\t%s\t%u\t%u\t%u\t"
	    "%u\t%u\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t"
	    "%.2f\t%.2f\t%.2f\t"
	    "%.6f\t%.6f\t%.6f\t%.6f\t"
	    "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t"
	    "%.4f\t%.4f\t%.4f\t%.4f\t"
	    "%.2f\t%u\t%lu\t%ld\t"
	    "%.6f\t%.6f\t%.4f\t%.4f\t%.6f\t%.6e\n",
	    spec.task,
	    spec.id,
	    spec.label,
	    activeParams,
	    storedParams,
	    spec.epochs,
	    totalRuns,
	    convergedRuns,
	    vec_mean(trainAccs),
	    vec_std(trainAccs),
	    vec_mean(testAccs),
	    vec_std(testAccs),
	    vec_mean(canonTestAccs),
	    vec_mean(epM1),
	    vec_median(epM1),
	    vec_mean(epM2),
	    vec_mean(trainLosses),
	    vec_mean(testLosses),
	    vec_std(testLosses),
	    vec_mean(testBriers),
	    vec_mean(f1C0),
	    vec_mean(f1C1),
	    vec_mean(f1C2),
	    vec_mean(macroF1s),
	    vec_std(macroF1s),
	    vec_mean(trainMs),
	    vec_std(trainMs),
	    meanPredSplitUs,
	    meanPredSampleUs,
	    peakHeapKib,
	    coreStateBytes,
	    static_cast<unsigned long>(weightsBinBytes),
	    rssKib,
	    vec_mean(regRmses),
	    vec_std(regRmses),
	    vec_mean(regAuxs),
	    vec_std(regAuxs),
	    vec_mean(regR2s),
	    vec_mean(equivErrs));

	return 0;
}

} // namespace

int main(int argc, char** argv)
{
	glades::init();

	std::string task;
	std::string modelId;
	std::string dataDir = "unit-tests/datasets";
	std::string repoDataDir = "datasets";
	unsigned int predictRepeats = 0u;
	bool doList = false;
	bool doHeader = false;
	bool jsonOut = false;

	for (int i = 1; i < argc; ++i)
	{
		if (std::strcmp(argv[i], "--list") == 0)
		{
			doList = true;
		}
		else if (std::strcmp(argv[i], "--tsv-header") == 0)
		{
			doHeader = true;
		}
		else if (std::strcmp(argv[i], "--json") == 0)
		{
			jsonOut = true;
		}
		else if (std::strcmp(argv[i], "--task") == 0 && i + 1 < argc)
		{
			task = argv[++i];
		}
		else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc)
		{
			modelId = argv[++i];
		}
		else if (std::strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc)
		{
			dataDir = argv[++i];
		}
		else if (std::strcmp(argv[i], "--repo-data-dir") == 0 && i + 1 < argc)
		{
			repoDataDir = argv[++i];
		}
		else if (std::strcmp(argv[i], "--predict-repeats") == 0 && i + 1 < argc)
		{
			predictRepeats = static_cast<unsigned int>(std::atoi(argv[++i]));
		}
		else
		{
			std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
			return 1;
		}
	}

	if (doList)
	{
		print_list();
		return 0;
	}
	if (doHeader)
	{
		print_tsv_header();
		return 0;
	}

	if (task.empty() || modelId.empty())
	{
		std::fprintf(stderr, "Usage: %s --task <task> --model <id> [--data-dir PATH] [--repo-data-dir PATH] [--predict-repeats N] [--json]\n", argv[0]);
		return 1;
	}

	for (size_t i = 0u; i < kNumSpecs; ++i)
	{
		if (task == kSpecs[i].task && modelId == kSpecs[i].id)
		{
			if (predictRepeats == 0u)
			{
				if (task == "xor" || task == "xorparity3" || task == "parity3_odd" || task == "parity4")
					predictRepeats = 100u;
				else if (task == "so3_rot_curve" || task == "so3_rot_ood" || task == "so3_comp_id" || task == "so3_comp_ood" || task == "dow_ohlc" ||
				         task == "mol_pes_id" || task == "mol_pes_ood" || task == "orbital_slater_id" || task == "orbital_slater_ood" ||
				         task == "orbital_kepler_id" || task == "orbital_kepler_ood")
					predictRepeats = 20u;
				else
					predictRepeats = 50u;
			}
			return run_worker(kSpecs[i], dataDir, repoDataDir, predictRepeats, jsonOut);
		}
	}

	std::fprintf(stderr, "ERROR: unknown (task, model) pair: (%s, %s)\n", task.c_str(), modelId.c_str());
	return 1;
}
