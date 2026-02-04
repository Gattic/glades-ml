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
#include "nn-save-load-test.h"
#include "../../unit-test.h"
#include "Backend/Database/GList.h"
#include "../../../Backend/Machine Learning/main.h"
#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/DataObjects/ImageInput.h"
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"
#include "../../../Backend/Machine Learning/State/Terminator.h"

#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"
#include "../../../Backend/Machine Learning/State/layer.h"
#include "../../../Backend/Machine Learning/State/node.h"

#include <cmath>
#include <fstream>
#include <string>

void NNSaveLoadUnitTest()
{
    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("NN Save/Load Test (Unified model package)\n");
    printf("-----------------------------------\n");

    // Unit tests should be deterministic, but we still want a visible "the model ran"
    // signal in the output. The default callbacks also print a full graph dump on run end,
    // which is noisy for unit tests, so we use a minimal test console callback instead.
    class TestConsoleCallbacks : public glades::ITrainingCallbacks
    {
    public:
        virtual void onRunStart(const glades::NNetwork& net, int runType)
        {
            const glades::NNInfo* sk = net.getNNInfo();
            // IMPORTANT: NNInfo::getName() returns a GString by value. Calling .c_str() on that
            // temporary would yield a dangling pointer. Copy into a stable local first.
            const std::string name = (sk ? std::string(sk->getName().c_str()) : std::string("(unnamed)"));
            if (runType == glades::NNetwork::RUN_TRAIN)
                printf("[UT-NN] %s Training...\n", name.c_str());
            else if (runType == glades::NNetwork::RUN_TEST)
                printf("[UT-NN] %s Testing...\n", name.c_str());
        }

        virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
        {
            if (m.runType != glades::NNetwork::RUN_TRAIN)
                return false;

            if (m.outputType == glades::GMath::REGRESSION)
            {
                printf("[UT-NN] epoch=%d R2=%f%% MSE=%f MAE=%f RMSE=%f lr=%g(mult=%g) gradNorm=%g(scale=%g)\n",
                       m.epoch,
                       m.totalAccuracy,
                       m.totalError,
                       m.regMAE,
                       m.regRMSE,
                       m.learningRate,
                       m.lrMultiplier,
                       m.gradNorm,
                       m.gradNormScale);
            }
            else
            {
                printf("[UT-NN] epoch=%d acc=%f%% MCC=%f%% prec=%f%% recall=%f%% spec=%f%% f1=%f%%\n",
                       m.epoch,
                       m.classAccuracy,
                       m.classMCC,
                       m.classPrecision,
                       m.classRecall,
                       m.classSpecificity,
                       m.classF1);
            }

            return false;
        }

        virtual void onRunEnd(const glades::NNetwork&, int) {}
    };
    TestConsoleCallbacks testCb;

    // This test validates the production-facing persistence API:
    //   NNetwork::saveModel() / NNetwork::loadModel()
    //
    // It verifies:
    // - Package files are created (manifest + nninfo + weights)
    // - Weights round-trip correctly (tensor-first)
    // - netType, epochs, RNG seed, and TrainingConfig are restored from manifest

    // ============================
    // Case A: DFF round-trip
    // ============================
    {
        glades::NumberInput* di = new glades::NumberInput();
        di->trainMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di->trainExpectedMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di->trainMatrix[0][0] = 1.0f;
        di->trainExpectedMatrix[0][0] = 0.0f;
        di->trainMatrix[1][0] = 2.0f;
        di->trainExpectedMatrix[1][0] = 0.0f;

        // Mirror train->test so the test split is well-formed.
        di->testMatrix = di->trainMatrix;
        di->testExpectedMatrix = di->trainExpectedMatrix;

        // Build a minimal 1->1 regression NNInfo with deterministic graph weights.
        // Use a small non-zero LR so the run performs a real parameter update (visible in output),
        // while still being fully deterministic for this tiny dataset.
        glades::InputLayerInfo* in = new glades::InputLayerInfo(
            /*batchSize*/ 2,
            /*learningRate*/ 0.1f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden;
        glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info = new glades::NNInfo("ut_save_load_dff", in, hidden, out);

        glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
        net.setSeed(12345u);
        net.getTerminatorMutable().setEpoch(1);
        net.getTerminatorMutable().setAccuracy(0);

        // Set non-default TrainingConfig knobs to validate manifest persistence.
        glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
        cfg.minibatchSizeOverride = 7;
        cfg.tbpttWindowOverride = 13;
        cfg.globalGradClipNorm = 5.5f;
        cfg.perElementGradClip = 9.0f;
        cfg.lrSchedule.setStep(/*stepSize*/ 3, /*gamma*/ 0.25f);

        net.graphMutable().build(info, di, glades::NNetwork::TYPE_DFF);
        net.setMustdBuildMeat(false);

        glades::Layer* outLayer = net.graphMutable().getOutputLayer(1);
        glades::Node* outNode = net.graphMutable().getOutputNode(outLayer, 0);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::DFF_NodeMissing() Failed==============",
                 outLayer != NULL && outNode != NULL);
        if (outNode)
        {
            // Deterministic weights and bias.
            outNode->setEdgeWeight(0, 2.0f); // w
            outNode->setEdgeWeight(1, -1.0f); // bias edge
        }

        // Run a tiny train pass (1 epoch) so weights change deterministically.
        const glades::NNetworkStatus stTrain = net.train(di, &testCb);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::DFF_TrainStatus() Failed==============",
                 stTrain.ok());
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::DFF_EpochIncrement() Failed==============",
                 net.getEpochs() == 1);

        // Verify training actually performed a parameter update (sanity: weights moved).
        net.materializeGraphParameters();
        if (outNode)
        {
            const float w = outNode->getEdgeWeight(0);
            const float b = outNode->getEdgeWeight(1);
            const float tol = 1e-8f;
            G_assert(__FILE__, __LINE__,
                     "==============NNSaveLoad::DFF_WeightsDidNotUpdate() Failed==============",
                     (std::fabs(w - 2.0f) > tol) || (std::fabs(b - (-1.0f)) > tol));
        }

        const std::string modelName = "ut_model_pkg_dff_v2";
        const glades::NNetworkStatus stSave = net.saveModel(modelName);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::DFF_SaveModel() Failed==============",
                 stSave.ok());

        // Basic package existence check (manifest + nninfo + weights).
        {
            std::ifstream mf(("database/models/" + modelName + "/manifest.txt").c_str());
            std::ifstream ni(("database/models/" + modelName + "/nninfo.csv").c_str());
            std::ifstream wt(("database/models/" + modelName + "/weights.txt").c_str());
            G_assert(__FILE__, __LINE__,
                     "==============NNSaveLoad::DFF_PackageFilesMissing() Failed==============",
                     static_cast<bool>(mf) && static_cast<bool>(ni) && static_cast<bool>(wt));
        }

        // Verify manifest content includes our core metadata (magic/version/netType).
        {
            std::ifstream mf(("database/models/" + modelName + "/manifest.txt").c_str());
            std::string line;
            bool sawMagic = false;
            bool sawVersion = false;
            bool sawNetType = false;
            while (std::getline(mf, line))
            {
                if (line == "GLADES_MODEL")
                    sawMagic = true;
                if (line.find("version=") == 0)
                    sawVersion = true;
                if (line == "netType=0") // DFF
                    sawNetType = true;
            }
            G_assert(__FILE__, __LINE__,
                     "==============NNSaveLoad::DFF_ManifestMissingFields() Failed==============",
                     sawMagic && sawVersion && sawNetType);
        }

        // Load into a fresh network and verify round-trip.
        glades::NNetwork net2(glades::NNetwork::TYPE_DFF);
        const glades::NNetworkStatus stLoad = net2.loadModel(modelName, di);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::DFF_LoadModel() Failed==============",
                 stLoad.ok());

        // Verify metadata restored.
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::DFF_SeedRestored() Failed==============",
                 net2.getSeed() == 12345u);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::DFF_EpochsRestored() Failed==============",
                 net2.getEpochs() == 1);

        const glades::TrainingConfig& cfg2 = net2.getTrainingConfig();
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::DFF_TrainingConfigRestored() Failed==============",
                 cfg2.minibatchSizeOverride == cfg.minibatchSizeOverride &&
                 cfg2.tbpttWindowOverride == cfg.tbpttWindowOverride &&
                 cfg2.globalGradClipNorm == cfg.globalGradClipNorm &&
                 cfg2.perElementGradClip == cfg.perElementGradClip &&
                 cfg2.lrSchedule.type == cfg.lrSchedule.type &&
                 cfg2.lrSchedule.stepSizeEpochs == cfg.lrSchedule.stepSizeEpochs &&
                 cfg2.lrSchedule.gamma == cfg.lrSchedule.gamma);

        // Verify weights restored by materializing the graph snapshot.
        net.materializeGraphParameters();
        net2.materializeGraphParameters();
        glades::Layer* outLayer2 = net2.graphMutable().getOutputLayer(1);
        glades::Node* outNode2 = net2.graphMutable().getOutputNode(outLayer2, 0);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::DFF_NodeMissingAfterLoad() Failed==============",
                 outLayer2 != NULL && outNode2 != NULL);
        if (outNode && outNode2)
        {
            const float tol = 1e-5f;
            const float w1 = outNode->getEdgeWeight(0);
            const float b1 = outNode->getEdgeWeight(1);
            const float w2 = outNode2->getEdgeWeight(0);
            const float b2 = outNode2->getEdgeWeight(1);
            G_assert(__FILE__, __LINE__,
                     "==============NNSaveLoad::DFF_WeightsRoundTrip() Failed==============",
                     std::fabs(w1 - w2) < tol && std::fabs(b1 - b2) < tol);
        }

        delete di;
        delete info; // owns in/out
    }

    // ============================
    // Case B: GRU round-trip (weights only; no training)
    // ============================
    {
        printf("[UT-NN] GRU save/load round-trip (no training)\n");
        glades::NumberInput* di = new glades::NumberInput();
        di->trainMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
        di->trainExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
        di->testMatrix = di->trainMatrix;
        di->testExpectedMatrix = di->trainExpectedMatrix;

        glades::InputLayerInfo* in = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden;
        hidden.push_back(new glades::HiddenLayerInfo(
            /*size*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        ));
        glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info = new glades::NNInfo("ut_save_load_gru", in, hidden, out);

        glades::NNetwork net(info, glades::NNetwork::TYPE_GRU);
        net.setSeed(777u);
        net.getTerminatorMutable().setEpoch(1);
        net.getTerminatorMutable().setAccuracy(0);

        net.graphMutable().build(info, di, glades::NNetwork::TYPE_GRU);
        net.setMustdBuildMeat(false);

        glades::Layer* hiddenLayer = net.graphMutable().getOutputLayer(1);
        glades::Layer* outLayer = net.graphMutable().getOutputLayer(2);
        glades::Node* hiddenNode = net.graphMutable().getOutputNode(hiddenLayer, 0);
        glades::Node* outNode = net.graphMutable().getOutputNode(outLayer, 0);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::GRU_NodesMissing() Failed==============",
                 hiddenLayer != NULL && outLayer != NULL && hiddenNode != NULL && outNode != NULL);

        if (hiddenNode && outNode)
        {
            // GRU layout for 1 input, 1 hidden:
            // Node edges: 3*(fanIn+1)=6: z(W,b), r(W,b), h(W,b)
            hiddenNode->setEdgeWeight(0, 0.11f);
            hiddenNode->setEdgeWeight(1, -0.12f);
            hiddenNode->setEdgeWeight(2, 0.21f);
            hiddenNode->setEdgeWeight(3, -0.22f);
            hiddenNode->setEdgeWeight(4, 0.31f);
            hiddenNode->setEdgeWeight(5, -0.32f);

            glades::Node* ctx = hiddenNode->getContextNode();
            G_assert(__FILE__, __LINE__,
                     "==============NNSaveLoad::GRU_ContextMissing() Failed==============",
                     ctx != NULL);
            if (ctx)
            {
                // Context edges: 3*hidden=3: Uz, Ur, Uh
                ctx->setEdgeWeight(0, 0.41f);
                ctx->setEdgeWeight(1, 0.42f);
                ctx->setEdgeWeight(2, 0.43f);
            }

            // Output dense: (fanIn+1)=2: Wy, b
            outNode->setEdgeWeight(0, -0.51f);
            outNode->setEdgeWeight(1, 0.52f);
        }

        const std::string modelName = "ut_model_pkg_gru_v2";
        const glades::NNetworkStatus stSave = net.saveModel(modelName);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::GRU_SaveModel() Failed==============",
                 stSave.ok());

        glades::NNetwork net2(glades::NNetwork::TYPE_GRU);
        const glades::NNetworkStatus stLoad = net2.loadModel(modelName, di);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::GRU_LoadModel() Failed==============",
                 stLoad.ok());
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::GRU_SeedRestored() Failed==============",
                 net2.getSeed() == 777u);

        net.materializeGraphParameters();
        net2.materializeGraphParameters();

        glades::Layer* hiddenLayer2 = net2.graphMutable().getOutputLayer(1);
        glades::Layer* outLayer2 = net2.graphMutable().getOutputLayer(2);
        glades::Node* hiddenNode2 = net2.graphMutable().getOutputNode(hiddenLayer2, 0);
        glades::Node* outNode2 = net2.graphMutable().getOutputNode(outLayer2, 0);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::GRU_NodesMissingAfterLoad() Failed==============",
                 hiddenLayer2 != NULL && outLayer2 != NULL && hiddenNode2 != NULL && outNode2 != NULL);

        if (hiddenNode && hiddenNode2 && outNode && outNode2)
        {
            const float tol = 1e-6f;
            // Node edges
            for (unsigned int i = 0; i < 6; ++i)
            {
                G_assert(__FILE__, __LINE__,
                         "==============NNSaveLoad::GRU_NodeWeightsRoundTrip() Failed==============",
                         std::fabs(hiddenNode->getEdgeWeight(i) - hiddenNode2->getEdgeWeight(i)) < tol);
            }
            // Context edges
            glades::Node* ctx1 = hiddenNode->getContextNode();
            glades::Node* ctx2 = hiddenNode2->getContextNode();
            G_assert(__FILE__, __LINE__,
                     "==============NNSaveLoad::GRU_ContextMissingAfterLoad() Failed==============",
                     ctx1 != NULL && ctx2 != NULL);
            if (ctx1 && ctx2)
            {
                for (unsigned int i = 0; i < 3; ++i)
                {
                    G_assert(__FILE__, __LINE__,
                             "==============NNSaveLoad::GRU_ContextWeightsRoundTrip() Failed==============",
                             std::fabs(ctx1->getEdgeWeight(i) - ctx2->getEdgeWeight(i)) < tol);
                }
            }
            // Output edges
            for (unsigned int i = 0; i < 2; ++i)
            {
                G_assert(__FILE__, __LINE__,
                         "==============NNSaveLoad::GRU_OutputWeightsRoundTrip() Failed==============",
                         std::fabs(outNode->getEdgeWeight(i) - outNode2->getEdgeWeight(i)) < tol);
            }
        }

        delete di;
        delete info; // owns in/hidden/out
    }

    printf("\n============================================================\n");
}
