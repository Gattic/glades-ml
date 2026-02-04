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
#include "nn-test.h"
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
#include "../../../Backend/Machine Learning/State/node.h"
#include <cmath>

// === This is the primary unit testing function:
// void G_assert(const char* fileName, int lineNo, const char* failureMsg, bool expr)

void NNUnitTest()
{
    printf("============================================================\n");
    printf("NN Test 1\n");
    printf("-----------------------------------\n");
 
    shmea::GString netName = "xornet";
    shmea::GString inputFName = "xorgate.csv";
    int inputType = glades::DataInput::CSV;
    //int inputType = glades::DataInput::IMAGE;
    //int inputType = glades::DataInput::TEXT;
    
    // Modify the paths to properly load the data later
    glades::DataInput* di = NULL;
    if (inputType == glades::DataInput::CSV)
    {
    	inputFName = "datasets/" + inputFName;
    	di = new glades::NumberInput();
    }
    else if (inputType == glades::DataInput::IMAGE)
    {
    	// inputFName = "datasets/images/" + inputFName + "/";
    	di = new glades::ImageInput();
    }
    else if (inputType == glades::DataInput::TEXT)
    {
    	// TODO
    	return;
    }
    else
    	return;
    
    if (!di)
    	return;
    
    // Load the input data
    di->import(inputFName);
    
    // Deterministic, hardcoded config (replaces on-disk config load).
    // Historical config source: unit-tests/database/neuralnetworks/xornet
    glades::InputLayerInfo* in1 = new glades::InputLayerInfo(
        /*batchSize*/ 1,
        /*learningRate*/ 0.003f,
        /*momentumFactor*/ 0.0f,
        /*weightDecay1*/ 0.0f,
        /*weightDecay2*/ 0.0f,
        /*pDropout*/ 0.0f,
        /*activationType*/ glades::GMath::TANH,
        /*activationParam*/ 0.0f
    );
    std::vector<glades::HiddenLayerInfo*> hidden1;
    hidden1.push_back(new glades::HiddenLayerInfo(
        /*size*/ 2,
        /*learningRate*/ 0.003f,
        /*momentumFactor*/ 0.0f,
        /*weightDecay1*/ 0.0f,
        /*weightDecay2*/ 0.0f,
        /*pDropout*/ 0.0f,
        /*activationType*/ glades::GMath::TANH,
        /*activationParam*/ 0.0f
    ));
    glades::OutputLayerInfo* out1 = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
    glades::NNInfo* info1 = new glades::NNInfo(netName, in1, hidden1, out1);
    glades::NNetwork cNetwork(info1);
    delete info1;
    cNetwork.setSeed(0xC0FFEE01ULL);
    
    // Termination Conditions
    //cNetwork.setTimestamp(maxTimeStamp);
    cNetwork.getTerminatorMutable().setEpoch(100000);
    cNetwork.getTerminatorMutable().setAccuracy(95);
    
    // Run the training and retrieve a metanetwork
    glades::MetaNetwork* newTrainNet =
    	glades::train(&cNetwork, di);
    G_assert (__FILE__, __LINE__, "==============NN1-test::TrainStatus() Failed==============", newTrainNet != NULL);

    G_assert (__FILE__, __LINE__, "==============NN1-test::Accuracy() Failed==============", cNetwork.getAccuracy() >= 95.0f);
    delete newTrainNet;
    delete di;

    printf("-----------------------------------\n");
    printf("NN Test 2\n");
    printf("-----------------------------------\n");
 
    netName = "iris";
    inputFName = "iris.data";
    inputType = glades::DataInput::CSV;
    //int inputType = glades::DataInput::IMAGE;
    //int inputType = glades::DataInput::TEXT;
    
    // Modify the paths to properly load the data later
    glades::DataInput* di2 = NULL;
    if (inputType == glades::DataInput::CSV)
    {
    	inputFName = "datasets/" + inputFName;
    	di2 = new glades::NumberInput();
    }
    else if (inputType == glades::DataInput::IMAGE)
    {
    	// inputFName = "datasets/images/" + inputFName + "/";
    	di2 = new glades::ImageInput();
    }
    else if (inputType == glades::DataInput::TEXT)
    {
    	// TODO
    	return;
    }
    else
    	return;
    
    if (!di2)
    	return;
    
    // Load the input data
    di2->import(inputFName);
    
    // Deterministic, hardcoded config (replaces on-disk config load).
    // Historical config source: unit-tests/database/neuralnetworks/iris
    glades::InputLayerInfo* in2 = new glades::InputLayerInfo(
        /*batchSize*/ 1,
        /*learningRate*/ 0.01f,
        /*momentumFactor*/ 0.0f,
        /*weightDecay1*/ 0.0f,
        /*weightDecay2*/ 0.0f,
        /*pDropout*/ 0.0f,
        /*activationType*/ glades::GMath::SIGMOID,
        /*activationParam*/ 0.0f
    );
    std::vector<glades::HiddenLayerInfo*> hidden2;
    hidden2.push_back(new glades::HiddenLayerInfo(
        /*size*/ 5,
        /*learningRate*/ 0.01f,
        /*momentumFactor*/ 0.0f,
        /*weightDecay1*/ 0.0f,
        /*weightDecay2*/ 0.0f,
        /*pDropout*/ 0.0f,
        /*activationType*/ glades::GMath::SIGMOID,
        /*activationParam*/ 0.0f
    ));
    glades::OutputLayerInfo* out2 = new glades::OutputLayerInfo(3, glades::OutputLayerInfo::CLASSIFICATION);
    glades::NNInfo* info2 = new glades::NNInfo(netName, in2, hidden2, out2);
    glades::NNetwork cNetwork2(info2);
    delete info2;
    cNetwork2.setSeed(0xC0FFEE02ULL);
    
    // Termination Conditions
    //cNetwork2.setTimestamp(maxTimeStamp);
    cNetwork2.getTerminatorMutable().setEpoch(100000);
    cNetwork2.getTerminatorMutable().setAccuracy(95);
    
    // Run the training and retrieve a metanetwork
    glades::MetaNetwork* newTrainNet2 =
    	glades::train(&cNetwork2, di2);
    G_assert (__FILE__, __LINE__, "==============NN2-test::TrainStatus() Failed==============", newTrainNet2 != NULL);
    
    G_assert (__FILE__, __LINE__, "==============NN2-test::Accuracy() Failed==============", cNetwork2.getAccuracy() >= 95.0f);
    delete newTrainNet2;
    delete di2;


    printf("-----------------------------------\n");
    printf("NN Test 3\n");
    printf("-----------------------------------\n");
 
    netName = "xorgateText";
    inputFName = "xorgateText.csv";
    inputType = glades::DataInput::CSV;
    //int inputType = glades::DataInput::IMAGE;
    //int inputType = glades::DataInput::TEXT;
    
    // Modify the paths to properly load the data later
    glades::DataInput* di3 = NULL;
    if (inputType == glades::DataInput::CSV)
    {
    	inputFName = "datasets/" + inputFName;
    	di3 = new glades::NumberInput();
    }
    else if (inputType == glades::DataInput::IMAGE)
    {
    	// inputFName = "datasets/images/" + inputFName + "/";
    	di3 = new glades::ImageInput();
    }
    else if (inputType == glades::DataInput::TEXT)
    {
    	// TODO
    	return;
    }
    else
    	return;
    
    if (!di3)
    	return;
    
    // Load the input data
    di3->import(inputFName);
    
    // Deterministic, hardcoded config (replaces on-disk config load).
    // Historical config source: unit-tests/database/neuralnetworks/xorgateText
    glades::InputLayerInfo* in3 = new glades::InputLayerInfo(
        /*batchSize*/ 1,
        /*learningRate*/ 0.001f,
        /*momentumFactor*/ 0.0f,
        /*weightDecay1*/ 0.0f,
        /*weightDecay2*/ 0.0f,
        /*pDropout*/ 0.0f,
        /*activationType*/ glades::GMath::TANH,
        /*activationParam*/ 0.0f
    );
    std::vector<glades::HiddenLayerInfo*> hidden3;
    hidden3.push_back(new glades::HiddenLayerInfo(
        /*size*/ 3,
        /*learningRate*/ 0.001f,
        /*momentumFactor*/ 0.0f,
        /*weightDecay1*/ 0.0f,
        /*weightDecay2*/ 0.0f,
        /*pDropout*/ 0.0f,
        /*activationType*/ glades::GMath::TANH,
        /*activationParam*/ 0.0f
    ));
    glades::OutputLayerInfo* out3 = new glades::OutputLayerInfo(2, glades::OutputLayerInfo::REGRESSION);
    glades::NNInfo* info3 = new glades::NNInfo(netName, in3, hidden3, out3);
    glades::NNetwork cNetwork3(info3);
    delete info3;
    cNetwork3.setSeed(0xC0FFEE03ULL);
    
    // Termination Conditions
    //cNetwork3.setTimestamp(maxTimeStamp);
    cNetwork3.getTerminatorMutable().setEpoch(100000);
    cNetwork3.getTerminatorMutable().setAccuracy(95);
    
    // Run the training and retrieve a metanetwork
    glades::MetaNetwork* newTrainNet3 =
    	glades::train(&cNetwork3, di3);
    G_assert (__FILE__, __LINE__, "==============NN3-test::TrainStatus() Failed==============", newTrainNet3 != NULL);

    G_assert (__FILE__, __LINE__, "==============NN3-test::Accuracy() Failed==============", cNetwork3.getAccuracy() >= 95.0f);
    delete newTrainNet3;
    delete di3;

    printf("-----------------------------------\n");
    printf("NN Test 4 (Minibatch application timing)\n");
    printf("-----------------------------------\n");

    // This test specifically guards against the historical minibatch bug where updates were applied
    // on inputRowCounter == 0 (first sample) instead of end-of-batch, and where partial batches
    // (minibatchSize > trainSize) would ignore most samples.
    //
    // We build a tiny 1->1 regression network with linear activation and deterministic weights,
    // run exactly 1 epoch, and verify the weight update reflects *both* samples.
    {
        // Toy dataset (2 samples, 1 feature, 1 target)
        glades::NumberInput* di4 = new glades::NumberInput();
        di4->trainMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di4->trainExpectedMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di4->trainMatrix[0][0] = 1.0f;
        di4->trainExpectedMatrix[0][0] = 0.0f;
        di4->trainMatrix[1][0] = 2.0f;
        di4->trainExpectedMatrix[1][0] = 1.0f;

        // Build a minimal NNInfo:
        // - minibatchSize intentionally larger than trainSize to exercise partial batch behavior
        // - linear activation so gradients are easy to reason about
        glades::InputLayerInfo* in4 = new glades::InputLayerInfo(
            /*batchSize*/ 10,
            /*learningRate*/ 0.1f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden4;
        glades::OutputLayerInfo* out4 = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info4 = new glades::NNInfo("ut_minibatch_timing", in4, hidden4, out4);

        glades::NNetwork cNetwork4(info4);
        cNetwork4.getTerminatorMutable().setEpoch(1);     // exactly 1 epoch
        cNetwork4.getTerminatorMutable().setAccuracy(0);  // don't terminate by accuracy

        // Pre-build "meat" so we can set deterministic weights, then prevent rebuild in train()
        cNetwork4.graphMutable().build(info4, di4, glades::NNetwork::TYPE_DFF);
        cNetwork4.setMustdBuildMeat(false);

        // Set deterministic initial weight: w0 = 1.0 (single edge: input0 -> output0)
        glades::Layer* outLayer4 = cNetwork4.graphMutable().getOutputLayer(1);
        glades::Node* outNode4 = cNetwork4.graphMutable().getOutputNode(outLayer4, 0);
        outNode4->setEdgeWeight(0, 1.0f);

        // Train
        const glades::NNetworkStatus st = cNetwork4.train(di4);
        G_assert(__FILE__, __LINE__,
                 "==============NN4-test::TrainStatus() Failed==============",
                 st.ok());

        // Parameters are tensor-first; the Node/Edge graph is a derived view.
        // Materialize it on demand before inspecting weights through Node APIs.
        cNetwork4.materializeGraphParameters();

        // Expected update (MSE, linear activation, no momentum/decay, batchCount=2):
        // delta(sample) = lr * 2 * (p - y) * x
        // sample1: x=1, y=0, p=1 => delta1 = 0.2
        // sample2: x=2, y=1, p=2 => delta2 = 0.4
        // avg delta = (0.2 + 0.4) / 2 = 0.3 => w1 = 1.0 - 0.3 = 0.7
        const float wFinal = outNode4->getEdgeWeight(0);
        const float expectedW = 0.7f;
        const float tol = 1e-3f;

        printf("[UT] NN4 final weight = %f (expected ~%f)\n", wFinal, expectedW);

        /* For this toy dataset the accuracy can easily show 0% even though the gradient update is correct.
         * The weight assertion is the real signal here
         */
        G_assert(__FILE__, __LINE__,
                 "==============NN4-test::MinibatchUpdate() Failed==============",
                 (wFinal > expectedW - tol) && (wFinal < expectedW + tol));

        delete di4;
        delete info4; // owns in4/out4
    }

    printf("-----------------------------------\n");
    printf("NN Test 4b (Weight decay: L2 decays weights)\n");
    printf("-----------------------------------\n");

    // This test verifies that weightDecay2 (L2) actually decays the *weight* (not the input),
    // and that it works even when the data gradient is exactly zero.
    //
    // We use x=0, y=0 so dLoss/dw == 0, and initialize w0=1.
    // With lr=0.1 and weightDecay2=1.0:
    //   delta = lr * (lambda2 * w) = 0.1 * 1 * 1 = 0.1
    //   w1 = w0 - delta = 0.9
    {
        glades::NumberInput* di4b = new glades::NumberInput();
        di4b->trainMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
        di4b->trainExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
        di4b->trainMatrix[0][0] = 0.0f;
        di4b->trainExpectedMatrix[0][0] = 0.0f;

        const float lr = 0.1f;
        glades::InputLayerInfo* in4b = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ lr,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 1.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden4b;
        glades::OutputLayerInfo* out4b = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info4b = new glades::NNInfo("ut_weight_decay_l2", in4b, hidden4b, out4b);

        glades::NNetwork net4b(info4b);
        net4b.getTerminatorMutable().setEpoch(1);
        net4b.getTerminatorMutable().setAccuracy(0);

        net4b.graphMutable().build(info4b, di4b, glades::NNetwork::TYPE_DFF);
        net4b.setMustdBuildMeat(false);

        glades::Layer* outLayer4b = net4b.graphMutable().getOutputLayer(1);
        glades::Node* outNode4b = net4b.graphMutable().getOutputNode(outLayer4b, 0);
        outNode4b->setEdgeWeight(0, 1.0f);

        net4b.train(di4b);
        G_assert(__FILE__, __LINE__,
                 "==============NN4b-test::TrainStatus() Failed==============",
                 net4b.getLastStatus().ok());

        net4b.materializeGraphParameters();

        const float wFinal = outNode4b->getEdgeWeight(0);
        const float expectedW = 0.9f;
        const float tol = 1e-3f;
        printf("[UT] NN4b L2 final weight = %f (expected ~%f)\n", wFinal, expectedW);
        G_assert(__FILE__, __LINE__,
                 "==============NN4b-test::L2WeightDecay() Failed==============",
                 (wFinal > expectedW - tol) && (wFinal < expectedW + tol));

        delete di4b;
        delete info4b;
    }

    printf("-----------------------------------\n");
    printf("NN Test 4c (Weight decay: L1 decays weights toward 0)\n");
    printf("-----------------------------------\n");

    // This test verifies L1 regularization behavior for both positive and negative weights.
    // With x=0,y=0 => no data gradient, only L1 decay:
    //   w1 = w0 - lr*lambda1*sign(w0)
    // For lr=0.1, lambda1=1:
    //   w0=+1 => w1=0.9
    //   w0=-1 => w1=-0.9
    {
        glades::NumberInput* di4c = new glades::NumberInput();
        di4c->trainMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
        di4c->trainExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
        di4c->trainMatrix[0][0] = 0.0f;
        di4c->trainExpectedMatrix[0][0] = 0.0f;

        const float lr = 0.1f;
        const float lambda1 = 1.0f;
        const float tol = 1e-3f;

        // Case 1: w0 = +1
        {
            glades::InputLayerInfo* in = new glades::InputLayerInfo(
                /*batchSize*/ 1,
                /*learningRate*/ lr,
                /*momentumFactor*/ 0.0f,
                /*weightDecay1*/ lambda1,
                /*weightDecay2*/ 0.0f,
                /*pDropout*/ 0.0f,
                /*activationType*/ glades::GMath::LINEAR,
                /*activationParam*/ 1.0f
            );
            std::vector<glades::HiddenLayerInfo*> hidden;
            glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
            glades::NNInfo* info = new glades::NNInfo("ut_weight_decay_l1_pos", in, hidden, out);

            glades::NNetwork net(info);
            net.getTerminatorMutable().setEpoch(1);
            net.getTerminatorMutable().setAccuracy(0);
            net.graphMutable().build(info, di4c, glades::NNetwork::TYPE_DFF);
            net.setMustdBuildMeat(false);

            glades::Layer* outLayer = net.graphMutable().getOutputLayer(1);
            glades::Node* outNode = net.graphMutable().getOutputNode(outLayer, 0);
            outNode->setEdgeWeight(0, 1.0f);

            net.train(di4c);
            G_assert(__FILE__, __LINE__,
                     "==============NN4c-test::TrainStatus_Pos() Failed==============",
                     net.getLastStatus().ok());

            net.materializeGraphParameters();

            const float wFinal = outNode->getEdgeWeight(0);
            const float expectedW = 0.9f;
            printf("[UT] NN4c L1(+1) final weight = %f (expected ~%f)\n", wFinal, expectedW);
            G_assert(__FILE__, __LINE__,
                     "==============NN4c-test::L1WeightDecay_Pos() Failed==============",
                     (wFinal > expectedW - tol) && (wFinal < expectedW + tol));

            delete info;
        }

        // Case 2: w0 = -1
        {
            glades::InputLayerInfo* in = new glades::InputLayerInfo(
                /*batchSize*/ 1,
                /*learningRate*/ lr,
                /*momentumFactor*/ 0.0f,
                /*weightDecay1*/ lambda1,
                /*weightDecay2*/ 0.0f,
                /*pDropout*/ 0.0f,
                /*activationType*/ glades::GMath::LINEAR,
                /*activationParam*/ 1.0f
            );
            std::vector<glades::HiddenLayerInfo*> hidden;
            glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
            glades::NNInfo* info = new glades::NNInfo("ut_weight_decay_l1_neg", in, hidden, out);

            glades::NNetwork net(info);
            net.getTerminatorMutable().setEpoch(1);
            net.getTerminatorMutable().setAccuracy(0);
            net.graphMutable().build(info, di4c, glades::NNetwork::TYPE_DFF);
            net.setMustdBuildMeat(false);

            glades::Layer* outLayer = net.graphMutable().getOutputLayer(1);
            glades::Node* outNode = net.graphMutable().getOutputNode(outLayer, 0);
            outNode->setEdgeWeight(0, -1.0f);

            net.train(di4c);
            G_assert(__FILE__, __LINE__,
                     "==============NN4c-test::TrainStatus_Neg() Failed==============",
                     net.getLastStatus().ok());

            net.materializeGraphParameters();

            const float wFinal = outNode->getEdgeWeight(0);
            const float expectedW = -0.9f;
            printf("[UT] NN4c L1(-1) final weight = %f (expected ~%f)\n", wFinal, expectedW);
            G_assert(__FILE__, __LINE__,
                     "==============NN4c-test::L1WeightDecay_Neg() Failed==============",
                     (wFinal > expectedW - tol) && (wFinal < expectedW + tol));

            delete info;
        }

        delete di4c;
    }

    printf("-----------------------------------\n");
    printf("NN Test 5 (RNN context nodes)\n");
    printf("-----------------------------------\n");

    // This test validates the RNN "context node" mechanism:
    // - Context state is reset at the start of an epoch/run.
    // - Context is updated from the hidden node's *output activation* (post-squash),
    //   not just the input-edge activation sum.
    //
    // We create a tiny 1->1->1 RNN with linear activations:
    //   h_t = Wx*x_t + Wh*h_{t-1}
    //   y_t = Wy*h_t
    //
    // With Wx=Wh=Wy=1, x1=2, x2=3, and h0=0:
    //   h1=2, h2=5, y2=5
    {
        glades::NumberInput* di0 = new glades::NumberInput();
        di0->trainMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di0->trainExpectedMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di0->trainMatrix[0][0] = 2.0f;
        di0->trainMatrix[1][0] = 3.0f;
        di0->trainExpectedMatrix[0][0] = 0.0f; // not used by this assertion
        di0->trainExpectedMatrix[1][0] = 0.0f; // not used by this assertion

        // Evaluation semantics: test() reads the test split.
        // For deterministic forward-pass unit tests, mirror train->test.
        di0->testMatrix = di0->trainMatrix;
        di0->testExpectedMatrix = di0->trainExpectedMatrix;

        glades::InputLayerInfo* in0 = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden0;
        hidden0.push_back(new glades::HiddenLayerInfo(
            /*size*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        ));
        glades::OutputLayerInfo* out0 = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info0 = new glades::NNInfo("ut_rnn_context", in0, hidden0, out0);

        glades::NNetwork rnnNet0(info0, glades::NNetwork::TYPE_RNN);
        rnnNet0.getTerminatorMutable().setEpoch(1);
        rnnNet0.getTerminatorMutable().setAccuracy(0);

        // Pre-build so we can set deterministic weights, then prevent rebuild in test()
        rnnNet0.graphMutable().build(info0, di0, glades::NNetwork::TYPE_RNN);
        rnnNet0.setMustdBuildMeat(false);

        // Hidden layer (counter=1) and output layer (counter=2)
        glades::Layer* hiddenLayer0 = rnnNet0.graphMutable().getOutputLayer(1);
        glades::Layer* outLayer0 = rnnNet0.graphMutable().getOutputLayer(2);
        glades::Node* hiddenNode0 = rnnNet0.graphMutable().getOutputNode(hiddenLayer0, 0);
        glades::Node* outNode0 = rnnNet0.graphMutable().getOutputNode(outLayer0, 0);

        // Deterministic weights: Wx=1, Wh=1, Wy=1; biases=0
        hiddenLayer0->setBiasWeight(0.0f);
        outLayer0->setBiasWeight(0.0f);
        hiddenNode0->setEdgeWeight(0, 1.0f); // Wx
        outNode0->setEdgeWeight(0, 1.0f);    // Wy

        // Context edge weight Wh and a non-zero starting state to ensure reset works
        glades::Node* ctx0 = hiddenNode0->getContextNode();
        G_assert(__FILE__, __LINE__,
                 "==============NN5-test::ContextNodeMissing() Failed==============",
                 ctx0 != NULL);
        if (ctx0)
        {
            ctx0->setEdgeWeight(0, 1.0f); // Wh
            ctx0->setWeight(10.0f);       // should be reset to 0 at run start
        }

        // Run inference (no weight updates)
        const glades::NNetworkStatus stTest0 = rnnNet0.test(di0);
        G_assert(__FILE__, __LINE__,
                 "==============NN5-test::TestStatus() Failed==============",
                 stTest0.ok());

        const float expected = 5.0f;
        const float tol = 1e-3f;
        const float y2 = outNode0->getWeight();
        const float h2 = hiddenNode0->getWeight();
        const float ctxAfter = (ctx0 ? ctx0->getWeight() : 0.0f);

        printf("[UT] RNN y2=%f h2=%f ctx=%f (expected ~%f)\n", y2, h2, ctxAfter, expected);

        // Core signal: context must reflect h2 and output must reflect the recurrence.
        G_assert(__FILE__, __LINE__,
                 "==============NN5-test::RNNOutputWithContext() Failed==============",
                 (y2 > expected - tol) && (y2 < expected + tol));
        G_assert(__FILE__, __LINE__,
                 "==============NN5-test::RNNHiddenState() Failed==============",
                 (h2 > expected - tol) && (h2 < expected + tol));
        G_assert(__FILE__, __LINE__,
                 "==============NN5-test::RNNContextUpdatedFromHidden() Failed==============",
                 (ctxAfter > expected - tol) && (ctxAfter < expected + tol));

        delete di0;
        delete info0; // owns in0/hidden0/out0
    }

    printf("-----------------------------------\n");
    printf("NN Test 5b (RNN multiple sequences reset at boundaries)\n");
    printf("-----------------------------------\n");

    // This test validates the "proper sequence" abstraction:
    // - DataInput can represent multiple sequences with boundaries.
    // - RNN forward pass must reset hidden context at each sequence boundary.
    //
    // We create a 1->1->1 linear RNN with Wx=Wh=Wy=1, bias=0.
    // Sequence 1: x=[2,3] => y_last = 5
    // Sequence 2: x=[7,11] => y_last = 18  (NOT 23, which would happen if context leaked from seq1)
    {
        glades::NumberInput* di5b = new glades::NumberInput();
        di5b->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));
        di5b->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));
        di5b->trainMatrix[0][0] = 2.0f;
        di5b->trainMatrix[1][0] = 3.0f;
        di5b->trainMatrix[2][0] = 7.0f;
        di5b->trainMatrix[3][0] = 11.0f;

        // Mirror train->test so test() evaluates the same rows.
        di5b->testMatrix = di5b->trainMatrix;
        di5b->testExpectedMatrix = di5b->trainExpectedMatrix;

        std::vector<glades::DataInput::SequenceSpan> spans;
        spans.push_back(glades::DataInput::SequenceSpan(0u, 2u));
        spans.push_back(glades::DataInput::SequenceSpan(2u, 2u));
        G_assert(__FILE__, __LINE__,
                 "==============NN5b-test::SetTrainSequences() Failed==============",
                 di5b->setTrainSequences(spans));
        G_assert(__FILE__, __LINE__,
                 "==============NN5b-test::SetTestSequences() Failed==============",
                 di5b->setTestSequences(spans));

        glades::InputLayerInfo* in5b = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden5b;
        hidden5b.push_back(new glades::HiddenLayerInfo(
            /*size*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        ));
        glades::OutputLayerInfo* out5b = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info5b = new glades::NNInfo("ut_rnn_multi_seq_reset", in5b, hidden5b, out5b);

        glades::NNetwork rnnNet5b(info5b, glades::NNetwork::TYPE_RNN);
        rnnNet5b.getTerminatorMutable().setEpoch(1);
        rnnNet5b.getTerminatorMutable().setAccuracy(0);

        rnnNet5b.graphMutable().build(info5b, di5b, glades::NNetwork::TYPE_RNN);
        rnnNet5b.setMustdBuildMeat(false);

        glades::Layer* hiddenLayer = rnnNet5b.graphMutable().getOutputLayer(1);
        glades::Layer* outLayer = rnnNet5b.graphMutable().getOutputLayer(2);
        glades::Node* hiddenNode = rnnNet5b.graphMutable().getOutputNode(hiddenLayer, 0);
        glades::Node* outNode = rnnNet5b.graphMutable().getOutputNode(outLayer, 0);
        G_assert(__FILE__, __LINE__,
                 "==============NN5b-test::NodesMissing() Failed==============",
                 (hiddenLayer != NULL) && (outLayer != NULL) && (hiddenNode != NULL) && (outNode != NULL));

        if (hiddenLayer && outLayer && hiddenNode && outNode)
        {
            hiddenLayer->setBiasWeight(0.0f);
            outLayer->setBiasWeight(0.0f);
            hiddenNode->setEdgeWeight(0, 1.0f); // Wx
            outNode->setEdgeWeight(0, 1.0f);    // Wy

            glades::Node* ctx = hiddenNode->getContextNode();
            G_assert(__FILE__, __LINE__,
                     "==============NN5b-test::ContextNodeMissing() Failed==============",
                     ctx != NULL);
            if (ctx)
            {
                ctx->setEdgeWeight(0, 1.0f); // Wh
                ctx->setWeight(999.0f);      // should be reset at sequence boundary
            }

            const glades::NNetworkStatus st = rnnNet5b.test(di5b);
            G_assert(__FILE__, __LINE__,
                     "==============NN5b-test::TestStatus() Failed==============",
                     st.ok());

            const float expected = 18.0f;
            const float tol = 1e-3f;
            const float yLast = outNode->getWeight();
            printf("[UT] RNN(multi-seq) yLast=%f (expected ~%f)\n", yLast, expected);
            G_assert(__FILE__, __LINE__,
                     "==============NN5b-test::RNNResetsAtSequenceBoundaries() Failed==============",
                     (yLast > expected - tol) && (yLast < expected + tol));
        }

        delete di5b;
        delete info5b;
    }

    printf("-----------------------------------\n");
    printf("NN Test 6 (RNN dataset: rnn.csv)\n");
    printf("-----------------------------------\n");

    // This test loads the rnn.csv dataset and runs a deterministic RNN forward pass over it.
    // We explicitly mark the output column to avoid relying on default heuristics.
    //
    // Dataset columns: x,y,z (z is output).
    // We'll configure a 2->1(hidden)->1(output) RNN with linear activations where:
    // - hidden tracks y (Wx_y=1, Wx_x=0, Wh=0, bias=0)
    // - output computes z = 0.5*y - 1.5
    //
    // This also exercises the RNN context update path over many timesteps.
    {
        shmea::GTable raw("datasets/rnn.csv", ',', shmea::GTable::TYPE_FILE);
        raw.clearOutputs();
        raw.toggleOutput(2); // z

        glades::NumberInput* di6 = new glades::NumberInput();
        // Keep raw values so the deterministic mapping below stays exact.
        di6->import(raw, /*standardizeFlag*/ glades::GMath::NONE);

        // Mirror train->test so test() evaluates the same imported rows.
        di6->testMatrix = di6->trainMatrix;
        di6->testExpectedMatrix = di6->trainExpectedMatrix;

        glades::InputLayerInfo* in6 = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden6;
        hidden6.push_back(new glades::HiddenLayerInfo(
            /*size*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        ));
        glades::OutputLayerInfo* out6 = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info6 = new glades::NNInfo("ut_rnn_csv_forward", in6, hidden6, out6);

        glades::NNetwork rnnNet6(info6, glades::NNetwork::TYPE_RNN);
        rnnNet6.getTerminatorMutable().setEpoch(1);
        rnnNet6.getTerminatorMutable().setAccuracy(0);

        rnnNet6.graphMutable().build(info6, di6, glades::NNetwork::TYPE_RNN);
        rnnNet6.setMustdBuildMeat(false);

        // Hidden layer (counter=1) and output layer (counter=2)
        glades::Layer* hiddenLayer6 = rnnNet6.graphMutable().getOutputLayer(1);
        glades::Layer* outLayer6 = rnnNet6.graphMutable().getOutputLayer(2);
        glades::Node* hiddenNode6 = rnnNet6.graphMutable().getOutputNode(hiddenLayer6, 0);
        glades::Node* outNode6 = rnnNet6.graphMutable().getOutputNode(outLayer6, 0);

        hiddenLayer6->setBiasWeight(0.0f);
        outLayer6->setBiasWeight(-1.5f);

        // Inputs are [x,y] in that order; output is z.
        hiddenNode6->setEdgeWeight(0, 0.0f); // Wx_x
        hiddenNode6->setEdgeWeight(1, 1.0f); // Wx_y

        glades::Node* ctx6 = hiddenNode6->getContextNode();
        G_assert(__FILE__, __LINE__,
                 "==============NN6-test::ContextNodeMissing() Failed==============",
                 ctx6 != NULL);
        if (ctx6)
        {
            ctx6->setEdgeWeight(0, 0.0f); // Wh = 0 so recurrence doesn't affect mapping
            ctx6->setWeight(12345.0f);    // should be reset to 0 at run start
        }

        outNode6->setEdgeWeight(0, 0.5f); // Wy

        rnnNet6.test(di6);
        G_assert(__FILE__, __LINE__,
                 "==============NN6-test::TestStatus() Failed==============",
                 rnnNet6.getLastStatus().ok());

        // Last row in dataset is: x=350.5, y=201, z=99
        const float expectedHLast = 201.0f;
        const float expectedZLast = 99.0f;
        const float tol = 1e-3f;

        const float zLast = outNode6->getWeight();
        const float hLast = hiddenNode6->getWeight();
        const float ctxLast = (ctx6 ? ctx6->getWeight() : 0.0f);

        printf("[UT] RNN(csv) last h=%f ctx=%f z=%f (expected h~%f z~%f)\n",
               hLast, ctxLast, zLast, expectedHLast, expectedZLast);

        G_assert(__FILE__, __LINE__,
                 "==============NN6-test::RNNCSV_LastHidden() Failed==============",
                 (hLast > expectedHLast - tol) && (hLast < expectedHLast + tol));
        G_assert(__FILE__, __LINE__,
                 "==============NN6-test::RNNCSV_ContextTracksHidden() Failed==============",
                 (ctxLast > expectedHLast - tol) && (ctxLast < expectedHLast + tol));
        G_assert(__FILE__, __LINE__,
                 "==============NN6-test::RNNCSV_LastOutput() Failed==============",
                 (zLast > expectedZLast - tol) && (zLast < expectedZLast + tol));

        // Should be effectively perfect on this dataset
        G_assert(__FILE__, __LINE__,
                 "==============NN6-test::RNNCSV_Accuracy() Failed==============",
                 rnnNet6.getAccuracy() > 99.0f);

        delete di6;
        delete info6; // owns in6/hidden6/out6
    }

    printf("-----------------------------------\n");
    printf("NN Test 6b (GRU deterministic forward pass)\n");
    printf("-----------------------------------\n");

    // Deterministic GRU forward semantics over 2 timesteps.
    //
    // We use a 1->1(GRU)->1 regression model with weights chosen to simplify the GRU:
    //   z_t = sigmoid(0) = 0.5
    //   r_t = sigmoid(0) = 0.5
    //   h~_t = tanh(Wh*x_t) with Wh=1
    //   h_t = (1 - z_t)*h_{t-1} + z_t*h~_t
    //   y_t = Wy*h_t with Wy=1
    //
    // With h0=0, x1=0, x2=2:
    //   h1 = 0.5*tanh(0) = 0
    //   h2 = 0.5*h1 + 0.5*tanh(2) = 0.5*tanh(2)
    {
        glades::NumberInput* di6b = new glades::NumberInput();
        di6b->trainMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di6b->trainExpectedMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di6b->trainMatrix[0][0] = 0.0f;
        di6b->trainMatrix[1][0] = 2.0f;
        // Not used by assertions (regression expected values only needed for shape checks)
        di6b->trainExpectedMatrix[0][0] = 0.0f;
        di6b->trainExpectedMatrix[1][0] = 0.0f;

        // Mirror train->test so test() evaluates the same two timesteps.
        di6b->testMatrix = di6b->trainMatrix;
        di6b->testExpectedMatrix = di6b->trainExpectedMatrix;

        glades::InputLayerInfo* in6b = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden6b;
        hidden6b.push_back(new glades::HiddenLayerInfo(
            /*size*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        ));
        glades::OutputLayerInfo* out6b = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info6b = new glades::NNInfo("ut_gru_forward", in6b, hidden6b, out6b);

        glades::NNetwork gruNet(info6b, glades::NNetwork::TYPE_GRU);
        gruNet.getTerminatorMutable().setEpoch(1);
        gruNet.getTerminatorMutable().setAccuracy(0);

        // Pre-build so we can set deterministic weights, then prevent rebuild in test()
        gruNet.graphMutable().build(info6b, di6b, glades::NNetwork::TYPE_GRU);
        gruNet.setMustdBuildMeat(false);

        glades::Layer* hiddenLayer = gruNet.graphMutable().getOutputLayer(1);
        glades::Layer* outLayer = gruNet.graphMutable().getOutputLayer(2);
        glades::Node* hiddenNode = gruNet.graphMutable().getOutputNode(hiddenLayer, 0);
        glades::Node* outNode = gruNet.graphMutable().getOutputNode(outLayer, 0);

        G_assert(__FILE__, __LINE__,
                 "==============NN6b-test::NodesMissing() Failed==============",
                 (hiddenLayer != NULL) && (outLayer != NULL) && (hiddenNode != NULL) && (outNode != NULL));

        if (hiddenLayer && outLayer && hiddenNode && outNode)
        {
            hiddenLayer->setBiasWeight(0.0f);
            outLayer->setBiasWeight(0.0f);

            // GRU layout for 1 input, 1 hidden:
            // Node edges: gateCount*(fanIn+1) = 3*(1+1) = 6
            //   z: Wz at 0, bz at 1
            //   r: Wr at 2, br at 3
            //   h: Wh at 4, bh at 5
            hiddenNode->setEdgeWeight(0, 0.0f); // Wz
            hiddenNode->setEdgeWeight(1, 0.0f); // bz
            hiddenNode->setEdgeWeight(2, 0.0f); // Wr
            hiddenNode->setEdgeWeight(3, 0.0f); // br
            hiddenNode->setEdgeWeight(4, 1.0f); // Wh
            hiddenNode->setEdgeWeight(5, 0.0f); // bh

            // Context edges: gateCount*hiddenSize = 3*1 = 3 => Uz, Ur, Uh
            glades::Node* ctx = hiddenNode->getContextNode();
            G_assert(__FILE__, __LINE__,
                     "==============NN6b-test::ContextNodeMissing() Failed==============",
                     ctx != NULL);
            if (ctx)
            {
                ctx->setEdgeWeight(0, 0.0f); // Uz
                ctx->setEdgeWeight(1, 0.0f); // Ur
                ctx->setEdgeWeight(2, 0.0f); // Uh
                ctx->setWeight(123.0f);       // should be reset at run start
            }

            // Output: y = Wy*h + b, with Wy=1 and b=0 (per-neuron bias edge)
            outNode->setEdgeWeight(0, 1.0f); // Wy
            outNode->setEdgeWeight(1, 0.0f); // b

            const glades::NNetworkStatus st = gruNet.test(di6b);
            G_assert(__FILE__, __LINE__,
                     "==============NN6b-test::TestStatus() Failed==============",
                     st.ok());

            const float expected = 0.5f * static_cast<float>(tanh(2.0));
            const float tol = 1e-3f;
            const float yLast = outNode->getWeight();
            const float hLast = hiddenNode->getWeight();
            const float ctxLast = (ctx ? ctx->getWeight() : 0.0f);
            printf("[UT] GRU yLast=%f hLast=%f ctx=%f (expected ~%f)\n", yLast, hLast, ctxLast, expected);
            G_assert(__FILE__, __LINE__,
                     "==============NN6b-test::GRUForward_Output() Failed==============",
                     (yLast > expected - tol) && (yLast < expected + tol));
            G_assert(__FILE__, __LINE__,
                     "==============NN6b-test::GRUForward_Hidden() Failed==============",
                     (hLast > expected - tol) && (hLast < expected + tol));
            G_assert(__FILE__, __LINE__,
                     "==============NN6b-test::GRUForward_ContextTracksHidden() Failed==============",
                     (ctxLast > expected - tol) && (ctxLast < expected + tol));
        }

        delete di6b;
        delete info6b; // owns in6b/hidden6b/out6b
    }

    printf("-----------------------------------\n");
    printf("NN Test 6c (LSTM deterministic forward pass)\n");
    printf("-----------------------------------\n");

    // Deterministic LSTM forward semantics over 2 timesteps.
    //
    // We use a 1->1(LSTM)->1 regression model with weights chosen to simplify the LSTM:
    //   i_t = sigmoid(0) = 0.5
    //   f_t = sigmoid(0) = 0.5
    //   o_t = sigmoid(0) = 0.5
    //   g_t = tanh(Wg*x_t) with Wg=1
    //   c_t = f_t*c_{t-1} + i_t*g_t
    //   h_t = o_t*tanh(c_t)
    //   y_t = Wy*h_t with Wy=1
    //
    // With c0=h0=0, x1=0, x2=2:
    //   c1=0, h1=0
    //   c2=0.5*tanh(2), h2=0.5*tanh(c2)
    {
        glades::NumberInput* di6c = new glades::NumberInput();
        di6c->trainMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di6c->trainExpectedMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di6c->trainMatrix[0][0] = 0.0f;
        di6c->trainMatrix[1][0] = 2.0f;
        di6c->trainExpectedMatrix[0][0] = 0.0f;
        di6c->trainExpectedMatrix[1][0] = 0.0f;

        di6c->testMatrix = di6c->trainMatrix;
        di6c->testExpectedMatrix = di6c->trainExpectedMatrix;

        glades::InputLayerInfo* in6c = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden6c;
        hidden6c.push_back(new glades::HiddenLayerInfo(
            /*size*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        ));
        glades::OutputLayerInfo* out6c = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info6c = new glades::NNInfo("ut_lstm_forward", in6c, hidden6c, out6c);

        glades::NNetwork lstmNet(info6c, glades::NNetwork::TYPE_LSTM);
        lstmNet.getTerminatorMutable().setEpoch(1);
        lstmNet.getTerminatorMutable().setAccuracy(0);

        lstmNet.graphMutable().build(info6c, di6c, glades::NNetwork::TYPE_LSTM);
        lstmNet.setMustdBuildMeat(false);

        glades::Layer* hiddenLayer = lstmNet.graphMutable().getOutputLayer(1);
        glades::Layer* outLayer = lstmNet.graphMutable().getOutputLayer(2);
        glades::Node* hiddenNode = lstmNet.graphMutable().getOutputNode(hiddenLayer, 0);
        glades::Node* outNode = lstmNet.graphMutable().getOutputNode(outLayer, 0);

        G_assert(__FILE__, __LINE__,
                 "==============NN6c-test::NodesMissing() Failed==============",
                 (hiddenLayer != NULL) && (outLayer != NULL) && (hiddenNode != NULL) && (outNode != NULL));

        if (hiddenLayer && outLayer && hiddenNode && outNode)
        {
            hiddenLayer->setBiasWeight(0.0f);
            outLayer->setBiasWeight(0.0f);

            // LSTM layout for 1 input, 1 hidden:
            // Node edges: gateCount*(fanIn+1) = 4*(1+1) = 8
            //   i: Wi at 0, bi at 1
            //   f: Wf at 2, bf at 3
            //   o: Wo at 4, bo at 5
            //   g: Wg at 6, bg at 7
            hiddenNode->setEdgeWeight(0, 0.0f); // Wi
            hiddenNode->setEdgeWeight(1, 0.0f); // bi
            hiddenNode->setEdgeWeight(2, 0.0f); // Wf
            hiddenNode->setEdgeWeight(3, 0.0f); // bf
            hiddenNode->setEdgeWeight(4, 0.0f); // Wo
            hiddenNode->setEdgeWeight(5, 0.0f); // bo
            hiddenNode->setEdgeWeight(6, 1.0f); // Wg
            hiddenNode->setEdgeWeight(7, 0.0f); // bg

            // Context edges: gateCount*hiddenSize = 4*1 = 4 => Ui, Uf, Uo, Ug
            glades::Node* ctx = hiddenNode->getContextNode();
            G_assert(__FILE__, __LINE__,
                     "==============NN6c-test::ContextNodeMissing() Failed==============",
                     ctx != NULL);
            if (ctx)
            {
                ctx->setEdgeWeight(0, 0.0f); // Ui
                ctx->setEdgeWeight(1, 0.0f); // Uf
                ctx->setEdgeWeight(2, 0.0f); // Uo
                ctx->setEdgeWeight(3, 0.0f); // Ug
                ctx->setWeight(456.0f);      // should be reset at run start
            }

            outNode->setEdgeWeight(0, 1.0f); // Wy
            outNode->setEdgeWeight(1, 0.0f); // b

            const glades::NNetworkStatus st = lstmNet.test(di6c);
            G_assert(__FILE__, __LINE__,
                     "==============NN6c-test::TestStatus() Failed==============",
                     st.ok());

            const float c2 = 0.5f * static_cast<float>(tanh(2.0));
            const float expected = 0.5f * static_cast<float>(tanh(static_cast<double>(c2)));
            const float tol = 1e-3f;
            const float yLast = outNode->getWeight();
            const float hLast = hiddenNode->getWeight();
            const float ctxLast = (ctx ? ctx->getWeight() : 0.0f);
            printf("[UT] LSTM yLast=%f hLast=%f ctx=%f (expected ~%f)\n", yLast, hLast, ctxLast, expected);
            G_assert(__FILE__, __LINE__,
                     "==============NN6c-test::LSTMForward_Output() Failed==============",
                     (yLast > expected - tol) && (yLast < expected + tol));
            G_assert(__FILE__, __LINE__,
                     "==============NN6c-test::LSTMForward_Hidden() Failed==============",
                     (hLast > expected - tol) && (hLast < expected + tol));
            G_assert(__FILE__, __LINE__,
                     "==============NN6c-test::LSTMForward_ContextTracksHidden() Failed==============",
                     (ctxLast > expected - tol) && (ctxLast < expected + tol));
        }

        delete di6c;
        delete info6c; // owns in6c/hidden6c/out6c
    }

    printf("-----------------------------------\n");
    printf("NN Test 7 (RNN full BPTT: future loss updates earlier Wx)\n");
    printf("-----------------------------------\n");

    // This test detects whether gradients propagate through time (BPTT).
    //
    // We use a 1->1->1 linear RNN (no biases) with weights:
    //   h_t = Wx*x_t + Wh*h_{t-1}
    //   y_t = Wy*h_t
    // Initialize Wx=Wh=Wy=1, h0=0
    //
    // Sequence: x1=1, x2=0
    // Targets:  y1=1 (matches initial y1), y2=0 (incurs loss only at t=2)
    //
    // With full BPTT, the loss at t=2 should backprop to t=1 and update Wx (because x1 influences h1,
    // which influences y2 via recurrence). Without BPTT, Wx would not change because x2=0.
    //
    // With lr=0.1 and mean-over-T update (T=2), expected Wx becomes ~0.9 after 1 epoch.
    {
        glades::NumberInput* di7 = new glades::NumberInput();
        di7->trainMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di7->trainExpectedMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di7->trainMatrix[0][0] = 1.0f;  // x1
        di7->trainMatrix[1][0] = 0.0f;  // x2
        di7->trainExpectedMatrix[0][0] = 1.0f; // y1 target (no loss at t=1)
        di7->trainExpectedMatrix[1][0] = 0.0f; // y2 target (loss at t=2)

        glades::InputLayerInfo* in7 = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ 0.1f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden7;
        hidden7.push_back(new glades::HiddenLayerInfo(
            /*size*/ 1,
            /*learningRate*/ 0.1f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        ));
        glades::OutputLayerInfo* out7 = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info7 = new glades::NNInfo("ut_rnn_bptt", in7, hidden7, out7);

        glades::NNetwork rnnNet7(info7, glades::NNetwork::TYPE_RNN);
        rnnNet7.getTerminatorMutable().setEpoch(1);
        rnnNet7.getTerminatorMutable().setAccuracy(0);

        rnnNet7.graphMutable().build(info7, di7, glades::NNetwork::TYPE_RNN);
        rnnNet7.setMustdBuildMeat(false);

        glades::Layer* hiddenLayer7 = rnnNet7.graphMutable().getOutputLayer(1);
        glades::Layer* outLayer7 = rnnNet7.graphMutable().getOutputLayer(2);
        glades::Node* hiddenNode7 = rnnNet7.graphMutable().getOutputNode(hiddenLayer7, 0);
        glades::Node* outNode7 = rnnNet7.graphMutable().getOutputNode(outLayer7, 0);

        hiddenLayer7->setBiasWeight(0.0f);
        outLayer7->setBiasWeight(0.0f);

        // Wx=1, Wy=1
        hiddenNode7->setEdgeWeight(0, 1.0f);
        outNode7->setEdgeWeight(0, 1.0f);

        // Wh=1 and ensure context resets from a non-zero starting value
        glades::Node* ctx7 = hiddenNode7->getContextNode();
        G_assert(__FILE__, __LINE__,
                 "==============NN7-test::ContextNodeMissing() Failed==============",
                 ctx7 != NULL);
        if (ctx7)
        {
            ctx7->setEdgeWeight(0, 1.0f); // Wh
            ctx7->setWeight(10.0f);       // should be reset to 0 at run start
        }

        // Train exactly 1 epoch
        const glades::NNetworkStatus stTrain7 = rnnNet7.train(di7);
        G_assert(__FILE__, __LINE__,
                 "==============NN7-test::TrainStatus() Failed==============",
                 stTrain7.ok());

        rnnNet7.materializeGraphParameters();

        /*
         * Accuracy is not meant to be 100%
         */
        const float wxFinal = hiddenNode7->getEdgeWeight(0);
        const float expectedWx = 0.9f;
        const float tol = 1e-3f;
        printf("[UT] RNN(BPTT) Wx final=%f (expected ~%f)\n", wxFinal, expectedWx);
        G_assert(__FILE__, __LINE__,
                 "==============NN7-test::RNNBPTT_UpdatesWxFromFutureLoss() Failed==============",
                 (wxFinal > expectedWx - tol) && (wxFinal < expectedWx + tol));

        delete di7;
        delete info7;
    }

    printf("-----------------------------------\n");
    printf("NN Test 8 (RNN train on rnn.csv with BPTT)\n");
    printf("-----------------------------------\n");

    // This test trains an RNN end-to-end on datasets/rnn.csv and expects high accuracy.
    // It is similar in spirit to Tests 1-3: train the network and assert accuracy threshold.
    //
    // Dataset columns: x,y,z (z is output).
    // We'll use a small linear RNN. Context recurrence is allowed but initialized to 0.
    {
        shmea::GTable raw("datasets/rnn.csv", ',', shmea::GTable::TYPE_FILE);
        raw.clearOutputs();
        raw.toggleOutput(2); // z

        glades::NumberInput* di8 = new glades::NumberInput();
        // Use NumberInput's built-in min-max normalization; for this dataset it makes
        // y_norm == z_norm exactly (since y = 2z + 3), which is ideal for a stability-focused
        // end-to-end training test.
        di8->import(raw, /*standardizeFlag*/ glades::GMath::MINMAX);

        // Mirror train->test so test() evaluates the same imported rows.
        di8->testMatrix = di8->trainMatrix;
        di8->testExpectedMatrix = di8->trainExpectedMatrix;

        const float lr = 0.05f;
        glades::InputLayerInfo* in8 = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ lr,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden8;
        hidden8.push_back(new glades::HiddenLayerInfo(
            /*size*/ 1,
            /*learningRate*/ lr,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        ));
        glades::OutputLayerInfo* out8 = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info8 = new glades::NNInfo("ut_rnn_csv_train_bptt", in8, hidden8, out8);

        glades::NNetwork rnnNet8(info8, glades::NNetwork::TYPE_RNN);
        rnnNet8.getTerminatorMutable().setEpoch(500);

        // Build so we can set deterministic starting weights, then prevent rebuild in train()
        rnnNet8.graphMutable().build(info8, di8, glades::NNetwork::TYPE_RNN);
        rnnNet8.setMustdBuildMeat(false);

        glades::Layer* hiddenLayer8 = rnnNet8.graphMutable().getOutputLayer(1);
        glades::Layer* outLayer8 = rnnNet8.graphMutable().getOutputLayer(2);
        glades::Node* hiddenNode8 = rnnNet8.graphMutable().getOutputNode(hiddenLayer8, 0);
        glades::Node* outNode8 = rnnNet8.graphMutable().getOutputNode(outLayer8, 0);

        hiddenLayer8->setBiasWeight(0.0f);
        // In MINMAX space for this dataset:
        //   y_norm = (y - 3) / 198
        //   z_norm = (z - 0) / 99
        // Since y = 2z + 3, we have y_norm == z_norm exactly.
        outLayer8->setBiasWeight(0.0f);

        // Inputs are [x,y] in that order. Start close to the known mapping: z ≈ 0.5*y - 1.5
        // but not perfect, so training must still move.
        hiddenNode8->setEdgeWeight(0, 0.0f); // Wx_x
        hiddenNode8->setEdgeWeight(1, 1.0f); // Wx_y (copy y_norm)
        outNode8->setEdgeWeight(0, 0.8f);    // Wy (close to 1.0 but not perfect)

        glades::Node* ctx8 = hiddenNode8->getContextNode();
        G_assert(__FILE__, __LINE__,
                 "==============NN8-test::ContextNodeMissing() Failed==============",
                 ctx8 != NULL);
        if (ctx8)
        {
            ctx8->setEdgeWeight(0, 0.0f); // Wh initialized to 0
            ctx8->setWeight(0.0f);
        }

        const glades::NNetworkStatus stTrain8 = rnnNet8.train(di8);
        G_assert(__FILE__, __LINE__,
                 "==============NN8-test::TrainStatus() Failed==============",
                 stTrain8.ok());

        // Evaluate full-sequence accuracy over timesteps (RUN_TEST path reports timestep-average)
        const glades::NNetworkStatus stTest8 = rnnNet8.test(di8);
        G_assert(__FILE__, __LINE__,
                 "==============NN8-test::TestStatus() Failed==============",
                 stTest8.ok());

        printf("[UT] RNN(rnn.csv) accuracy=%f%%\n", rnnNet8.getAccuracy());
        G_assert(__FILE__, __LINE__,
                 "==============NN8-test::RNNCSV_TrainAccuracy() Failed==============",
                 rnnNet8.getAccuracy() >= 95.0f);

        delete di8;
        delete info8;
    }

    printf("-----------------------------------\n");
    printf("NN Test 9 (GRU deterministic forward)\n");
    printf("-----------------------------------\n");

    // Deterministic 1->1->1 GRU forward pass.
    // Gates are saturated so the recurrence is easy to predict:
    // - z ~= 1, r ~= 1 (bias +20, weights 0)
    // - candidate = tanh(x) (Wh=1, Uh=0, bias 0)
    // - output = hidden (Wy=1, bias 0)
    // Sequence: x=[0,1] => y2 ~= tanh(1)
    {
        glades::NumberInput* di9 = new glades::NumberInput();
        di9->trainMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di9->trainExpectedMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di9->trainMatrix[0][0] = 0.0f;
        di9->trainMatrix[1][0] = 1.0f;

        // Mirror train->test so test() evaluates the same rows.
        di9->testMatrix = di9->trainMatrix;
        di9->testExpectedMatrix = di9->trainExpectedMatrix;

        glades::InputLayerInfo* in9 = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden9;
        hidden9.push_back(new glades::HiddenLayerInfo(
            /*size*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        ));
        glades::OutputLayerInfo* out9 = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info9 = new glades::NNInfo("ut_gru_forward", in9, hidden9, out9);

        glades::NNetwork gruNet(info9, glades::NNetwork::TYPE_GRU);
        gruNet.getTerminatorMutable().setEpoch(1);
        gruNet.getTerminatorMutable().setAccuracy(0);

        gruNet.graphMutable().build(info9, di9, glades::NNetwork::TYPE_GRU);
        gruNet.setMustdBuildMeat(false);

        glades::Layer* hiddenLayer9 = gruNet.graphMutable().getOutputLayer(1);
        glades::Layer* outLayer9 = gruNet.graphMutable().getOutputLayer(2);
        glades::Node* hNode9 = gruNet.graphMutable().getOutputNode(hiddenLayer9, 0);
        glades::Node* yNode9 = gruNet.graphMutable().getOutputNode(outLayer9, 0);

        G_assert(__FILE__, __LINE__,
                 "==============NN9-test::NodesMissing() Failed==============",
                 (hiddenLayer9 != NULL) && (outLayer9 != NULL) && (hNode9 != NULL) && (yNode9 != NULL));

        glades::Node* ctx9 = (hNode9 ? hNode9->getContextNode() : NULL);
        G_assert(__FILE__, __LINE__,
                 "==============NN9-test::ContextNodeMissing() Failed==============",
                 ctx9 != NULL);

        if (hNode9 && yNode9 && ctx9)
        {
            // Hidden node edge layout (prevSize=1, stride=2):
            // z: [w0,b] => idx 0,1
            // r: [w0,b] => idx 2,3
            // h: [w0,b] => idx 4,5
            hNode9->setEdgeWeight(0, 0.0f);  // Wz
            hNode9->setEdgeWeight(1, 20.0f); // bz
            hNode9->setEdgeWeight(2, 0.0f);  // Wr
            hNode9->setEdgeWeight(3, 20.0f); // br
            hNode9->setEdgeWeight(4, 1.0f);  // Wh
            hNode9->setEdgeWeight(5, 0.0f);  // bh

            // Recurrent weights: [Uz,Ur,Uh] each length hiddenSize(=1)
            ctx9->setEdgeWeight(0, 0.0f);
            ctx9->setEdgeWeight(1, 0.0f);
            ctx9->setEdgeWeight(2, 0.0f);

            // Output: y = h
            yNode9->setEdgeWeight(0, 1.0f);
            yNode9->setEdgeWeight(1, 0.0f); // bias edge
        }

        const glades::NNetworkStatus st9 = gruNet.test(di9);
        G_assert(__FILE__, __LINE__,
                 "==============NN9-test::TestStatus() Failed==============",
                 st9.ok());

        const float expected = static_cast<float>(tanh(1.0));
        const float y2 = (yNode9 ? yNode9->getWeight() : 0.0f);
        const float tol = 1e-3f;
        printf("[UT] GRU y2=%f (expected ~%f)\n", y2, expected);
        G_assert(__FILE__, __LINE__,
                 "==============NN9-test::GRUForward() Failed==============",
                 (y2 > expected - tol) && (y2 < expected + tol));

        delete di9;
        delete info9;
    }

    printf("-----------------------------------\n");
    printf("NN Test 10 (LSTM deterministic forward)\n");
    printf("-----------------------------------\n");

    // Deterministic 1->1->1 LSTM forward pass.
    // Gates are saturated so the dynamics are easy to predict:
    // - i ~= 1, f ~= 0, o ~= 1 (bias +20, -20, +20; weights 0)
    // - g = tanh(x) (Wg=1, Ug=0, bias 0)
    // => c2 = tanh(1), h2 = tanh(c2) = tanh(tanh(1))
    {
        glades::NumberInput* di10 = new glades::NumberInput();
        di10->trainMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di10->trainExpectedMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di10->trainMatrix[0][0] = 0.0f;
        di10->trainMatrix[1][0] = 1.0f;

        // Mirror train->test so test() evaluates the same rows.
        di10->testMatrix = di10->trainMatrix;
        di10->testExpectedMatrix = di10->trainExpectedMatrix;

        glades::InputLayerInfo* in10 = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden10;
        hidden10.push_back(new glades::HiddenLayerInfo(
            /*size*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        ));
        glades::OutputLayerInfo* out10 = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info10 = new glades::NNInfo("ut_lstm_forward", in10, hidden10, out10);

        glades::NNetwork lstmNet(info10, glades::NNetwork::TYPE_LSTM);
        lstmNet.getTerminatorMutable().setEpoch(1);
        lstmNet.getTerminatorMutable().setAccuracy(0);

        lstmNet.graphMutable().build(info10, di10, glades::NNetwork::TYPE_LSTM);
        lstmNet.setMustdBuildMeat(false);

        glades::Layer* hiddenLayer10 = lstmNet.graphMutable().getOutputLayer(1);
        glades::Layer* outLayer10 = lstmNet.graphMutable().getOutputLayer(2);
        glades::Node* hNode10 = lstmNet.graphMutable().getOutputNode(hiddenLayer10, 0);
        glades::Node* yNode10 = lstmNet.graphMutable().getOutputNode(outLayer10, 0);

        G_assert(__FILE__, __LINE__,
                 "==============NN10-test::NodesMissing() Failed==============",
                 (hiddenLayer10 != NULL) && (outLayer10 != NULL) && (hNode10 != NULL) && (yNode10 != NULL));

        glades::Node* ctx10 = (hNode10 ? hNode10->getContextNode() : NULL);
        G_assert(__FILE__, __LINE__,
                 "==============NN10-test::ContextNodeMissing() Failed==============",
                 ctx10 != NULL);

        if (hNode10 && yNode10 && ctx10)
        {
            // Hidden node edge layout (prevSize=1, stride=2), gate order [i,f,o,g]:
            // i: idx 0,1
            // f: idx 2,3
            // o: idx 4,5
            // g: idx 6,7
            hNode10->setEdgeWeight(0, 0.0f);   // Wi
            hNode10->setEdgeWeight(1, 20.0f);  // bi
            hNode10->setEdgeWeight(2, 0.0f);   // Wf
            hNode10->setEdgeWeight(3, -20.0f); // bf
            hNode10->setEdgeWeight(4, 0.0f);   // Wo
            hNode10->setEdgeWeight(5, 20.0f);  // bo
            hNode10->setEdgeWeight(6, 1.0f);   // Wg
            hNode10->setEdgeWeight(7, 0.0f);   // bg

            // Recurrent weights: [Ui,Uf,Uo,Ug] each length hiddenSize(=1)
            ctx10->setEdgeWeight(0, 0.0f);
            ctx10->setEdgeWeight(1, 0.0f);
            ctx10->setEdgeWeight(2, 0.0f);
            ctx10->setEdgeWeight(3, 0.0f);

            // Output: y = h
            yNode10->setEdgeWeight(0, 1.0f);
            yNode10->setEdgeWeight(1, 0.0f); // bias edge
        }

        const glades::NNetworkStatus st10 = lstmNet.test(di10);
        G_assert(__FILE__, __LINE__,
                 "==============NN10-test::TestStatus() Failed==============",
                 st10.ok());

        const float expected = static_cast<float>(tanh(tanh(1.0)));
        const float y2 = (yNode10 ? yNode10->getWeight() : 0.0f);
        const float c2 = (hNode10 ? hNode10->getCellState() : 0.0f);
        const float expectedC = static_cast<float>(tanh(1.0));
        const float tol = 1e-3f;
        printf("[UT] LSTM y2=%f c2=%f (expected y~%f c~%f)\n", y2, c2, expected, expectedC);
        G_assert(__FILE__, __LINE__,
                 "==============NN10-test::LSTMForward() Failed==============",
                 (y2 > expected - tol) && (y2 < expected + tol));
        G_assert(__FILE__, __LINE__,
                 "==============NN10-test::LSTMCellState() Failed==============",
                 (c2 > expectedC - tol) && (c2 < expectedC + tol));

        delete di10;
        delete info10;
    }

    printf("-----------------------------------\n");
    printf("NN Test 11 (GRU BPTT: future loss updates candidate Wx)\n");
    printf("-----------------------------------\n");

    // This test verifies that GRU gradients propagate through time (BPTT).
    //
    // We configure a 1->1->1 GRU to behave like a simple tanh RNN by saturating gates:
    // - z ~= 1, r ~= 1 via large positive biases (weights 0)
    // Then h_t = tanh(Wx * x_t + Uh * h_{t-1})
    // and y_t = Wy * h_t.
    //
    // Sequence: x1=1, x2=0. Targets: y1 == y1_pred (no loss), y2=0 (loss only at t=2).
    // With BPTT, that future loss must update Wx (since x1 influences h1 which influences y2).
    {
        glades::NumberInput* di11 = new glades::NumberInput();
        di11->trainMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di11->trainExpectedMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di11->trainMatrix[0][0] = 1.0f; // x1
        di11->trainMatrix[1][0] = 0.0f; // x2

        const float a = static_cast<float>(tanh(1.0));   // h1 with Wx=1, Uh=1, h0=0
        di11->trainExpectedMatrix[0][0] = a;             // y1 target matches prediction => no loss at t=1
        di11->trainExpectedMatrix[1][0] = 0.0f;          // y2 target => loss at t=2 only

        const float lr = 0.1f;
        glades::InputLayerInfo* in11 = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ lr,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden11;
        hidden11.push_back(new glades::HiddenLayerInfo(
            /*size*/ 1,
            /*learningRate*/ lr,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        ));
        glades::OutputLayerInfo* out11 = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info11 = new glades::NNInfo("ut_gru_bptt", in11, hidden11, out11);

        glades::NNetwork gruNet11(info11, glades::NNetwork::TYPE_GRU);
        gruNet11.getTerminatorMutable().setEpoch(1);
        gruNet11.getTerminatorMutable().setAccuracy(0);

        gruNet11.graphMutable().build(info11, di11, glades::NNetwork::TYPE_GRU);
        gruNet11.setMustdBuildMeat(false);

        glades::Layer* hiddenLayer11 = gruNet11.graphMutable().getOutputLayer(1);
        glades::Layer* outLayer11 = gruNet11.graphMutable().getOutputLayer(2);
        glades::Node* hNode11 = gruNet11.graphMutable().getOutputNode(hiddenLayer11, 0);
        glades::Node* yNode11 = gruNet11.graphMutable().getOutputNode(outLayer11, 0);
        glades::Node* ctx11 = (hNode11 ? hNode11->getContextNode() : NULL);

        G_assert(__FILE__, __LINE__,
                 "==============NN11-test::NodesMissing() Failed==============",
                 (hiddenLayer11 != NULL) && (outLayer11 != NULL) && (hNode11 != NULL) && (yNode11 != NULL) && (ctx11 != NULL));

        if (hNode11 && yNode11 && ctx11)
        {
            // Hidden node edge layout (prevSize=1, stride=2):
            // z: idx 0,1
            // r: idx 2,3
            // h: idx 4,5
            hNode11->setEdgeWeight(0, 0.0f);  // Wz
            hNode11->setEdgeWeight(1, 20.0f); // bz => z ~= 1
            hNode11->setEdgeWeight(2, 0.0f);  // Wr
            hNode11->setEdgeWeight(3, 20.0f); // br => r ~= 1
            hNode11->setEdgeWeight(4, 1.0f);  // Wh (candidate input weight)  <-- we assert this changes
            hNode11->setEdgeWeight(5, 0.0f);  // bh

            // Recurrent weights: [Uz,Ur,Uh]
            ctx11->setEdgeWeight(0, 0.0f);
            ctx11->setEdgeWeight(1, 0.0f);
            ctx11->setEdgeWeight(2, 1.0f); // Uh = 1 so future loss flows back to t=1
            ctx11->setWeight(0.0f);

            // Output: y = h (linear)
            yNode11->setEdgeWeight(0, 1.0f);
            yNode11->setEdgeWeight(1, 0.0f); // bias edge
        }

        const glades::NNetworkStatus st11 = gruNet11.train(di11);
        G_assert(__FILE__, __LINE__,
                 "==============NN11-test::TrainStatus() Failed==============",
                 st11.ok());

        gruNet11.materializeGraphParameters();

        // Expected update (mirrors network.cpp GRU math, with window length T=2 averaging):
        // h1 = tanh(1) = a
        // h2 = tanh(a) = b
        // deltaY2 = 2*(b - 0) (linear output, MSE)
        // daH2 = deltaY2 * (1 - b^2)
        // dh1_from_future = daH2 * Uh (Uh=1)
        // daH1 = dh1_from_future * (1 - a^2)
        // Wx_new = 1 - lr * daH1 / T
        const float b = static_cast<float>(tanh(static_cast<double>(a)));
        const float deltaY2 = 2.0f * (b - 0.0f);
        const float daH2 = deltaY2 * (1.0f - (b * b));
        const float daH1 = daH2 * (1.0f - (a * a));
        const float expectedWx = 1.0f - (lr * daH1 / 2.0f);

        const float wxFinal = (hNode11 ? hNode11->getEdgeWeight(4) : 0.0f);
        const float tol = 2e-3f;
        printf("[UT] GRU(BPTT) Wx final=%f (expected ~%f)\n", wxFinal, expectedWx);
        G_assert(__FILE__, __LINE__,
                 "==============NN11-test::GRUBPTT_UpdatesWxFromFutureLoss() Failed==============",
                 (wxFinal > expectedWx - tol) && (wxFinal < expectedWx + tol));

        delete di11;
        delete info11;
    }

    printf("-----------------------------------\n");
    printf("NN Test 12 (Determinism: same seed => same trained weights)\n");
    printf("-----------------------------------\n");

    // This test verifies that the ML engine no longer depends on global rand()/srand()
    // behavior and is reproducible when using NNetwork::setSeed().
    //
    // We train two identical networks with the same seed on the same dataset for the
    // same number of epochs, with non-zero dropout to exercise stochasticity.
    // The final weights must match (within floating tolerance).
    {
        // Toy dataset: 4 samples, 2 features, 1 regression output
        glades::NumberInput* di12 = new glades::NumberInput();
        di12->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
        di12->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));

        di12->trainMatrix[0][0] = 0.0f; di12->trainMatrix[0][1] = 0.0f; di12->trainExpectedMatrix[0][0] = 0.0f;
        di12->trainMatrix[1][0] = 1.0f; di12->trainMatrix[1][1] = 0.0f; di12->trainExpectedMatrix[1][0] = 1.0f;
        di12->trainMatrix[2][0] = 0.0f; di12->trainMatrix[2][1] = 1.0f; di12->trainExpectedMatrix[2][0] = 1.0f;
        di12->trainMatrix[3][0] = 1.0f; di12->trainMatrix[3][1] = 1.0f; di12->trainExpectedMatrix[3][0] = 2.0f;

        const float lr = 0.05f;
        glades::InputLayerInfo* in12a = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ lr,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.5f, // exercise input dropout
            /*activationType*/ glades::GMath::TANH,
            /*activationParam*/ 0.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden12a;
        hidden12a.push_back(new glades::HiddenLayerInfo(
            /*size*/ 3,
            /*learningRate*/ lr,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.5f, // exercise hidden dropout
            /*activationType*/ glades::GMath::TANH,
            /*activationParam*/ 0.0f
        ));
        glades::OutputLayerInfo* out12a = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info12a = new glades::NNInfo("ut_determinism_a", in12a, hidden12a, out12a);

        // Create a second, identical skeleton (do not share the same NNInfo instance).
        glades::InputLayerInfo* in12b = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ lr,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.5f,
            /*activationType*/ glades::GMath::TANH,
            /*activationParam*/ 0.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden12b;
        hidden12b.push_back(new glades::HiddenLayerInfo(
            /*size*/ 3,
            /*learningRate*/ lr,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.5f,
            /*activationType*/ glades::GMath::TANH,
            /*activationParam*/ 0.0f
        ));
        glades::OutputLayerInfo* out12b = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info12b = new glades::NNInfo("ut_determinism_b", in12b, hidden12b, out12b);

        glades::NNetwork netA(info12a, glades::NNetwork::TYPE_DFF);
        glades::NNetwork netB(info12b, glades::NNetwork::TYPE_DFF);
        netA.getTerminatorMutable().setEpoch(3);
        netA.getTerminatorMutable().setAccuracy(0);
        netB.getTerminatorMutable().setEpoch(3);
        netB.getTerminatorMutable().setAccuracy(0);

        const uint64_t seed = 123456789ULL;

        netA.setSeed(seed);
        const glades::NNetworkStatus stA = netA.train(di12);
        G_assert(__FILE__, __LINE__, "==============NN12-test::TrainStatus_A() Failed==============", stA.ok());

        netB.setSeed(seed);
        const glades::NNetworkStatus stB = netB.train(di12);
        G_assert(__FILE__, __LINE__, "==============NN12-test::TrainStatus_B() Failed==============", stB.ok());

        netA.materializeGraphParameters();
        netB.materializeGraphParameters();

        // Compare all feedforward weights (including bias edges) across layers.
        double sumA = 0.0;
        double sumB = 0.0;
        unsigned int countA = 0;
        unsigned int countB = 0;

        const unsigned int layersA = netA.graphMutable().getLayersSize();
        const unsigned int layersB = netB.graphMutable().getLayersSize();
        G_assert(__FILE__, __LINE__, "==============NN12-test::LayerCountMismatch() Failed==============", layersA == layersB);

        const unsigned int L = std::min(layersA, layersB);
        for (unsigned int li = 1; li <= L; ++li)
        {
            glades::Layer* la = netA.graphMutable().getOutputLayer(li);
            glades::Layer* lb = netB.graphMutable().getOutputLayer(li);
            if (!la || !lb) continue;

            const unsigned int na = la->size();
            const unsigned int nb = lb->size();
            if (na != nb) continue;

            for (unsigned int j = 0; j < na; ++j)
            {
                glades::Node* a = netA.graphMutable().getOutputNode(la, j);
                glades::Node* b = netB.graphMutable().getOutputNode(lb, j);
                if (!a || !b) continue;
                if (a->numEdges() != b->numEdges()) continue;

                for (unsigned int e = 0; e < a->numEdges(); ++e)
                {
                    sumA += static_cast<double>(a->getEdgeWeight(e));
                    sumB += static_cast<double>(b->getEdgeWeight(e));
                    ++countA;
                    ++countB;
                }
            }
        }

        printf("[UT] Determinism sums: A=%f B=%f (count=%u)\n", (float)sumA, (float)sumB, countA);
        const double tol = 1e-6;
        G_assert(__FILE__, __LINE__, "==============NN12-test::WeightCountMismatch() Failed==============", countA == countB && countA > 0);
        G_assert(__FILE__, __LINE__, "==============NN12-test::DeterministicWeights() Failed==============", fabs(sumA - sumB) < tol);

        delete di12;
        delete info12a;
        delete info12b;
    }

    printf("-----------------------------------\n");
    printf("NN Test 13 (LR schedule: exp decay multiplier + effective LR)\n");
    printf("-----------------------------------\n");

    // This test verifies the modern learning-rate schedule plumbing:
    // - setLearningRateScheduleExp(gamma) produces lrMultiplier = gamma^(epoch-1)
    // - the reported metrics.learningRate equals baseLR * lrMultiplier
    // - base LR is restored after each epoch (no persistent mutation of NNInfo).
    {
        class CaptureMetricsCb : public glades::ITrainingCallbacks
        {
        public:
            std::vector<glades::NNetworkEpochMetrics> epochs;
            virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
            {
                epochs.push_back(m);
                return false;
            }
        };

        glades::NumberInput* di13 = new glades::NumberInput();
        di13->trainMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di13->trainExpectedMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di13->trainMatrix[0][0] = 1.0f;
        di13->trainExpectedMatrix[0][0] = 2.0f;
        di13->trainMatrix[1][0] = 2.0f;
        di13->trainExpectedMatrix[1][0] = 4.0f;

        const float baseLR = 0.1f;
        glades::InputLayerInfo* in13 = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ baseLR,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden13;
        glades::OutputLayerInfo* out13 = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info13 = new glades::NNInfo("ut_lr_schedule_exp", in13, hidden13, out13);

        glades::NNetwork net13(info13, glades::NNetwork::TYPE_DFF);
        net13.getTerminatorMutable().setEpoch(3);
        net13.getTerminatorMutable().setAccuracy(0);

        const float gamma = 0.5f;
        net13.setLearningRateScheduleExp(gamma);

        CaptureMetricsCb cb;
        const glades::NNetworkStatus st13 = net13.train(di13, &cb);
        G_assert(__FILE__, __LINE__,
                 "==============NN13-test::TrainStatus() Failed==============",
                 st13.ok());

        G_assert(__FILE__, __LINE__,
                 "==============NN13-test::EpochCount() Failed==============",
                 cb.epochs.size() == 3);

        const float tol = 1e-6f;
        if (cb.epochs.size() == 3)
        {
            const float m1 = cb.epochs[0].lrMultiplier;
            const float m2 = cb.epochs[1].lrMultiplier;
            const float m3 = cb.epochs[2].lrMultiplier;

            // exp schedule: multiplier = gamma^(epoch-1)
            G_assert(__FILE__, __LINE__,
                     "==============NN13-test::MultiplierEpoch1() Failed==============",
                     fabs(m1 - 1.0f) < tol);
            G_assert(__FILE__, __LINE__,
                     "==============NN13-test::MultiplierEpoch2() Failed==============",
                     fabs(m2 - gamma) < tol);
            G_assert(__FILE__, __LINE__,
                     "==============NN13-test::MultiplierEpoch3() Failed==============",
                     fabs(m3 - (gamma * gamma)) < tol);

            // Effective LR should track baseLR * multiplier (output transition index == 0)
            G_assert(__FILE__, __LINE__,
                     "==============NN13-test::EffectiveLR_Epoch1() Failed==============",
                     fabs(cb.epochs[0].learningRate - (baseLR * 1.0f)) < 1e-5f);
            G_assert(__FILE__, __LINE__,
                     "==============NN13-test::EffectiveLR_Epoch2() Failed==============",
                     fabs(cb.epochs[1].learningRate - (baseLR * gamma)) < 1e-5f);
            G_assert(__FILE__, __LINE__,
                     "==============NN13-test::EffectiveLR_Epoch3() Failed==============",
                     fabs(cb.epochs[2].learningRate - (baseLR * gamma * gamma)) < 1e-5f);
        }

        // Base LR must be restored after the run (no persistent scaling).
        G_assert(__FILE__, __LINE__,
                 "==============NN13-test::BaseLRRestored() Failed==============",
                 (net13.getNNInfo() != NULL) && fabs(net13.getNNInfo()->getLearningRate(0) - baseLR) < 1e-6f);

        delete di13;
        delete info13;
    }

    printf("-----------------------------------\n");
    printf("NN Test 14 (Global grad-norm clipping: DFF)\n");
    printf("-----------------------------------\n");

    // This test verifies global gradient-norm clipping for the DFF tensor path:
    // - When clipNorm is small and gradients are large, metrics.gradNormScale < 1.
    // - metrics.gradNorm reports the unclipped norm (before scaling).
    {
        class CaptureMetricsCb : public glades::ITrainingCallbacks
        {
        public:
            glades::NNetworkEpochMetrics last;
            bool saw;
            CaptureMetricsCb() : saw(false) {}
            virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
            {
                last = m;
                saw = true;
                return false;
            }
        };

        glades::NumberInput* di14 = new glades::NumberInput();
        di14->trainMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
        di14->trainExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
        di14->trainMatrix[0][0] = 1000.0f;        // huge input => huge gradient
        di14->trainExpectedMatrix[0][0] = 1000.0f; // huge target => large error if weights ~0

        glades::InputLayerInfo* in14 = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ 0.1f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden14;
        glades::OutputLayerInfo* out14 = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info14 = new glades::NNInfo("ut_global_grad_clip_dff", in14, hidden14, out14);

        glades::NNetwork net14(info14, glades::NNetwork::TYPE_DFF);
        net14.getTerminatorMutable().setEpoch(1);
        net14.getTerminatorMutable().setAccuracy(0);

        // Build so we can force deterministic initial weights.
        net14.graphMutable().build(info14, di14, glades::NNetwork::TYPE_DFF);
        net14.setMustdBuildMeat(false);

        glades::Layer* outLayer14 = net14.graphMutable().getOutputLayer(1);
        glades::Node* outNode14 = net14.graphMutable().getOutputNode(outLayer14, 0);
        if (outNode14)
        {
            // Weight edge (idx 0) = 0, bias edge (idx 1) = 0
            outNode14->setEdgeWeight(0, 0.0f);
            outNode14->setEdgeWeight(1, 0.0f);
        }

        const float clipNorm = 10.0f;
        net14.setGlobalGradClipNorm(clipNorm);

        CaptureMetricsCb cb;
        const glades::NNetworkStatus st14 = net14.train(di14, &cb);
        G_assert(__FILE__, __LINE__,
                 "==============NN14-test::TrainStatus() Failed==============",
                 st14.ok());
        G_assert(__FILE__, __LINE__,
                 "==============NN14-test::SawMetrics() Failed==============",
                 cb.saw);

        // The clip should trigger: large gradients => scale < 1 and norm > clipNorm.
        if (cb.saw)
        {
            printf("[UT] Grad clip: norm=%g scale=%g clip=%g\n", cb.last.gradNorm, cb.last.gradNormScale, clipNorm);
            G_assert(__FILE__, __LINE__,
                     "==============NN14-test::GradNormReported() Failed==============",
                     cb.last.gradNorm > clipNorm);
            G_assert(__FILE__, __LINE__,
                     "==============NN14-test::GradNormScaleClips() Failed==============",
                     cb.last.gradNormScale > 0.0f && cb.last.gradNormScale < 1.0f);
        }

        delete di14;
        delete info14;
    }

    printf("-----------------------------------\n");
    printf("NN Test 15 (LR schedule: step decay)\n");
    printf("-----------------------------------\n");

    // Step schedule: multiplier = gamma^floor((epoch-1)/stepSize)
    // We validate multipliers over 4 epochs with stepSize=2.
    {
        class CaptureMetricsCb : public glades::ITrainingCallbacks
        {
        public:
            std::vector<glades::NNetworkEpochMetrics> epochs;
            virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
            {
                epochs.push_back(m);
                return false;
            }
        };

        glades::NumberInput* di15 = new glades::NumberInput();
        di15->trainMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
        di15->trainExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
        di15->trainMatrix[0][0] = 1.0f;
        di15->trainExpectedMatrix[0][0] = 1.0f;

        const float baseLR = 0.01f;
        glades::InputLayerInfo* in15 = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ baseLR,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden15;
        glades::OutputLayerInfo* out15 = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info15 = new glades::NNInfo("ut_lr_schedule_step", in15, hidden15, out15);

        glades::NNetwork net15(info15, glades::NNetwork::TYPE_DFF);
        net15.getTerminatorMutable().setEpoch(4);
        net15.getTerminatorMutable().setAccuracy(0);

        const int stepSize = 2;
        const float gamma = 0.5f;
        net15.setLearningRateScheduleStep(stepSize, gamma);

        CaptureMetricsCb cb;
        const glades::NNetworkStatus st15 = net15.train(di15, &cb);
        G_assert(__FILE__, __LINE__, "==============NN15-test::TrainStatus() Failed==============", st15.ok());
        G_assert(__FILE__, __LINE__, "==============NN15-test::EpochCount() Failed==============", cb.epochs.size() == 4);

        const float tol = 1e-6f;
        if (cb.epochs.size() == 4)
        {
            // epoch 1 => k=0 => 1
            // epoch 2 => k=0 => 1
            // epoch 3 => k=1 => gamma
            // epoch 4 => k=1 => gamma
            G_assert(__FILE__, __LINE__, "==============NN15-test::Mult1() Failed==============", fabs(cb.epochs[0].lrMultiplier - 1.0f) < tol);
            G_assert(__FILE__, __LINE__, "==============NN15-test::Mult2() Failed==============", fabs(cb.epochs[1].lrMultiplier - 1.0f) < tol);
            G_assert(__FILE__, __LINE__, "==============NN15-test::Mult3() Failed==============", fabs(cb.epochs[2].lrMultiplier - gamma) < tol);
            G_assert(__FILE__, __LINE__, "==============NN15-test::Mult4() Failed==============", fabs(cb.epochs[3].lrMultiplier - gamma) < tol);

            // Effective output LR should track baseLR * multiplier.
            G_assert(__FILE__, __LINE__, "==============NN15-test::LR1() Failed==============", fabs(cb.epochs[0].learningRate - (baseLR * 1.0f)) < 1e-5f);
            G_assert(__FILE__, __LINE__, "==============NN15-test::LR2() Failed==============", fabs(cb.epochs[1].learningRate - (baseLR * 1.0f)) < 1e-5f);
            G_assert(__FILE__, __LINE__, "==============NN15-test::LR3() Failed==============", fabs(cb.epochs[2].learningRate - (baseLR * gamma)) < 1e-5f);
            G_assert(__FILE__, __LINE__, "==============NN15-test::LR4() Failed==============", fabs(cb.epochs[3].learningRate - (baseLR * gamma)) < 1e-5f);
        }

        delete di15;
        delete info15;
    }

    printf("-----------------------------------\n");
    printf("NN Test 16 (LR schedule: cosine)\n");
    printf("-----------------------------------\n");

    // Cosine schedule: multiplier = min + 0.5*(1-min)*(1+cos(pi*t/T))
    // with t = min(epochFromStart, T). We validate epoch1 (t=0 => 1) and epoch(T+1) (t=T => min).
    {
        class CaptureMetricsCb : public glades::ITrainingCallbacks
        {
        public:
            std::vector<glades::NNetworkEpochMetrics> epochs;
            virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
            {
                epochs.push_back(m);
                return false;
            }
        };

        glades::NumberInput* di16 = new glades::NumberInput();
        di16->trainMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
        di16->trainExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
        di16->trainMatrix[0][0] = 1.0f;
        di16->trainExpectedMatrix[0][0] = 1.0f;

        const float baseLR = 0.02f;
        glades::InputLayerInfo* in16 = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ baseLR,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden16;
        glades::OutputLayerInfo* out16 = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info16 = new glades::NNInfo("ut_lr_schedule_cosine", in16, hidden16, out16);

        glades::NNetwork net16(info16, glades::NNetwork::TYPE_DFF);
        const int tMax = 2; // so epoch3 has t=2 => minMultiplier
        const float minMult = 0.1f;
        net16.setLearningRateScheduleCosine(tMax, minMult);
        net16.getTerminatorMutable().setEpoch(3);
        net16.getTerminatorMutable().setAccuracy(0);

        CaptureMetricsCb cb;
        const glades::NNetworkStatus st16 = net16.train(di16, &cb);
        G_assert(__FILE__, __LINE__, "==============NN16-test::TrainStatus() Failed==============", st16.ok());
        G_assert(__FILE__, __LINE__, "==============NN16-test::EpochCount() Failed==============", cb.epochs.size() == 3);

        const float tol = 1e-5f;
        if (cb.epochs.size() == 3)
        {
            // epoch1 (t=0): multiplier == 1
            G_assert(__FILE__, __LINE__, "==============NN16-test::Mult1() Failed==============", fabs(cb.epochs[0].lrMultiplier - 1.0f) < tol);
            // epoch3 (t=T): multiplier == minMult
            G_assert(__FILE__, __LINE__, "==============NN16-test::Mult3() Failed==============", fabs(cb.epochs[2].lrMultiplier - minMult) < tol);

            G_assert(__FILE__, __LINE__, "==============NN16-test::LR1() Failed==============", fabs(cb.epochs[0].learningRate - (baseLR * 1.0f)) < tol);
            G_assert(__FILE__, __LINE__, "==============NN16-test::LR3() Failed==============", fabs(cb.epochs[2].learningRate - (baseLR * minMult)) < tol);
        }

        delete di16;
        delete info16;
    }

    printf("-----------------------------------\n");
    printf("NN Test 17 (Regression metrics: MSE/MAE/RMSE values)\n");
    printf("-----------------------------------\n");

    // This test validates the computed regression metrics fields:
    // - totalError == MSE
    // - regMAE == MAE
    // - regRMSE == sqrt(MSE)
    //
    // We use RUN_TEST to avoid weight updates; we set deterministic weights.
    {
        class CaptureMetricsCb : public glades::ITrainingCallbacks
        {
        public:
            glades::NNetworkEpochMetrics last;
            bool saw;
            CaptureMetricsCb() : saw(false) {}
            virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
            {
                last = m;
                saw = true;
                return false;
            }
        };

        glades::NumberInput* di17 = new glades::NumberInput();
        di17->trainMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di17->trainExpectedMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di17->trainMatrix[0][0] = 1.0f;
        di17->trainExpectedMatrix[0][0] = 2.0f;
        di17->trainMatrix[1][0] = 2.0f;
        di17->trainExpectedMatrix[1][0] = 4.0f;

        // Mirror train->test so test() evaluates the same rows.
        di17->testMatrix = di17->trainMatrix;
        di17->testExpectedMatrix = di17->trainExpectedMatrix;

        glades::InputLayerInfo* in17 = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ 0.0f, // irrelevant for test
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden17;
        glades::OutputLayerInfo* out17 = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info17 = new glades::NNInfo("ut_reg_metrics", in17, hidden17, out17);

        glades::NNetwork net17(info17, glades::NNetwork::TYPE_DFF);
        // Pre-build to set deterministic weights.
        net17.graphMutable().build(info17, di17, glades::NNetwork::TYPE_DFF);
        net17.setMustdBuildMeat(false);

        glades::Layer* outLayer17 = net17.graphMutable().getOutputLayer(1);
        glades::Node* outNode17 = net17.graphMutable().getOutputNode(outLayer17, 0);
        if (outNode17)
        {
            // Set y = 1*x + 0, so predictions are [1,2] vs targets [2,4].
            outNode17->setEdgeWeight(0, 1.0f);
            outNode17->setEdgeWeight(1, 0.0f); // bias edge
        }

        CaptureMetricsCb cb;
        const glades::NNetworkStatus st17 = net17.test(di17, &cb);
        G_assert(__FILE__, __LINE__, "==============NN17-test::TestStatus() Failed==============", st17.ok());
        G_assert(__FILE__, __LINE__, "==============NN17-test::SawMetrics() Failed==============", cb.saw);

        // Expected:
        // errors = [1,2]
        // MSE = (1^2 + 2^2)/2 = 2.5
        // MAE = (1+2)/2 = 1.5
        // RMSE = sqrt(2.5)
        const float expMSE = 2.5f;
        const float expMAE = 1.5f;
        const float expRMSE = static_cast<float>(sqrt(2.5));
        const float tol = 1e-4f;

        if (cb.saw)
        {
            printf("[UT] Reg metrics: MSE=%f MAE=%f RMSE=%f\n", cb.last.totalError, cb.last.regMAE, cb.last.regRMSE);
            G_assert(__FILE__, __LINE__, "==============NN17-test::MSE() Failed==============", fabs(cb.last.totalError - expMSE) < tol);
            G_assert(__FILE__, __LINE__, "==============NN17-test::MAE() Failed==============", fabs(cb.last.regMAE - expMAE) < tol);
            G_assert(__FILE__, __LINE__, "==============NN17-test::RMSE() Failed==============", fabs(cb.last.regRMSE - expRMSE) < tol);
        }

        delete di17;
        delete info17;
    }

    printf("-----------------------------------\n");
    printf("NN Test 18 (Global grad-norm clip disabled => scale=1)\n");
    printf("-----------------------------------\n");

    // With grad clip disabled (default), gradNorm stays 0 and scale stays 1 in metrics.
    {
        class CaptureMetricsCb : public glades::ITrainingCallbacks
        {
        public:
            glades::NNetworkEpochMetrics last;
            bool saw;
            CaptureMetricsCb() : saw(false) {}
            virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
            {
                last = m;
                saw = true;
                return false;
            }
        };

        glades::NumberInput* di18 = new glades::NumberInput();
        di18->trainMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
        di18->trainExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
        di18->trainMatrix[0][0] = 10.0f;
        di18->trainExpectedMatrix[0][0] = 0.0f;

        glades::InputLayerInfo* in18 = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ 0.1f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden18;
        glades::OutputLayerInfo* out18 = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info18 = new glades::NNInfo("ut_grad_clip_disabled", in18, hidden18, out18);

        glades::NNetwork net18(info18, glades::NNetwork::TYPE_DFF);
        net18.getTerminatorMutable().setEpoch(1);
        net18.getTerminatorMutable().setAccuracy(0);

        CaptureMetricsCb cb;
        const glades::NNetworkStatus st18 = net18.train(di18, &cb);
        G_assert(__FILE__, __LINE__, "==============NN18-test::TrainStatus() Failed==============", st18.ok());
        G_assert(__FILE__, __LINE__, "==============NN18-test::SawMetrics() Failed==============", cb.saw);
        if (cb.saw)
        {
            G_assert(__FILE__, __LINE__, "==============NN18-test::GradScaleIsOne() Failed==============", fabs(cb.last.gradNormScale - 1.0f) < 1e-6f);
            G_assert(__FILE__, __LINE__, "==============NN18-test::GradNormZero() Failed==============", fabs(cb.last.gradNorm - 0.0f) < 1e-6f);
        }

        delete di18;
        delete info18;
    }

    printf("\n============================================================\n");
}

void NNRecurrentUnitTest()
{
    printf("============================================================\n");
    printf("NN Recurrent Test Suite (RNN/GRU/LSTM)\n");
    printf("============================================================\n");

    printf("-----------------------------------\n");
    printf("NN Test 7 (RNN full BPTT: future loss updates earlier Wx)\n");
    printf("-----------------------------------\n");

    // Copied from NNUnitTest(): see NN Test 7 block.
    {
        glades::NumberInput* di7 = new glades::NumberInput();
        di7->trainMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di7->trainExpectedMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di7->trainMatrix[0][0] = 1.0f;  // x1
        di7->trainMatrix[1][0] = 0.0f;  // x2
        di7->trainExpectedMatrix[0][0] = 1.0f; // y1 target (no loss at t=1)
        di7->trainExpectedMatrix[1][0] = 0.0f; // y2 target (loss at t=2)

        glades::InputLayerInfo* in7 = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ 0.1f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden7;
        hidden7.push_back(new glades::HiddenLayerInfo(
            /*size*/ 1,
            /*learningRate*/ 0.1f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        ));
        glades::OutputLayerInfo* out7 = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info7 = new glades::NNInfo("ut_rnn_bptt", in7, hidden7, out7);

        glades::NNetwork rnnNet7(info7, glades::NNetwork::TYPE_RNN);
        rnnNet7.getTerminatorMutable().setEpoch(1);
        rnnNet7.getTerminatorMutable().setAccuracy(0);

        rnnNet7.graphMutable().build(info7, di7, glades::NNetwork::TYPE_RNN);
        rnnNet7.setMustdBuildMeat(false);

        glades::Layer* hiddenLayer7 = rnnNet7.graphMutable().getOutputLayer(1);
        glades::Layer* outLayer7 = rnnNet7.graphMutable().getOutputLayer(2);
        glades::Node* hiddenNode7 = rnnNet7.graphMutable().getOutputNode(hiddenLayer7, 0);
        glades::Node* outNode7 = rnnNet7.graphMutable().getOutputNode(outLayer7, 0);

        hiddenLayer7->setBiasWeight(0.0f);
        outLayer7->setBiasWeight(0.0f);

        // Wx=1, Wy=1
        hiddenNode7->setEdgeWeight(0, 1.0f);
        outNode7->setEdgeWeight(0, 1.0f);

        // Wh=1 and ensure context resets from a non-zero starting value
        glades::Node* ctx7 = hiddenNode7->getContextNode();
        G_assert(__FILE__, __LINE__,
                 "==============NN7-test::ContextNodeMissing() Failed==============",
                 ctx7 != NULL);
        if (ctx7)
        {
            ctx7->setEdgeWeight(0, 1.0f); // Wh
            ctx7->setWeight(10.0f);       // should be reset to 0 at run start
        }

        // Train exactly 1 epoch
        const glades::NNetworkStatus stTrain7 = rnnNet7.train(di7);
        G_assert(__FILE__, __LINE__,
                 "==============NN7-test::TrainStatus() Failed==============",
                 stTrain7.ok());

        rnnNet7.materializeGraphParameters();

        const float wxFinal = hiddenNode7->getEdgeWeight(0);
        const float expectedWx = 0.9f;
        const float tol = 1e-3f;
        printf("[UT] RNN(BPTT) Wx final=%f (expected ~%f)\n", wxFinal, expectedWx);
        G_assert(__FILE__, __LINE__,
                 "==============NN7-test::RNNBPTT_UpdatesWxFromFutureLoss() Failed==============",
                 (wxFinal > expectedWx - tol) && (wxFinal < expectedWx + tol));

        delete di7;
        delete info7;
    }

    printf("-----------------------------------\n");
    printf("NN Test 8 (RNN train on rnn.csv with BPTT)\n");
    printf("-----------------------------------\n");

    // Copied from NNUnitTest(): see NN Test 8 block.
    {
        shmea::GTable raw("datasets/rnn.csv", ',', shmea::GTable::TYPE_FILE);
        raw.clearOutputs();
        raw.toggleOutput(2); // z

        glades::NumberInput* di8 = new glades::NumberInput();
        di8->import(raw, /*standardizeFlag*/ glades::GMath::MINMAX);

        di8->testMatrix = di8->trainMatrix;
        di8->testExpectedMatrix = di8->trainExpectedMatrix;

        const float lr = 0.05f;
        glades::InputLayerInfo* in8 = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ lr,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden8;
        hidden8.push_back(new glades::HiddenLayerInfo(
            /*size*/ 1,
            /*learningRate*/ lr,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        ));
        glades::OutputLayerInfo* out8 = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info8 = new glades::NNInfo("ut_rnn_csv_train_bptt", in8, hidden8, out8);

        glades::NNetwork rnnNet8(info8, glades::NNetwork::TYPE_RNN);
        rnnNet8.getTerminatorMutable().setEpoch(500);

        rnnNet8.graphMutable().build(info8, di8, glades::NNetwork::TYPE_RNN);
        rnnNet8.setMustdBuildMeat(false);

        glades::Layer* hiddenLayer8 = rnnNet8.graphMutable().getOutputLayer(1);
        glades::Layer* outLayer8 = rnnNet8.graphMutable().getOutputLayer(2);
        glades::Node* hiddenNode8 = rnnNet8.graphMutable().getOutputNode(hiddenLayer8, 0);
        glades::Node* outNode8 = rnnNet8.graphMutable().getOutputNode(outLayer8, 0);

        hiddenLayer8->setBiasWeight(0.0f);
        outLayer8->setBiasWeight(0.0f);

        hiddenNode8->setEdgeWeight(0, 0.0f); // Wx_x
        hiddenNode8->setEdgeWeight(1, 1.0f); // Wx_y
        outNode8->setEdgeWeight(0, 0.8f);    // Wy

        glades::Node* ctx8 = hiddenNode8->getContextNode();
        G_assert(__FILE__, __LINE__,
                 "==============NN8-test::ContextNodeMissing() Failed==============",
                 ctx8 != NULL);
        if (ctx8)
        {
            ctx8->setEdgeWeight(0, 0.0f); // Wh initialized to 0
            ctx8->setWeight(0.0f);
        }

        const glades::NNetworkStatus stTrain8 = rnnNet8.train(di8);
        G_assert(__FILE__, __LINE__,
                 "==============NN8-test::TrainStatus() Failed==============",
                 stTrain8.ok());

        // Evaluate on the mirrored test set
        const glades::NNetworkStatus stTest8 = rnnNet8.test(di8);
        G_assert(__FILE__, __LINE__,
                 "==============NN8-test::TestStatus() Failed==============",
                 stTest8.ok());
        G_assert(__FILE__, __LINE__,
                 "==============NN8-test::Accuracy() Failed==============",
                 rnnNet8.getAccuracy() > 99.0f);

        delete di8;
        delete info8;
    }

    printf("-----------------------------------\n");
    printf("NN Test 10 (LSTM deterministic forward)\n");
    printf("-----------------------------------\n");

    // Copied from NNUnitTest(): see NN Test 10 block.
    {
        glades::NumberInput* di10 = new glades::NumberInput();
        di10->trainMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di10->trainExpectedMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di10->trainMatrix[0][0] = 0.0f;
        di10->trainMatrix[1][0] = 1.0f;

        di10->testMatrix = di10->trainMatrix;
        di10->testExpectedMatrix = di10->trainExpectedMatrix;

        glades::InputLayerInfo* in10 = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden10;
        hidden10.push_back(new glades::HiddenLayerInfo(
            /*size*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        ));
        glades::OutputLayerInfo* out10 = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info10 = new glades::NNInfo("ut_lstm_forward", in10, hidden10, out10);

        glades::NNetwork lstmNet(info10, glades::NNetwork::TYPE_LSTM);
        lstmNet.getTerminatorMutable().setEpoch(1);
        lstmNet.getTerminatorMutable().setAccuracy(0);

        lstmNet.graphMutable().build(info10, di10, glades::NNetwork::TYPE_LSTM);
        lstmNet.setMustdBuildMeat(false);

        glades::Layer* hiddenLayer10 = lstmNet.graphMutable().getOutputLayer(1);
        glades::Layer* outLayer10 = lstmNet.graphMutable().getOutputLayer(2);
        glades::Node* hNode10 = lstmNet.graphMutable().getOutputNode(hiddenLayer10, 0);
        glades::Node* yNode10 = lstmNet.graphMutable().getOutputNode(outLayer10, 0);

        G_assert(__FILE__, __LINE__,
                 "==============NN10-test::NodesMissing() Failed==============",
                 (hiddenLayer10 != NULL) && (outLayer10 != NULL) && (hNode10 != NULL) && (yNode10 != NULL));

        glades::Node* ctx10 = (hNode10 ? hNode10->getContextNode() : NULL);
        G_assert(__FILE__, __LINE__,
                 "==============NN10-test::ContextNodeMissing() Failed==============",
                 ctx10 != NULL);

        if (hNode10 && yNode10 && ctx10)
        {
            hNode10->setEdgeWeight(0, 0.0f);   // Wi
            hNode10->setEdgeWeight(1, 20.0f);  // bi
            hNode10->setEdgeWeight(2, 0.0f);   // Wf
            hNode10->setEdgeWeight(3, -20.0f); // bf
            hNode10->setEdgeWeight(4, 0.0f);   // Wo
            hNode10->setEdgeWeight(5, 20.0f);  // bo
            hNode10->setEdgeWeight(6, 1.0f);   // Wg
            hNode10->setEdgeWeight(7, 0.0f);   // bg

            ctx10->setEdgeWeight(0, 0.0f);
            ctx10->setEdgeWeight(1, 0.0f);
            ctx10->setEdgeWeight(2, 0.0f);
            ctx10->setEdgeWeight(3, 0.0f);

            yNode10->setEdgeWeight(0, 1.0f);
            yNode10->setEdgeWeight(1, 0.0f); // bias edge
        }

        const glades::NNetworkStatus st10 = lstmNet.test(di10);
        G_assert(__FILE__, __LINE__,
                 "==============NN10-test::TestStatus() Failed==============",
                 st10.ok());

        const float expected = static_cast<float>(tanh(tanh(1.0)));
        const float y2 = (yNode10 ? yNode10->getWeight() : 0.0f);
        const float c2 = (hNode10 ? hNode10->getCellState() : 0.0f);
        const float expectedC = static_cast<float>(tanh(1.0));
        const float tol = 1e-3f;
        printf("[UT] LSTM y2=%f c2=%f (expected y~%f c~%f)\n", y2, c2, expected, expectedC);
        G_assert(__FILE__, __LINE__,
                 "==============NN10-test::LSTMForward() Failed==============",
                 (y2 > expected - tol) && (y2 < expected + tol));
        G_assert(__FILE__, __LINE__,
                 "==============NN10-test::LSTMCellState() Failed==============",
                 (c2 > expectedC - tol) && (c2 < expectedC + tol));

        delete di10;
        delete info10;
    }

    printf("-----------------------------------\n");
    printf("NN Test 11 (GRU BPTT: future loss updates candidate Wx)\n");
    printf("-----------------------------------\n");

    // Copied from NNUnitTest(): see NN Test 11 block.
    {
        glades::NumberInput* di11 = new glades::NumberInput();
        di11->trainMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di11->trainExpectedMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di11->trainMatrix[0][0] = 1.0f; // x1
        di11->trainMatrix[1][0] = 0.0f; // x2

        const float a = static_cast<float>(tanh(1.0));
        di11->trainExpectedMatrix[0][0] = a;
        di11->trainExpectedMatrix[1][0] = 0.0f;

        const float lr = 0.1f;
        glades::InputLayerInfo* in11 = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ lr,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<glades::HiddenLayerInfo*> hidden11;
        hidden11.push_back(new glades::HiddenLayerInfo(
            /*size*/ 1,
            /*learningRate*/ lr,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        ));
        glades::OutputLayerInfo* out11 = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info11 = new glades::NNInfo("ut_gru_bptt", in11, hidden11, out11);

        glades::NNetwork gruNet11(info11, glades::NNetwork::TYPE_GRU);
        gruNet11.getTerminatorMutable().setEpoch(1);
        gruNet11.getTerminatorMutable().setAccuracy(0);

        gruNet11.graphMutable().build(info11, di11, glades::NNetwork::TYPE_GRU);
        gruNet11.setMustdBuildMeat(false);

        glades::Layer* hiddenLayer11 = gruNet11.graphMutable().getOutputLayer(1);
        glades::Layer* outLayer11 = gruNet11.graphMutable().getOutputLayer(2);
        glades::Node* hNode11 = gruNet11.graphMutable().getOutputNode(hiddenLayer11, 0);
        glades::Node* yNode11 = gruNet11.graphMutable().getOutputNode(outLayer11, 0);
        glades::Node* ctx11 = (hNode11 ? hNode11->getContextNode() : NULL);

        G_assert(__FILE__, __LINE__,
                 "==============NN11-test::NodesMissing() Failed==============",
                 (hiddenLayer11 != NULL) && (outLayer11 != NULL) && (hNode11 != NULL) && (yNode11 != NULL) && (ctx11 != NULL));

        if (hNode11 && yNode11 && ctx11)
        {
            hNode11->setEdgeWeight(0, 0.0f);  // Wz
            hNode11->setEdgeWeight(1, 20.0f); // bz => z ~= 1
            hNode11->setEdgeWeight(2, 0.0f);  // Wr
            hNode11->setEdgeWeight(3, 20.0f); // br => r ~= 1
            hNode11->setEdgeWeight(4, 1.0f);  // Wh (candidate input weight)
            hNode11->setEdgeWeight(5, 0.0f);  // bh

            ctx11->setEdgeWeight(0, 0.0f);
            ctx11->setEdgeWeight(1, 0.0f);
            ctx11->setEdgeWeight(2, 1.0f); // Uh = 1
            ctx11->setWeight(0.0f);

            yNode11->setEdgeWeight(0, 1.0f);
            yNode11->setEdgeWeight(1, 0.0f); // bias edge
        }

        const glades::NNetworkStatus st11 = gruNet11.train(di11);
        G_assert(__FILE__, __LINE__,
                 "==============NN11-test::TrainStatus() Failed==============",
                 st11.ok());

        gruNet11.materializeGraphParameters();

        const float b = static_cast<float>(tanh(static_cast<double>(a)));
        const float deltaY2 = 2.0f * (b - 0.0f);
        const float daH2 = deltaY2 * (1.0f - (b * b));
        const float daH1 = daH2 * (1.0f - (a * a));
        const float expectedWx = 1.0f - (lr * daH1 / 2.0f);

        const float wxFinal = (hNode11 ? hNode11->getEdgeWeight(4) : 0.0f);
        const float tol = 2e-3f;
        printf("[UT] GRU(BPTT) Wx final=%f (expected ~%f)\n", wxFinal, expectedWx);
        G_assert(__FILE__, __LINE__,
                 "==============NN11-test::GRUBPTT_UpdatesWxFromFutureLoss() Failed==============",
                 (wxFinal > expectedWx - tol) && (wxFinal < expectedWx + tol));

        delete di11;
        delete info11;
    }

    printf("\n============================================================\n");
}
