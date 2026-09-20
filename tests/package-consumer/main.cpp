#include "Backend/Machine Learning/main.h"
#include "Backend/Machine Learning/Networks/network.h"
#include "Backend/Machine Learning/Networks/training_callbacks.h"
#include "Backend/Machine Learning/DataObjects/NumberInput.h"
#include "Backend/Machine Learning/Structure/nninfo.h"
#include "Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "Backend/Machine Learning/Structure/outputlayerinfo.h"
#include "Backend/Machine Learning/GMath/gmath.h"
#include "Backend/Machine Learning/State/Terminator.h"
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>

static void require(bool ok, const std::string& message)
{
    if (!ok) throw std::runtime_error(message);
}
static void checked(const glades::NNetworkStatus& status) { require(status.ok(), status.message); }
static float predict(glades::NNetwork& net, glades::NumberInput& data)
{
    glades::ITrainingCallbacks quiet;
    checked(net.test(&data, &quiet));
    const shmea::GList result = net.getResults();
    require(result.size() == 2u, "prediction shape");
    const float value = result.getFloat(1);
    require(value == value && std::fabs(value) < 100.0f, "nonfinite/unbounded prediction");
    return value;
}
int main(int argc, char** argv) try
{
    require(argc == 3, "create NAME | read DIRECTORY | reject DIRECTORY | wrong-shape DIRECTORY");
    const std::string mode = argv[1];
    glades::NumberInput data;
    data.trainMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.25f));
    data.trainExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.75f));
    data.testMatrix = data.trainMatrix;
    data.testExpectedMatrix = data.trainExpectedMatrix;
    if (mode == "create")
    {
        shmea::GPointer<glades::InputLayerInfo> input(new glades::InputLayerInfo(1, .05f, 0, 0, 0, 0, glades::GMath::LINEAR, 1));
        std::vector<shmea::GPointer<glades::HiddenLayerInfo> > hidden;
        hidden.push_back(shmea::GPointer<glades::HiddenLayerInfo>(new glades::HiddenLayerInfo(3, .05f, 0, 0, 0, 0, glades::GMath::TANH, 1)));
        shmea::GPointer<glades::OutputLayerInfo> output(new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION));
        glades::NNInfo info("consumer", input, hidden, output);
        glades::NNetwork net(&info, glades::NNetwork::TYPE_DFF);
        net.setSeed(12345u);
        net.getTerminatorMutable().setEpoch(4);
        net.getTerminatorMutable().setAccuracy(0);
        glades::ITrainingCallbacks quiet;
        checked(net.train(&data, &quiet));
        checked(net.saveModel(argv[2]));
        glades::NNetwork named;
        checked(named.loadModel(argv[2], &data));
        require(predict(net, data) == predict(named, data), "named API prediction changed");
        std::cout << "PRED " << std::setprecision(9) << predict(net, data) << '\n';
        return 0;
    }
    glades::NNetwork net;
    if (mode == "wrong-shape")
    {
        data.trainMatrix[0].push_back(1.f);
        require(!net.loadModelDirectory(argv[2], &data).ok(), "accepted wrong DFF feature shape");
        return 0;
    }
    const glades::NNetworkStatus status = net.loadModelDirectory(argv[2], &data);
    if (mode == "reject")
    {
        require(!status.ok(), "accepted invalid package");
        std::cout << "REJECT " << status.message << '\n';
        return 0;
    }
    require(mode == "read", "unknown mode");
    checked(status);
    require(net.getSeed() == 12345u && net.getEpochs() == 4, "metadata roundtrip");
    const float first = predict(net, data);
    for (int i = 0; i < 100; ++i) require(first == predict(net, data), "nondeterministic inference");
    require(net.getEpochs() == 4 && net.getSeed() == 12345u, "evaluation mutated training state");
    const std::string nulPath = std::string(argv[2]) + std::string(1, '\0') + "suffix";
    require(!net.loadModelDirectory(nulPath, &data).ok(), "accepted embedded NUL path");
    require(!net.loadModelDirectory(argv[2], NULL).ok(), "accepted null shape");
    std::cout << "PRED " << std::setprecision(9) << first << '\n';
    return 0;
}
catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
