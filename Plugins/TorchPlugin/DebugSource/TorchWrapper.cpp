#include "TorchWrapper.h"

TorchWrapper::TorchWrapper() {}

TorchWrapper::~TorchWrapper() {}

bool TorchWrapper::Init(int state_size, int action_size) {
    return agent.Init(state_size, action_size);
}

std::vector<std::vector<float>> TorchWrapper::GetActions(const std::vector<std::vector<float>>& states) {
    return agent.GetActions(states);
}

void TorchWrapper::Train() {
    agent.Train();
}

extern "C" {
    __declspec(dllexport) ITorchWrapper* CreateTorchWrapperInstance() {
        return new TorchWrapper();
    }

    __declspec(dllexport) void DestroyTorchWrapperInstance(ITorchWrapper* instance) {
        delete instance;
    }
}
