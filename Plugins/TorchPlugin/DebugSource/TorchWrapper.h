#pragma once
#include "Agent.h"
#include "ITorchWrapper.h"

class TorchWrapper : public ITorchWrapper {
public:
    TorchWrapper();
    ~TorchWrapper();

    bool Init(int state_size, int action_size) override;
    std::vector<std::vector<float>> GetActions(const std::vector<std::vector<float>>& states) override;
    void Train() override;

private:
    Agent agent;
};