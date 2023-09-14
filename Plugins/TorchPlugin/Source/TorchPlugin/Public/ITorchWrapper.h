#pragma once
#include <vector>

class ITorchWrapper {
public:
    virtual ~ITorchWrapper() = default;
    virtual bool Init(int state_size, int action_size) = 0;
    virtual std::vector<std::vector<float>> GetActions(const std::vector<std::vector<float>>& states) = 0;
    virtual void Train() = 0;
};
