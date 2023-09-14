#pragma once

#include <torch/torch.h>
#include <vector>
#include <iostream>

class Agent {
public:
    Agent();
    ~Agent();

    bool Init(int state_size, int action_size);
    std::vector<std::vector<float>> GetActions(const std::vector<std::vector<float>>& states);
    void Train();

private:
    int state_size;
    int action_size;
    // Add any other necessary member variables here
};
