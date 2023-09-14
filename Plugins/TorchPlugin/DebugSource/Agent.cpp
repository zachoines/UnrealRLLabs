#include "Agent.h"

Agent::Agent() : state_size(0), action_size(0) {}

Agent::~Agent() {}

bool Agent::Init(int state_size, int action_size) {
    this->state_size = state_size;
    this->action_size = action_size;
    // Initialize the Reinforce agent here
    return true;
}

std::vector<std::vector<float>> Agent::GetActions(const std::vector<std::vector<float>>& states) {
    std::vector<std::vector<float>> actions;
    for (const auto& state : states) {
        std::vector<float> action(action_size, 0.5);  // Dummy action, replace with actual logic
        actions.push_back(action);
    }
    return actions;
}

void Agent::Train() {
    // Perform some basic torch training operations to test if torch is working on GPU
    struct Net : torch::nn::Module {
        Net() : fc1(1, 64), fc2(64, 1) {
            register_module("fc1", fc1);
            register_module("fc2", fc2);
        }

        torch::Tensor forward(torch::Tensor x) {
            x = torch::relu(fc1->forward(x));
            x = fc2->forward(x);
            return x;
        }

        torch::nn::Linear fc1, fc2;
    };

    Net net;

    // Create dummy data for training
    auto x_train = torch::linspace(1, 10, 10).view({-1, 1});
    auto y_train = torch::linspace(10, 1, 10).view({-1, 1});

    // Train the network
    torch::optim::SGD optimizer(net.parameters(), 0.01);

    for (int epoch = 1; epoch <= 100; ++epoch) {
        auto y_pred = net.forward(x_train);
        auto loss = torch::mse_loss(y_pred, y_train);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        std::cout << "Epoch: " << epoch << ", Loss: " << loss.item<float>() << std::endl;
    }
}
