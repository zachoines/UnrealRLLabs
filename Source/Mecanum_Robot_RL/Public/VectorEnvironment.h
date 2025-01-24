#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BaseEnvironment.h"
#include "RLTypes.h"
#include "VectorEnvironment.generated.h"

// Forward declare environment config
class UEnvironmentConfig;

/**
 * AVectorEnvironment spawns and manages multiple environment instances.
 * It also provides a unified interface for stepping, sampling random actions, etc.
 */
UCLASS()
class UNREALRLLABS_API AVectorEnvironment : public AActor
{
    GENERATED_BODY()

public:
    // Sets default values for this actor's properties
    AVectorEnvironment();

    // Initializes the environment with 'n' number of BaseEnvironments
    void InitEnv(TSubclassOf<ABaseEnvironment> EnvironmentClass, TArray<FBaseInitParams*> ParamsArray);

    // Reset the environment and return the initial states
    TArray<FState> ResetEnv(int NumAgents);

    /*
        Perform a step in each environment using the given actions:
          returns { Done, Trunc, Reward, Action, State, NextState } for last step
    */
    TTuple<TArray<float>, TArray<float>, TArray<float>, TArray<FAction>, TArray<FState>, TArray<FState>> Transition();

    // Pass actions to each environment to do a single "step" (frame)
    void Step(TArray<FAction> Actions);

    // Randomly sample actions for each environment (or each agent).
    TArray<FAction> SampleActions();

    // Get the current states for each environment
    TArray<FState> GetStates();

    // Current number of agents (for multi-agent environments)
    int CurrentAgents;

protected:
    virtual void BeginPlay() override;

private:
    // The environment instances we manage
    TArray<ABaseEnvironment*> Environments;

    // The last actions we applied
    TArray<FAction> LastActions;

    // The last states before stepping
    TArray<FState> LastStates;

    // The current states after stepping
    TArray<FState> CurrentStates;

    // Current done/trunc flags
    TArray<float> CurrentDones;
    TArray<float> CurrentTruncs;

    // Reference to the environment config (for action space, etc.)
    UPROPERTY()
    UEnvironmentConfig* EnvConfig;

    // Arrays describing the discrete & continuous action specs from the config
    TArray<int32> DiscreteActionSizes;
    TArray<FVector2D> ContinuousActionRanges;

private:
    /**
     * Parse the action space (e.g. "environment/shape/action/agent/...") from EnvConfig
     * and store in DiscreteActionSizes / ContinuousActionRanges.
     */
    void ParseActionSpaceFromConfig();

    /**
     * Sample random actions for a single environment (which has CurrentAgents agents).
     * This uses DiscreteActionSizes and ContinuousActionRanges as parsed from the config.
     */
    FAction EnvSample();
};
