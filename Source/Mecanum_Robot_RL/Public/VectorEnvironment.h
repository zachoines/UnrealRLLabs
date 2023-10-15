#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BaseEnvironment.h"
#include "VectorEnvironment.generated.h"

UCLASS()
class MECANUM_ROBOT_RL_API AVectorEnvironment : public AActor
{
    GENERATED_BODY()

public:
    // Sets default values for this actor's properties
    AVectorEnvironment();

    // Initialize the environment with 'n' number of BaseEnvironments
    void InitEnv(TSubclassOf<ABaseEnvironment> EnvironmentClass, TArray<FBaseInitParams*> ParamsArray);

    // Reset the environment and return the initial state
    TArray<FState> ResetEnv();

    /* 
        Perform a step in the environment using the given action :
        Returns transition Tuple for LAST tick: { Done, Trunc, Reward, Action, State, NextState } 
    */
    TTuple<TArray<bool>, TArray<bool>, TArray<float>, TArray<FAction>, TArray<FState>, TArray<FState>> Transition();

    void Step(TArray<FAction> Actions);

    // Randomly sample actions with shape (num envs, num actions)
    TArray<FAction> SampleActions();
protected:
    virtual void BeginPlay() override;

private:
    TArray<ABaseEnvironment*> Environments;
    TArray<FAction> LastActions;
    TArray<FState> LastStates;
    TArray<FState> CurrentStates;
};