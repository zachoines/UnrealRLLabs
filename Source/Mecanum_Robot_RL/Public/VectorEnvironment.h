#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BaseEnvironment.h"
#include "RLTypes.h"
#include "VectorEnvironment.generated.h"

UCLASS()
class UNREALRLLABS_API AVectorEnvironment : public AActor
{
    GENERATED_BODY()

public:
    // Sets default values for this actor's properties
    AVectorEnvironment();

    // Initialize the environment with 'n' number of BaseEnvironments
    void InitEnv(TSubclassOf<ABaseEnvironment> EnvironmentClass, TArray<FBaseInitParams*> ParamsArray);

    // Reset the environment and return the initial state
    TArray<FState> ResetEnv(int NumAgents);

    /*
        Perform a step in the environment using the given action :
        Returns transition Tuple for LAST tick: { Done, Trunc, Reward, Action, State, NextState }
    */
    TTuple<TArray<float>, TArray<float>, TArray<float>, TArray<FAction>, TArray<FState>, TArray<FState>> Transition();

    void Step(TArray<FAction> Actions);

    // Randomly sample actions with shape (num envs, num actions)
    TArray<FAction> SampleActions();

    TArray<FState> GetStates();

    FEnvInfo SingleEnvInfo;
    int CurrentAgents;

protected:
    virtual void BeginPlay() override;

private:
    TArray<ABaseEnvironment*> Environments;
    TArray<FAction> LastActions;
    TArray<FState> LastStates;
    TArray<FState> CurrentStates;
    TArray<float> CurrentDones;
    TArray<float> CurrentTruncs;

    FAction EnvSample(UActionSpace* ActionSpace);
};