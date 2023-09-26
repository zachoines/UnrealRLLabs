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

    // Perform a step in the environment using the given action
    TTuple<TArray<bool>, TArray<float>, TArray<FState>> Step(TArray<FAction> Actions);

    // Randomly sample actions with shape (num envs, num actions)
    TArray<FAction> SampleActions();
protected:
    virtual void BeginPlay() override;

private:
    TArray<ABaseEnvironment*> Environments;
    TArray<bool> ResetFlags;
};