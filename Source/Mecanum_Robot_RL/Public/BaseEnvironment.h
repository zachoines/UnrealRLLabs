#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "RLTypes.h"
#include "BaseEnvironment.generated.h"

UCLASS()
class MECANUM_ROBOT_RL_API ABaseEnvironment : public AActor
{
    GENERATED_BODY()

public:

    // Sets default values for this actor's properties
    ABaseEnvironment();

    UPROPERTY(BlueprintReadOnly, Category = "EnvInfo")
    FEnvInfo EnvInfo;

    // Initialize the environment using the FBaseInitParams struct
    virtual void InitEnv(FBaseInitParams* Params);

    // Reset the environment and return the initial state
    virtual FState ResetEnv(int NumAgents);

    virtual void PostInitializeComponents() override;

    // Update environment with actions
    virtual void Act(FAction Action);
  
    // Optional convenience function. Called after Step() in VectorEnvironment.
    virtual void PostStep();

    // Optional convenience function. Called after Transition() in VectorEnvironment.
    virtual void PostTransition();

    // Returns the public view of the state. Called after Update 
    virtual FState State();

    // Returns done conditon
    virtual bool Done();

    // Returns truncation conditon
    virtual bool Trunc();

    // Calculate the reward for the current state
    virtual float Reward();

    FBaseInitParams* params;
};
