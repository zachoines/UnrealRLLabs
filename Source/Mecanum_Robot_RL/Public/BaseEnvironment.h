#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "RLTypes.h"
#include "BaseEnvironment.generated.h"

UCLASS()
class UNREALRLLABS_API ABaseEnvironment : public AActor
{
    GENERATED_BODY()

public:
    /** Sets default values for this actor's properties. */
    ABaseEnvironment();

    /** Initializes the environment with the provided init parameters. */
    virtual void InitEnv(FBaseInitParams* Params);

    /** Resets the environment and returns the initial state. */
    virtual FState ResetEnv(int NumAgents);

    virtual void PostInitializeComponents() override;

    /** Applies an action to the environment. */
    virtual void Act(FAction Action);
  
    /** Optional hook invoked after Step() in VectorEnvironment. */
    virtual void PostStep();

    /** Optional hook invoked after Transition() in VectorEnvironment. */
    virtual void PostTransition();

    /** Optional hook invoked before Step() in VectorEnvironment. */
    virtual void PreStep();

    /** Optional hook invoked before Transition() in VectorEnvironment. */
    virtual void PreTransition();

    /** Returns the public view of the current state. */
    virtual FState State();

    /** Returns true when the current episode should end. */
    virtual bool Done();

    /** Returns true when the episode is truncated. */
    virtual bool Trunc();

    /** Computes the reward for the current state. */
    virtual float Reward();

    FBaseInitParams* params;
};
