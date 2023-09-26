#include "BaseEnvironment.h"

ABaseEnvironment::ABaseEnvironment()
{
    // Set default values for this actor's properties
    PrimaryActorTick.bCanEverTick = false;
}

void ABaseEnvironment::PostInitializeComponents()
{
    Super::PostInitializeComponents();
    ActionSpace = NewObject<UActionSpace>(this);
}

void ABaseEnvironment::InitEnv(FBaseInitParams* Params)
{
    // Default implementation does nothing. Derived classes should provide specific behavior.
}

FState ABaseEnvironment::ResetEnv()
{
    // Default implementation returns an empty array. Derived classes should provide specific behavior.
    return FState();
}

TTuple<bool, float, FState> ABaseEnvironment::Step(FAction Action)
{
    // Default implementation returns a tuple with default values. Derived classes should provide specific behavior.
    return TTuple<bool, float, FState>(false, 0.0f, FState());
}

float ABaseEnvironment::Reward()
{
    // Default implementation returns 0. Derived classes should provide specific behavior.
    return 0.0f;
}