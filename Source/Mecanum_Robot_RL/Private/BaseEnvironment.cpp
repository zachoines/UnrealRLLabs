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

TArray<float> ABaseEnvironment::ResetEnv()
{
    // Default implementation returns an empty array. Derived classes should provide specific behavior.
    return TArray<float>();
}

TTuple<bool, float, TArray<float>> ABaseEnvironment::Step(TArray<float> Action)
{
    // Default implementation returns a tuple with default values. Derived classes should provide specific behavior.
    return TTuple<bool, float, TArray<float>>(false, 0.0f, TArray<float>());
}

float ABaseEnvironment::Reward()
{
    // Default implementation returns 0. Derived classes should provide specific behavior.
    return 0.0f;
}