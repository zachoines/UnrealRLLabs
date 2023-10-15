#include "BaseEnvironment.h"

ABaseEnvironment::ABaseEnvironment()
{
    PrimaryActorTick.bCanEverTick = false;
}

void ABaseEnvironment::PostInitializeComponents()
{
    Super::PostInitializeComponents();
    ActionSpace = NewObject<UActionSpace>(this);
}

void ABaseEnvironment::InitEnv(FBaseInitParams* Params)
{
    // Setup actors...
}

FState ABaseEnvironment::ResetEnv()
{
    Update();
    return State();
}

void ABaseEnvironment::Act(FAction Action)
{
    // Move actors...
}

void ABaseEnvironment::Update()
{
    // Update internal state...
}

FState ABaseEnvironment::State()
{
    return FState();
}

float ABaseEnvironment::Reward()
{
    return 0.0f;
}

bool ABaseEnvironment::Done()
{
    return false;
}

bool ABaseEnvironment::Trunc()
{
    return false;
}