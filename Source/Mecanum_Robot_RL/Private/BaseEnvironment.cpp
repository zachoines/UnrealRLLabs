#include "BaseEnvironment.h"

ABaseEnvironment::ABaseEnvironment()
{
    PrimaryActorTick.bCanEverTick = true;
}

void ABaseEnvironment::PostInitializeComponents()
{
    Super::PostInitializeComponents();
}

void ABaseEnvironment::InitEnv(FBaseInitParams* Params)
{
}

FState ABaseEnvironment::ResetEnv(int NumAgents)
{
    return State();
}

void ABaseEnvironment::Act(FAction Action)
{
}

void ABaseEnvironment::PostStep()
{
}

void ABaseEnvironment::PostTransition()
{
}

void ABaseEnvironment::PreStep()
{
}

void ABaseEnvironment::PreTransition()
{
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
