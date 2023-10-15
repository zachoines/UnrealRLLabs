#include "VectorEnvironment.h"

// Sets default values
AVectorEnvironment::AVectorEnvironment()
{
    // Disable ticking
    PrimaryActorTick.bCanEverTick = false;
}

void AVectorEnvironment::BeginPlay()
{
    Super::BeginPlay();
}

void AVectorEnvironment::InitEnv(TSubclassOf<ABaseEnvironment> EnvironmentClass, TArray<FBaseInitParams*> ParamsArray)
{
    for (FBaseInitParams* Param : ParamsArray)
    {
        ABaseEnvironment* Env = GetWorld()->SpawnActor<ABaseEnvironment>(EnvironmentClass, Param->Location, FRotator::ZeroRotator);
        Env->InitEnv(Param);
        Environments.Add(Env);
    }
}

TArray<FState> AVectorEnvironment::ResetEnv()
{
    TArray<FState> States;
    for (auto* Env : Environments)
    {
        States.Add(Env->ResetEnv());
    }
    CurrentStates = States;
    return CurrentStates;
}

TTuple<TArray<bool>, TArray<bool>, TArray<float>, TArray<FAction>, TArray<FState>, TArray<FState>> AVectorEnvironment::Transition()
{
    TArray<bool> Dones;
    TArray<bool> Truncs;
    TArray<float> Rewards;
    TArray<FState> TmpStates;
    LastStates = CurrentStates;

    for (int32 i = 0; i < Environments.Num(); i++)
    {
        Environments[i]->Update();
        bool Done = Environments[i]->Done();
        bool Trunc = Environments[i]->Trunc();
        float Reward = Environments[i]->Reward();
        FState State = Done || Trunc ? Environments[i]->ResetEnv() : Environments[i]->State();

        Dones.Add(Done);
        Truncs.Add(Trunc);
        Rewards.Add(Reward);
        TmpStates.Add(State);
    }

    CurrentStates = TmpStates;

    return TTuple<TArray<bool>, TArray<bool>, TArray<float>, TArray<FAction>, TArray<FState>, TArray<FState>>(
        Dones, Truncs, Rewards, LastActions, LastStates, CurrentStates
    );
}

void AVectorEnvironment::Step(TArray<FAction> Actions) 
{
    for (int32 i = 0; i < Environments.Num(); i++)
    {
        Environments[i]->Act(Actions[i]);
    }
    LastActions = Actions;
}

TArray<FAction> AVectorEnvironment::SampleActions()
{
    TArray<FAction> Actions;
    for (auto* Env : Environments)
    {
        Actions.Add(Env->ActionSpace->Sample());
    }
    return Actions;
}
