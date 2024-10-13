#include "VectorEnvironment.h"

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

    SingleEnvInfo = Environments[0]->EnvInfo;
}

TArray<FState> AVectorEnvironment::ResetEnv(int NumAgents)
{
    TArray<FState> States;
    CurrentAgents = NumAgents;
    for (auto* Env : Environments)
    {
        States.Add(Env->ResetEnv(CurrentAgents));
    }
    CurrentStates = States;
    return CurrentStates;
}

TTuple<TArray<float>, TArray<float>, TArray<float>, TArray<FAction>, TArray<FState>, TArray<FState>> AVectorEnvironment::Transition()
{
    TArray<float> Rewards;
    TArray<FState> States;
    LastStates = CurrentStates;

    for (int32 i = 0; i < Environments.Num(); i++)
    {
        Dones.Add(static_cast<float>(Environments[i]->Done()));
        Truncs.Add(static_cast<float>(Environments[i]->Trunc()));
        Rewards.Add(Environments[i]->Reward());
        States.Add(Environments[i]->State());
    }

    CurrentStates = States;

    return TTuple<TArray<float>, TArray<float>, TArray<float>, TArray<FAction>, TArray<FState>, TArray<FState>>(
        Dones, Truncs, Rewards, LastActions, LastStates, States
    );
}

TArray<FState> AVectorEnvironment::GetStates()
{
    TArray<FState> TmpStates;

    for (int32 i = 0; i < Environments.Num(); i++)
    {
        TmpStates.Add(Environments[i]->State());
    }
    CurrentStates = TmpStates;
    return TmpStates;
}

void AVectorEnvironment::Step(TArray<FAction> Actions)
{
    for (int32 i = 0; i < Environments.Num(); i++)
    {
        if (!Dones[i] && !Truncs[i]) {
            Environments[i]->Act(Actions[i]);
        }
    }

    for (int32 i = 0; i < Environments.Num(); i++)
    {
        if (!Dones[i] && !Truncs[i]) {
            Environments[i]->PostStep();
        }
    }
    LastActions = Actions;
}

TArray<FAction> AVectorEnvironment::SampleActions()
{
    TArray<FAction> Actions;
    for (auto* Env : Environments)
    {
        Actions.Add(EnvSample(Env->EnvInfo.ActionSpace));
    }
    return Actions;
}

FAction AVectorEnvironment::EnvSample(UActionSpace* ActionSpace)
{
    FAction SampledAction;

    // For each agent
    for (int i = 0; i < CurrentAgents; i++) {
        // Sample discrete actions
        for (const FDiscreteActionSpec& DiscreteAction : ActionSpace->DiscreteActions)
        {
            int32 RandomChoice = FMath::RandRange(0, DiscreteAction.NumChoices - 1);
            SampledAction.Values.Add(static_cast<float>(RandomChoice));
        }

        // Sample continuous actions
        for (const FContinuousActionSpec& ContinuousAction : ActionSpace->ContinuousActions)
        {
            float RandomValue = FMath::RandRange(ContinuousAction.Low, ContinuousAction.High);
            SampledAction.Values.Add(RandomValue);
        }
    }

    return SampledAction;
}