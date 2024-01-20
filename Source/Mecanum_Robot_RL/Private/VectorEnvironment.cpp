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

/*
    Returns the transition results from the previous step. Does not reset environments, 
    Make sure to call on next tick ofter Step(), otherwise you'll lose that step's transition info.
*/
TTuple<TArray<bool>, TArray<bool>, TArray<float>, TArray<FAction>, TArray<FState>, TArray<FState>> AVectorEnvironment::Transition()
{
    TArray<bool> Dones;
    TArray<bool> Truncs;
    TArray<float> Rewards;
    TArray<FState> TmpStates;
    LastStates = CurrentStates;

    for (int32 i = 0; i < Environments.Num(); i++)
    {
        Dones.Add(Environments[i]->Done());
        Truncs.Add(Environments[i]->Trunc());
        Rewards.Add(Environments[i]->Reward());
        TmpStates.Add(Environments[i]->State());
        Environments[i]->PostTransition();   
    }
    
    for (int32 i = 0; i < Environments.Num(); i++)
    {
        Environments[i]->PostTransition();
    }

    CurrentStates = TmpStates;

    return TTuple<TArray<bool>, TArray<bool>, TArray<float>, TArray<FAction>, TArray<FState>, TArray<FState>>(
        Dones, Truncs, Rewards, LastActions, LastStates, CurrentStates
    );
}

/*
    Gets current states. Reset environments if in a done state from previous step.
*/
TArray<FState> AVectorEnvironment::GetStates() {
    
    TArray<FState> TmpStates;

    for (int32 i = 0; i < Environments.Num(); i++)
    {
        TmpStates.Add(
            Environments[i]->Done() || Environments[i]->Trunc() ? 
                Environments[i]->ResetEnv(CurrentAgents) : 
                Environments[i]->State()
        );
    }
    CurrentStates = TmpStates;
    return TmpStates;
}

/*
    Steps through environments with actions. Make sure to call GetStates first.
*/
void AVectorEnvironment::Step(TArray<FAction> Actions) 
{
    for (int32 i = 0; i < Environments.Num(); i++)
    {
        Environments[i]->Act(Actions[i]);
    }

    for (int32 i = 0; i < Environments.Num(); i++)
    {
        Environments[i]->PostStep();
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

    return SampledAction;
}