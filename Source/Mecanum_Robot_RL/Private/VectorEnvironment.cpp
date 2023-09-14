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
        ResetFlags.Add(false);
    }
}


TArray<TArray<float>> AVectorEnvironment::ResetEnv()
{
    TArray<TArray<float>> States;
    for (auto* Env : Environments)
    {
        States.Add(Env->ResetEnv());
    }
    return States;
}

TTuple<TArray<bool>, TArray<float>, TArray<TArray<float>>> AVectorEnvironment::Step(TArray<TArray<float>> Actions)
{
    TArray<bool> Dones;
    TArray<float> Rewards;
    TArray<TArray<float>> States;

    for (int32 i = 0; i < Environments.Num(); i++)
    {
        TArray<float> State;
        bool Done;
        float Reward;

        // If the environment needs to be reset, reset it
        if (ResetFlags[i])
        {
            State = Environments[i]->ResetEnv();
            Done = false;
            Reward = 0.0f; // or whatever default reward you want to give on reset
            ResetFlags[i] = false;
        }
        else
        {
            TTuple<bool, float, TArray<float>> Result = Environments[i]->Step(Actions[i]);
            Done = Result.Get<0>();
            Reward = Result.Get<1>();
            State = Result.Get<2>();

            // If the environment has reached a terminal state, mark it for reset
            if (Done)
            {
                ResetFlags[i] = true;
            }
        }

        Dones.Add(Done);
        Rewards.Add(Reward);
        States.Add(State);
    }

    return TTuple<TArray<bool>, TArray<float>, TArray<TArray<float>>>(Dones, Rewards, States);
}

TArray<TArray<float>> AVectorEnvironment::SampleActions()
{
    TArray<TArray<float>> Actions;
    for (auto* Env : Environments)
    {
        Actions.Add(Env->ActionSpace->Sample());
    }
    return Actions;
}
