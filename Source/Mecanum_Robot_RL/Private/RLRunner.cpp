// NOTICE: This file includes modifications generated with the assistance of generative AI.
// Original code structure and logic by the project author.

#include "RLRunner.h"
#include "Kismet/GameplayStatics.h"
#include "Misc/Paths.h"

ARLRunner::ARLRunner()
{
    PrimaryActorTick.bCanEverTick = true;
    bIsMultiAgent = false;
    MinAgents = 1;
    MaxAgents = 1;
    CurrentAgents = 1;

    BufferSize = 1024;
    BatchSize = 128;
    ActionRepeat = 1;

    CurrentStep = 0;
    CurrentUpdate = 0;
}

void ARLRunner::InitRunner(
    TSubclassOf<ABaseEnvironment> EnvironmentClass,
    TArray<FBaseInitParams*> ParamsArray,
    UEnvironmentConfig* InEnvConfig
)
{
    EnvConfig = InEnvConfig;

    bIsMultiAgent = EnvConfig->HasPath(TEXT("environment/shape/state/agent"));
    if (bIsMultiAgent)
    {
        MinAgents = EnvConfig->Get(TEXT("environment/shape/state/agent/min"))->AsInt();
        MaxAgents = EnvConfig->Get(TEXT("environment/shape/state/agent/max"))->AsInt();
    }
    else
    {
        MinAgents = 1;
        MaxAgents = 1;
    }

    if (EnvConfig->HasPath(TEXT("train/buffer_size")))
    {
        BufferSize = EnvConfig->Get(TEXT("train/buffer_size"))->AsInt();
    }
    if (EnvConfig->HasPath(TEXT("train/batch_size")))
    {
        BatchSize = EnvConfig->Get(TEXT("train/batch_size"))->AsInt();
    }
    if (EnvConfig->HasPath(TEXT("train/ActionRepeat")))
    {
        ActionRepeat = EnvConfig->Get(TEXT("train/ActionRepeat"))->AsInt();
        if (ActionRepeat < 1) ActionRepeat = 1;
    }

    UWorld* World = GetWorld();
    if (!World)
    {
        UE_LOG(LogTemp, Error, TEXT("No valid UWorld in RLRunner."));
        return;
    }

    Environments.Empty();
    for (FBaseInitParams* Param : ParamsArray)
    {
        ABaseEnvironment* Env = World->SpawnActor<ABaseEnvironment>(
            EnvironmentClass,
            Param->Location,
            FRotator::ZeroRotator
        );
        if (Env)
        {
            Env->InitEnv(Param);
            Environments.Add(Env);
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to spawn environment actor."));
        }
    }

    // Create the Experience Buffer
    ExperienceBufferInstance = NewObject<UExperienceBuffer>(this);
    if (ExperienceBufferInstance)
    {
        ExperienceBufferInstance->Initialize(
            Environments.Num(), // # environments
            BufferSize,         // capacity
            false,              // sample_with_replacement (for PPO => false)
            false               // random-sample => false (for PPO => chronological)
        );
    }

    // Create communicator
    AgentComm = NewObject<USharedMemoryAgentCommunicator>(this);
    if (AgentComm)
    {
        AgentComm->Init(EnvConfig);
    }

    ParseActionSpaceFromConfig();

    CurrentAgents = (bIsMultiAgent) ? FMath::RandRange(MinAgents, MaxAgents) : 1;

    // Reset each environment
    for (ABaseEnvironment* Env : Environments)
    {
        Env->ResetEnv(CurrentAgents);
    }

    // Create pending transitions
    Pending.SetNum(Environments.Num());
    for (int32 i = 0; i < Environments.Num(); i++)
    {
        // We'll store the environment's current state
        FState st = GetEnvState(Environments[i]);
        StartNewBlock(i, st);
    }

    CurrentStep = 0;
    CurrentUpdate = 0;
}

void ARLRunner::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    // We skip collecting transitions on the first tick, since
    // there's no previous step to form a valid transition from.
    if (CurrentStep > 0)
    {
        CollectTransitions();
    }

    DecideActions();
    StepEnvironments();

    CurrentStep++;
}

// ---------------------
//  CollectTransitions
// ---------------------
void ARLRunner::CollectTransitions()
{
    for (int32 i = 0; i < Environments.Num(); i++)
    {
        ABaseEnvironment* Env = Environments[i];
        Env->PreTransition();

        float stepReward = Env->Reward();
        bool  bDone = Env->Done();
        bool  bTrunc = Env->Trunc();

        // accumulate
        Pending[i].AccumulatedReward += stepReward;

        // increment the counter
        Pending[i].RepeatCounter++;

        // if done/trunc => or actionRepeat block ended => finalize
        bool bActionRepeatFinished = (Pending[i].RepeatCounter >= ActionRepeat);
        if (bDone || bTrunc || bActionRepeatFinished)
        {
            // gather nextState
            FState nextState;
            if (bDone || bTrunc)
            {
                // reset
                ResetEnvironment(i);
                nextState = GetEnvState(Environments[i]);
            }
            else
            {
                nextState = GetEnvState(Environments[i]);
            }

            FinalizeTransition(i, nextState, bDone, bTrunc);

            if (!bDone && !bTrunc)
            {
                // start new block
                StartNewBlock(i, nextState);
            }
        }

        Env->PostTransition();
    }
}

// ---------------------
//  DecideActions
// ---------------------
void ARLRunner::DecideActions()
{
    // We'll see which envs need a new action => if their repeatCounter==0
    // Because we either just started or we just finished a block
    bool anyNeedNew = false;
    for (auto& p : Pending)
    {
        if (p.RepeatCounter == 0)
        {
            anyNeedNew = true;
            break;
        }
    }

    // If no environment is at repeatCounter=0, we keep old actions
    if (!anyNeedNew)
    {
        return;
    }

    // We gather states/dones/truncs for each environment
    TArray<FState> EnvStates;
    EnvStates.SetNum(Environments.Num());
    TArray<float> Dones;
    Dones.SetNum(Environments.Num());
    TArray<float> Truncs;
    Truncs.SetNum(Environments.Num());

    for (int32 i = 0; i < Environments.Num(); i++)
    {
        EnvStates[i] = GetEnvState(Environments[i]);
        bool bDone = Environments[i]->Done();
        bool bTrunc = Environments[i]->Trunc();
        Dones[i] = (bDone ? 1.f : 0.f);
        Truncs[i] = (bTrunc ? 1.f : 0.f);
    }

    // get actions from python or fallback
    TArray<FAction> newActions = GetActionsFromPython(EnvStates, Dones, Truncs);

    // assign to those envs whose repeatCounter == 0
    for (int32 i = 0; i < Environments.Num(); i++)
    {
        if (Pending[i].RepeatCounter == 0)
        {
            Pending[i].Action = newActions[i];
        }
    }
}

// ---------------------
//  StepEnvironments
// ---------------------
void ARLRunner::StepEnvironments()
{
    // For each env => if done, skip stepping
    for (int32 i = 0; i < Environments.Num(); i++)
    {
        ABaseEnvironment* Env = Environments[i];
        bool bDone = Env->Done();
        bool bTrunc = Env->Trunc();
        if (bDone || bTrunc)
        {
            continue; // skip stepping
        }

        Env->PreStep();
        Env->Act(Pending[i].Action);
        Env->PostStep();
    }
}

// ---------------------
//  FinalizeTransition
// ---------------------
void ARLRunner::FinalizeTransition(int EnvIndex, const FState& NextState, bool bDone, bool bTrunc)
{
    FPendingTransition& P = Pending[EnvIndex];

    // create experience
    FExperience Exp;
    Exp.State = P.PrevState;
    Exp.Action = P.Action;
    Exp.Reward = P.AccumulatedReward;
    Exp.NextState = NextState;
    Exp.Done = bDone;
    Exp.Trunc = bTrunc;

    // add to the experience buffer
    if (ExperienceBufferInstance)
    {
        ExperienceBufferInstance->AddExperience(EnvIndex, Exp);
    }

    // done => we don't start a new block here 
    // that is done in CollectTransitions if not done/trunc

    // check if each environment has >= batch_size => do an update if yes
    MaybeTrainUpdate();
}

// ---------------------
//  StartNewBlock
// ---------------------
void ARLRunner::StartNewBlock(int EnvIndex, const FState& State)
{
    FPendingTransition& p = Pending[EnvIndex];
    p.PrevState = State;
    p.Action = FAction();
    p.AccumulatedReward = 0.f;
    p.RepeatCounter = 0;
    p.bDoneOrTrunc = false;
}

// ---------------------
//  ResetEnvironment
// ---------------------
void ARLRunner::ResetEnvironment(int EnvIndex)
{
    Environments[EnvIndex]->ResetEnv(CurrentAgents);
}

// ---------------------
//  MaybeTrainUpdate
// ---------------------
void ARLRunner::MaybeTrainUpdate()
{
    if (!AgentComm || !ExperienceBufferInstance) return;

    // if all env deques have at least BatchSize
    int32 minSize = ExperienceBufferInstance->MinSizeAcrossEnvs();
    if (minSize >= BatchSize)
    {
        CurrentUpdate++;
        // we sample from each env => produce an array of FExperienceBatch => one per env
        TArray<FExperienceBatch> Batches = ExperienceBufferInstance->SampleEnvironmentTrajectories(BatchSize);
        // Send to Python
        AgentComm->Update(Batches, CurrentAgents);
    }
}

// ---------------------
//  parse action space
// ---------------------
void ARLRunner::ParseActionSpaceFromConfig()
{
    DiscreteActionSizes.Empty();
    ContinuousActionRanges.Empty();
    if (!EnvConfig) return;

    if (EnvConfig->HasPath(TEXT("environment/shape/action/agent/discrete")))
    {
        auto Node = EnvConfig->Get(TEXT("environment/shape/action/agent/discrete"));
        auto Arr = Node->AsArrayOfConfigs();
        for (auto* c : Arr)
        {
            int32 n = c->Get(TEXT("num_choices"))->AsInt();
            DiscreteActionSizes.Add(n);
        }
    }
    if (EnvConfig->HasPath(TEXT("environment/shape/action/agent/continuous")))
    {
        auto Node = EnvConfig->Get(TEXT("environment/shape/action/agent/continuous"));
        auto Arr = Node->AsArrayOfConfigs();
        for (auto* r : Arr)
        {
            float mn = r->Get(TEXT("min"))->AsNumber();
            float mx = r->Get(TEXT("max"))->AsNumber();
            ContinuousActionRanges.Add(FVector2D(mn, mx));
        }
    }
}

// ---------------------
//  EnvSample
// ---------------------
FAction ARLRunner::EnvSample(int EnvIndex)
{
    FAction act;
    for (int32 agentIdx = 0; agentIdx < CurrentAgents; agentIdx++)
    {
        // discrete
        for (int32 d = 0; d < DiscreteActionSizes.Num(); d++)
        {
            int32 range = DiscreteActionSizes[d];
            int32 choice = FMath::RandRange(0, range - 1);
            act.Values.Add((float)choice);
        }
        // continuous
        for (int32 c = 0; c < ContinuousActionRanges.Num(); c++)
        {
            float mn = ContinuousActionRanges[c].X;
            float mx = ContinuousActionRanges[c].Y;
            float val = FMath::RandRange(mn, mx);
            act.Values.Add(val);
        }
    }
    return act;
}

// ---------------------
//  GetEnvState
// ---------------------
FState ARLRunner::GetEnvState(ABaseEnvironment* Env)
{
    return Env->State();
}

// ---------------------
//  GetActionsFromPython
// ---------------------
TArray<FAction> ARLRunner::GetActionsFromPython(
    const TArray<FState>& EnvStates,
    const TArray<float>& EnvDones,
    const TArray<float>& EnvTruncs
)
{
    if (!AgentComm)
    {
        return SampleAllEnvActions();
    }
    // use communicator
    return AgentComm->GetActions(EnvStates, EnvDones, EnvTruncs, CurrentAgents);
}

TArray<FAction> ARLRunner::SampleAllEnvActions()
{
    TArray<FAction> out;
    out.SetNum(Environments.Num());
    for (int32 i = 0; i < Environments.Num(); i++)
    {
        out[i] = EnvSample(i);
    }
    return out;
}
