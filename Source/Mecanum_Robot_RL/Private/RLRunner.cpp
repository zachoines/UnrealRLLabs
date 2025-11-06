// NOTICE: This file includes modifications generated with the assistance of generative AI.
// Original code structure and logic by the project author.

#include "RLRunner.h"
#include "Kismet/GameplayStatics.h"
#include "Misc/Paths.h"
#include "Windows/AllowWindowsPlatformTypes.h"
#include "Windows.h"
#include "Windows/HideWindowsPlatformTypes.h"

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
    bTestingMode = false;
    PrimaryActorTick.TickGroup = ETickingGroup::TG_PostUpdateWork;
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

    ExperienceBufferInstance = NewObject<UExperienceBuffer>(this);
    if (ExperienceBufferInstance)
    {
        // PPO uses chronological, non-replacement sampling.
        ExperienceBufferInstance->Initialize(
            Environments.Num(),
            BufferSize,
            false,
            false
        );
    }

    AgentComm = NewObject<USharedMemoryAgentCommunicator>(this);
    if (AgentComm)
    {
        AgentComm->Init(EnvConfig);
    }

    ParseActionSpaceFromConfig();

    CurrentAgents = (bIsMultiAgent) ? FMath::RandRange(MinAgents, MaxAgents) : 1;

    for (ABaseEnvironment* Env : Environments)
    {
        Env->ResetEnv(CurrentAgents);
    }

    Pending.SetNum(Environments.Num());
    for (int32 i = 0; i < Environments.Num(); i++)
    {
        StartNewBlock(i, GetEnvState(Environments[i]));
    }

    CurrentStep = 0;
    CurrentUpdate = 0;
}

void ARLRunner::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    if (AgentComm)
    {
        void* beginHandle = AgentComm->GetBeginTestEventHandle();
        if (beginHandle && WaitForSingleObject((HANDLE)beginHandle, 0) == WAIT_OBJECT_0)
        {
            BeginTestMode();
        }
        void* endHandle = AgentComm->GetEndTestEventHandle();
        if (endHandle && WaitForSingleObject((HANDLE)endHandle, 0) == WAIT_OBJECT_0)
        {
            EndTestMode();
        }
    }

    if (CurrentStep > 0)
    {
        CollectTransitions();
    }

    DecideActions();
    StepEnvironments();

    CurrentStep++;
}

void ARLRunner::CollectTransitions()
{
    for (int32 i = 0; i < Environments.Num(); i++)
    {
        ABaseEnvironment* Env = Environments[i];
        Env->PreTransition();

        float stepReward = Env->Reward();
        bool bDone = Env->Done();
        bool bTrunc = Env->Trunc();

        Pending[i].AccumulatedReward += stepReward;
        Pending[i].RepeatCounter++;

        bool bActionRepeatFinished = (Pending[i].RepeatCounter >= ActionRepeat);

        if (bDone || bTrunc || bActionRepeatFinished)
        {
            FState nextStateForExperience;
            FState stateForNextBlockStart;

            if (bDone || bTrunc)
            {
                ResetEnvironment(i);
                nextStateForExperience = GetEnvState(Environments[i]);
                stateForNextBlockStart = nextStateForExperience;
            }
            else
            {
                nextStateForExperience = GetEnvState(Environments[i]);
                stateForNextBlockStart = nextStateForExperience;
            }

            FinalizeTransition(i, nextStateForExperience, bDone, bTrunc);
            StartNewBlock(i, stateForNextBlockStart);
        }
        Env->PostTransition();
    }
}

void ARLRunner::DecideActions()
{
    bool anyNeedNewAction = false;
    for (const FPendingTransition& p : Pending)
    {
        if (p.RepeatCounter == 0) // New action needed if counter is 0 (start of a block)
        {
            anyNeedNewAction = true;
            break;
        }
    }

    if (!anyNeedNewAction)
    {
        return; // All environments are in the middle of an action-repeat cycle.
    }

    TArray<FState> EnvStates;
    EnvStates.SetNum(Environments.Num());
    TArray<float> EnvDones;
    EnvDones.SetNum(Environments.Num());
    TArray<float> EnvTruncs;
    EnvTruncs.SetNum(Environments.Num());
    TArray<float> NeedsAction;
    NeedsAction.SetNum(Environments.Num());

    for (int32 i = 0; i < Environments.Num(); i++)
    {
        EnvStates[i] = GetEnvState(Environments[i]);

        bool bCurrentDone = Environments[i]->Done();
        bool bCurrentTrunc = Environments[i]->Trunc();
        if (Pending[i].bDoneOrTrunc && Pending[i].RepeatCounter == 0)
        {
            EnvDones[i] = 1.f;
            EnvTruncs[i] = 1.f;
        }
        else
        {
            EnvDones[i] = bCurrentDone ? 1.f : 0.f;
            EnvTruncs[i] = bCurrentTrunc ? 1.f : 0.f;
        }
        NeedsAction[i] = (Pending[i].RepeatCounter == 0) ? 1.f : 0.f;
    }

    TArray<FAction> newActions = GetActionsFromPython(EnvStates, EnvDones, EnvTruncs, NeedsAction);

    for (int32 i = 0; i < Environments.Num(); i++)
    {
        if (Pending[i].RepeatCounter == 0)
        {
            Pending[i].Action = newActions[i];
            Pending[i].bDoneOrTrunc = false;
        }
    }
}

void ARLRunner::StepEnvironments()
{
    for (int32 i = 0; i < Environments.Num(); i++)
    {
        ABaseEnvironment* Env = Environments[i];
        if (Env->Done() || Env->Trunc())
        {
            continue;
        }

        Env->PreStep();
        Env->Act(Pending[i].Action);
        Env->PostStep();
    }
}

void ARLRunner::FinalizeTransition(int EnvIndex, const FState& NextState, bool bDone, bool bTrunc)
{
    FPendingTransition& P = Pending[EnvIndex];

    FExperience Exp;
    Exp.State = P.PrevState;
    Exp.Action = P.Action;
    Exp.Reward = P.AccumulatedReward;
    Exp.NextState = NextState;
    Exp.Done = bDone;
    Exp.Trunc = bTrunc;

    if (ExperienceBufferInstance)
    {
        ExperienceBufferInstance->AddExperience(EnvIndex, Exp);
    }

    Pending[EnvIndex].bDoneOrTrunc = bDone || bTrunc;

    MaybeTrainUpdate();
}

void ARLRunner::StartNewBlock(int EnvIndex, const FState& State)
{
    FPendingTransition& p = Pending[EnvIndex];
    p.PrevState = State;
    p.AccumulatedReward = 0.f;
    p.RepeatCounter = 0;
}

void ARLRunner::ResetEnvironment(int EnvIndex)
{
    CurrentAgents = (bIsMultiAgent) ? FMath::RandRange(MinAgents, MaxAgents) : 1;
    Environments[EnvIndex]->ResetEnv(CurrentAgents);
}

void ARLRunner::MaybeTrainUpdate()
{
    if (!AgentComm || !ExperienceBufferInstance) return;

    int32 minSize = ExperienceBufferInstance->MinSizeAcrossEnvs();
    if (minSize >= BatchSize)
    {
        CurrentUpdate++;
        TArray<FExperienceBatch> Batches = ExperienceBufferInstance->SampleEnvironmentTrajectories(BatchSize);
        AgentComm->Update(Batches, CurrentAgents);
    }
}

void ARLRunner::ParseActionSpaceFromConfig()
{
    DiscreteActionSizes.Empty();
    ContinuousActionRanges.Empty();
    if (!EnvConfig) return;

    FString ActionBasePath = TEXT("environment/shape/action/");
    if (EnvConfig->HasPath(ActionBasePath + TEXT("agent")))
    {
        ActionBasePath += TEXT("agent/");
    }
    else if (EnvConfig->HasPath(ActionBasePath + TEXT("central")))
    {
        ActionBasePath += TEXT("central/");
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("RLRunner::ParseActionSpaceFromConfig - Neither agent nor central action space found in config."));
        return;
    }

    if (EnvConfig->HasPath(ActionBasePath + TEXT("discrete")))
    {
        auto Node = EnvConfig->Get(ActionBasePath + TEXT("discrete"));
        auto Arr = Node->AsArrayOfConfigs();
        for (auto* c : Arr)
        {
            if (c->HasPath(TEXT("num_choices")))
            {
                int32 n = c->Get(TEXT("num_choices"))->AsInt();
                DiscreteActionSizes.Add(n);
            }
        }
    }

    if (EnvConfig->HasPath(ActionBasePath + TEXT("continuous")))
    {
        auto Node = EnvConfig->Get(ActionBasePath + TEXT("continuous"));
        auto Arr = Node->AsArrayOfConfigs();
        for (auto* r : Arr)
        {
            if (r->HasPath(TEXT("min")) && r->HasPath(TEXT("max")))
            {
                float mn = r->Get(TEXT("min"))->AsNumber();
                float mx = r->Get(TEXT("max"))->AsNumber();
                ContinuousActionRanges.Add(FVector2D(mn, mx));
            }
        }
    }
}

FAction ARLRunner::EnvSample(int EnvIndex)
{
    FAction act;
    int32 numAgentsToSampleFor = bIsMultiAgent ? CurrentAgents : 1;

    for (int32 agentIdx = 0; agentIdx < numAgentsToSampleFor; agentIdx++)
    {
        for (int32 dSize : DiscreteActionSizes)
        {
            act.Values.Add(static_cast<float>(FMath::RandRange(0, dSize - 1)));
        }
        for (const FVector2D& range : ContinuousActionRanges)
        {
            act.Values.Add(FMath::FRandRange(range.X, range.Y));
        }
    }
    return act;
}

FState ARLRunner::GetEnvState(ABaseEnvironment* Env)
{
    return Env->State();
}

TArray<FAction> ARLRunner::GetActionsFromPython(
    const TArray<FState>& EnvStates,
    const TArray<float>& EnvDones,
    const TArray<float>& EnvTruncs,
    const TArray<float>& NeedsAction
)
{
    if (!AgentComm)
    {
        UE_LOG(LogTemp, Warning, TEXT("AgentComm is null in GetActionsFromPython, sampling random actions."));
        return SampleAllEnvActions();
    }
    return AgentComm->GetActions(EnvStates, EnvDones, EnvTruncs, NeedsAction, CurrentAgents);
}

TArray<FAction> ARLRunner::SampleAllEnvActions()
{
    TArray<FAction> out;
    out.Reserve(Environments.Num());
    for (int32 i = 0; i < Environments.Num(); i++)
    {
        out.Add(EnvSample(i));
    }
    return out;
}

void ARLRunner::BeginTestMode()
{
    if (bTestingMode) return;
    bTestingMode = true;
    if (ExperienceBufferInstance)
    {
        ExperienceBufferInstance->Clear();
    }
    for (int32 i = 0; i < Environments.Num(); ++i)
    {
        ResetEnvironment(i);
        FState st = GetEnvState(Environments[i]);
        StartNewBlock(i, st);
    }
}

void ARLRunner::EndTestMode()
{
    if (!bTestingMode) return;
    bTestingMode = false;
    if (ExperienceBufferInstance)
    {
        ExperienceBufferInstance->Clear();
    }
    for (int32 i = 0; i < Environments.Num(); ++i)
    {
        ResetEnvironment(i);
        FState st = GetEnvState(Environments[i]);
        StartNewBlock(i, st);
    }
}
