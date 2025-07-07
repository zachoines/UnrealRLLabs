// NOTICE: This file includes modifications generated with the assistance of generative AI.
// Original code structure and logic by the project author.
// Fix applied based on discussion to correctly handle PrevState after episode termination.

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
        ExperienceBufferInstance->Initialize(
            Environments.Num(),
            BufferSize,
            false, // sample_with_replacement (for PPO => false)
            false  // random-sample => false (for PPO => chronological)
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
        FState st = GetEnvState(Environments[i]);
        StartNewBlock(i, st); // Initialize pending transitions with the initial state
    }

    CurrentStep = 0;
    CurrentUpdate = 0;
}

void ARLRunner::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

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
        Env->PreTransition(); // Environment updates its internal state based on the last action

        // Get results of the last action
        float stepReward = Env->Reward();
        bool  bDone = Env->Done();
        bool  bTrunc = Env->Trunc();

        Pending[i].AccumulatedReward += stepReward;
        Pending[i].RepeatCounter++;

        bool bActionRepeatFinished = (Pending[i].RepeatCounter >= ActionRepeat);

        if (bDone || bTrunc || bActionRepeatFinished)
        {
            FState nextStateForExperience; // This will be S_t+1 (if continuing) or S'_0 (if episode just ended)
            FState stateForNextBlockStart; // This is S_t+1 (if continuing) or S'_0 (if episode just ended)

            if (bDone || bTrunc)
            {
                // Episode terminated. Env[i] is currently in the state that caused termination (S_t+1).
                // For the experience being finalized, PrevState = S_t (from Pending[i]), Action = A_t (from Pending[i]),
                // Reward = R_accumulated (from Pending[i]).
                // The NextState for this *terminating experience* should be the initial state of the new episode (S'_0).
                ResetEnvironment(i); // Env[i] is now reset and in state S'_0
                nextStateForExperience = GetEnvState(Environments[i]); // This is S'_0
                stateForNextBlockStart = nextStateForExperience;       // The new block will also start with S'_0
            }
            else
            {
                // Action repeat finished, but episode continues. Env[i] is in S_t+1.
                // The NextState for this experience is the current state S_t+1.
                nextStateForExperience = GetEnvState(Environments[i]); // This is S_t+1
                stateForNextBlockStart = nextStateForExperience;       // The new block will also start with S_t+1
            }

            // Finalize the current transition using Pending[i].PrevState, Pending[i].Action,
            // Pending[i].AccumulatedReward, and the determined nextStateForExperience.
            FinalizeTransition(i, nextStateForExperience, bDone, bTrunc);

            // After finalizing, always set up the Pending struct for the next segment.
            // stateForNextBlockStart will be S'_0 if terminated, or S_t+1 if action repeat finished and episode continues.
            // This correctly sets Pending[i].PrevState for the *next* transition to be formed.
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
        return; // All environments are in the middle of an action-repeat cycle
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
        // The state used for action decision should be the current state of the environment,
        // which is also Pending[i].PrevState if RepeatCounter is 0.
        EnvStates[i] = GetEnvState(Environments[i]); // Or Pending[i].PrevState if RepeatCounter is 0

        // Done/Trunc should reflect the current status before taking a new action.
        // If an environment just reset, Done/Trunc will be false.
        bool bCurrentDone = Environments[i]->Done();
        bool bCurrentTrunc = Environments[i]->Trunc();
        EnvDones[i] = bCurrentDone ? 1.f : 0.f;
        EnvTruncs[i] = bCurrentTrunc ? 1.f : 0.f;
        NeedsAction[i] = (Pending[i].RepeatCounter == 0) ? 1.f : 0.f;
    }

    TArray<FAction> newActions = GetActionsFromPython(EnvStates, EnvDones, EnvTruncs, NeedsAction);

    for (int32 i = 0; i < Environments.Num(); i++)
    {
        if (Pending[i].RepeatCounter == 0) // Only update action if it's the start of a new block
        {
            Pending[i].Action = newActions[i];
        }
    }
}

void ARLRunner::StepEnvironments()
{
    for (int32 i = 0; i < Environments.Num(); i++)
    {
        ABaseEnvironment* Env = Environments[i];
        // If an environment was just reset (bDone or bTrunc was true in the previous CollectTransitions),
        // it's now ready for a new action. Don't skip.
        // The original Done()/Trunc() check here was to skip stepping if an env *is currently* done.
        // This check is still valid.
        if (Env->Done() || Env->Trunc())
        {
            // This case should ideally not happen if an env that is done/trunc
            // was correctly reset and its pending block started.
            // If it is done/trunc here, it means it became so without a corresponding reset,
            // or ResetEnv() didn't clear the flags.
            // However, if ResetEnv clears flags, this is fine.
            continue;
        }

        Env->PreStep();
        Env->Act(Pending[i].Action); // Apply the action decided for this block
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

    // Note: StartNewBlock is now called by CollectTransitions after this function returns.
    MaybeTrainUpdate();
}

void ARLRunner::StartNewBlock(int EnvIndex, const FState& State)
{
    FPendingTransition& p = Pending[EnvIndex];
    p.PrevState = State;
    // Action will be filled by DecideActions if RepeatCounter is 0
    // p.Action = FAction(); // No need to clear here, DecideActions overwrites if necessary
    p.AccumulatedReward = 0.f;
    p.RepeatCounter = 0; // Reset for the new block
    p.bDoneOrTrunc = false; // Reset this flag
}

void ARLRunner::ResetEnvironment(int EnvIndex)
{
    // Determine CurrentAgents before reset, as it can change.
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
        // Ensure CurrentAgents reflects the agent count for the majority of collected experiences
        // or the agent count used for the last reset if it's consistent.
        // For simplicity, we use the globally tracked CurrentAgents. This assumes it's
        // reasonably consistent or that the Python side can handle variability if needed.
        AgentComm->Update(Batches, CurrentAgents);
    }
}

void ARLRunner::ParseActionSpaceFromConfig()
{
    DiscreteActionSizes.Empty();
    ContinuousActionRanges.Empty();
    if (!EnvConfig) return;

    // Path adjusted to be more generic, assuming either "agent" or "central" exists under "action"
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
            if (c->HasPath(TEXT("num_choices"))) // Check if "num_choices" key exists
            {
                int32 n = c->Get(TEXT("num_choices"))->AsInt();
                DiscreteActionSizes.Add(n);
            }
        }
    }

    if (EnvConfig->HasPath(ActionBasePath + TEXT("continuous")))
    {
        auto Node = EnvConfig->Get(ActionBasePath + TEXT("continuous"));
        auto Arr = Node->AsArrayOfConfigs(); // Each element is a config object like {"min": -1, "max": 1}
        for (auto* r : Arr)
        {
            if (r->HasPath(TEXT("min")) && r->HasPath(TEXT("max"))) // Check for keys
            {
                float mn = r->Get(TEXT("min"))->AsNumber();
                float mx = r->Get(TEXT("max"))->AsNumber();
                ContinuousActionRanges.Add(FVector2D(mn, mx));
            }
        }
    }
}

FAction ARLRunner::EnvSample(int EnvIndex) // EnvIndex not used, samples generically
{
    FAction act;
    // For multi-agent, actions need to be generated per agent up to CurrentAgents
    // This simple sampler assumes the action spec is per-agent if multi-agent.
    // If the action spec is global, this needs adjustment. Assuming per-agent for now.
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