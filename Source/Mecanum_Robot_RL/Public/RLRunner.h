#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"

#include "BaseEnvironment.h"
#include "ExperienceBuffer.h"
#include "SharedMemoryAgentCommunicator.h"
#include "ActionSpace.h"
#include "EnvironmentConfig.h"
#include "RLTypes.h"
#include "RLRunner.generated.h"

/** Pending transition data used while action repeat is in effect. */
USTRUCT()
struct FPendingTransition
{
    GENERATED_BODY()

    FState PrevState;
    FAction Action;
    float AccumulatedReward;
    int32 RepeatCounter;
    bool  bDoneOrTrunc;

    FPendingTransition()
        : AccumulatedReward(0.f)
        , RepeatCounter(0)
        , bDoneOrTrunc(false)
    {}
};

/** Actor responsible for running TerraShift environments and bridging to Python. */
UCLASS(Blueprintable, BlueprintType)
class UNREALRLLABS_API ARLRunner : public AActor
{
    GENERATED_BODY()

public:
    ARLRunner();

    virtual void Tick(float DeltaTime) override;

    /** Initialize runner: spawn envs, set up communicator, parse config, etc. */
    void InitRunner(
        TSubclassOf<ABaseEnvironment> EnvironmentClass,
        TArray<FBaseInitParams*> ParamsArray,
        UEnvironmentConfig* InEnvConfig
    );

private:
    /** Finalizes transitions once environments report new rewards and termination flags. */
    void CollectTransitions();

    /** Requests actions from Python for environments whose repeat counters expired. */
    void DecideActions();

    /** Applies pending actions to each environment and advances them one step. */
    void StepEnvironments();

    UPROPERTY()
    TArray<ABaseEnvironment*> Environments;

    bool bIsMultiAgent;
    int32 MinAgents;
    int32 MaxAgents;
    int32 CurrentAgents;

    UPROPERTY()
    UExperienceBuffer* ExperienceBufferInstance;

    UPROPERTY()
    USharedMemoryAgentCommunicator* AgentComm;

    UPROPERTY()
    UEnvironmentConfig* EnvConfig;

    int32 BufferSize;
    int32 BatchSize;
    int32 ActionRepeat;

    TArray<int32> DiscreteActionSizes;
    TArray<FVector2D> ContinuousActionRanges;

    TArray<FPendingTransition> Pending;

    uint64 CurrentStep;
    uint64 CurrentUpdate;

    bool bTestingMode;

    void BeginTestMode();
    void EndTestMode();

private:
    void ParseActionSpaceFromConfig();
    FAction EnvSample(int EnvIndex);

    /** Returns the current state for an environment (multi-agent combined or single). */
    FState GetEnvState(ABaseEnvironment* Env);

    /** Resets an environment and seeds its pending transition. */
    void ResetEnvironment(int EnvIndex);

    /** Starts a new transition block for an environment. */
    void StartNewBlock(int EnvIndex, const FState& State);

    /** Finalizes the pending block and adds it to the experience buffer. */
    void FinalizeTransition(int EnvIndex, const FState& NextState, bool bDone, bool bTrunc);

    /** Triggers a training update when each environment has enough transitions buffered. */
    void MaybeTrainUpdate();

    /** Pulls actions from Python or samples fallbacks when communication fails. */
    TArray<FAction> GetActionsFromPython(const TArray<FState>& EnvStates, const TArray<float>& EnvDones, const TArray<float>& EnvTruncs, const TArray<float>& NeedsAction);
    TArray<FAction> SampleAllEnvActions();
};
