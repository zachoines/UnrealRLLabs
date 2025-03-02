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

/**
 * Holds “pending” data for each environment with action-repeat:
 *   - The last state from the beginning of the repeated-block
 *   - The chosen action
 *   - Accumulated reward
 *   - Action repeat counter
 */
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

/**
 * The RLRunner orchestrates multiple environment instances for TerraShift,
 * each with asynchronous action repeat, bridging states & actions
 * with the Python RL side (SharedMemory).
 *
 * It adds transitions to the new UExperienceBuffer,
 * then triggers `AgentComm->Update()` once each environment has at least `batch_size` transitions.
 */
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
    // --------------------------
    //   MAIN FLOW
    // --------------------------

    /**
     * Called AFTER the environment has stepped at least once.
     * Gathers (reward, done, trunc) => finalizes transitions if needed,
     * and resets environment if done/trunc.
     */
    void CollectTransitions();

    /**
     * For environment that needs new actions (repeatCounter==0),
     * gather new actions from Python or random fallback.
     */
    void DecideActions();

    /**
     * Actually apply the current pending actions to each environment => PreStep->Act->PostStep
     */
    void StepEnvironments();

    // Each environment is stored in this array
    UPROPERTY()
    TArray<ABaseEnvironment*> Environments;

    // For multi-agent
    bool bIsMultiAgent;
    int32 MinAgents;
    int32 MaxAgents;
    int32 CurrentAgents;

    // Shared buffer
    UPROPERTY()
    UExperienceBuffer* ExperienceBufferInstance;

    // Communicator to Python
    UPROPERTY()
    USharedMemoryAgentCommunicator* AgentComm;

    // Config
    UPROPERTY()
    UEnvironmentConfig* EnvConfig;

    // Training hyperparams
    int32 BufferSize;
    int32 BatchSize;
    int32 ActionRepeat;

    // For building fallback random actions
    TArray<int32> DiscreteActionSizes;
    TArray<FVector2D> ContinuousActionRanges;

    // Per-environment pending transitions
    TArray<FPendingTransition> Pending;

    // Counters
    uint64 CurrentStep;
    uint64 CurrentUpdate;

private:
    // --------------------------
    //   Internal Helpers
    // --------------------------
    void ParseActionSpaceFromConfig();
    FAction EnvSample(int EnvIndex);

    /** Return environment's current state (multi-agent combined or single). */
    FState GetEnvState(ABaseEnvironment* Env);

    /** Reset environment i => store new state in the pending struct */
    void ResetEnvironment(int EnvIndex);

    /** Start a new block for environment i after finishing old. */
    void StartNewBlock(int EnvIndex, const FState& State);

    /** Finalize the pending block => create an FExperience => add to buffer => maybe train update. */
    void FinalizeTransition(int EnvIndex, const FState& NextState, bool bDone, bool bTrunc);

    /** Possibly do a training update => if all envs have at least BatchSize experiences. */
    void MaybeTrainUpdate();

    /** Gather actions from Python or fallback random. */
    TArray<FAction> GetActionsFromPython(const TArray<FState>& EnvStates, const TArray<float>& EnvDones, const TArray<float>& EnvTruncs);
    TArray<FAction> SampleAllEnvActions();
};
