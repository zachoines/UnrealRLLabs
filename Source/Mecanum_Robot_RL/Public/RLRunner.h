#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"

// Local includes
#include "ExperienceBuffer.h"
#include "VectorEnvironment.h"
#include "SharedMemoryAgentCommunicator.h"
#include "ActionSpace.h"
#include "RLRunner.generated.h"

// Forward declaration to avoid circular includes
class UEnvironmentConfig;
struct FBaseInitParams;

/**
 * The RLRunner is responsible for coordinating:
 *   - The VectorEnvironment (multiple Env instances)
 *   - Communication with Python-based RL (via USharedMemoryAgentCommunicator)
 *   - Collecting experiences / training updates
 *   - Handling multi-agent or single-agent setups
 */
UCLASS(Blueprintable, BlueprintType)
class UNREALRLLABS_API ARLRunner : public AActor
{
    GENERATED_BODY()

public:
    // Sets default values for this actor's properties
    ARLRunner();

    // Called every frame
    virtual void Tick(float DeltaTime) override;

    /**
     * Initialize the runner:
     *   1) Spawns the VectorEnvironment with the specified environment class + init-params
     *   2) Reads necessary config data from EnvConfig
     *   3) Sets up communicator / experience buffer / multi-agent logic
     */
    void InitRunner(
        TSubclassOf<ABaseEnvironment> EnvironmentClass,
        TArray<FBaseInitParams*> ParamsArray,
        UEnvironmentConfig* InEnvConfig
    );

    /**
     * Request actions from the Python side, or sample random if not connected
     */
    TArray<FAction> GetActions(TArray<FState> States, TArray<float> Dones, TArray<float> Truncs);

    /**
     * Add a batch of experiences to the replay buffer
     */
    void AddExperiences(const TArray<FExperienceBatch>& EnvironmentTrajectories);

    /**
     * Sample experiences from the buffer
     */
    TArray<FExperienceBatch> SampleExperiences(int bSize);

    /**
     * The current number of agents in each environment (if multi-agent).
     */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Train Params")
    int CurrentAgents;

private:

    /** Pointer to the vector environment that manages multiple environment instances. */
    UPROPERTY()
    AVectorEnvironment* VectorEnvironment;

    /** The replay/experience buffer */
    UPROPERTY()
    UExperienceBuffer* ExperienceBufferInstance;

    /** Shared memory communicator to Python-based RL */
    UPROPERTY()
    USharedMemoryAgentCommunicator* AgentComm;

    /** A pointer to our JSON-based environment config. */
    UPROPERTY()
    UEnvironmentConfig* EnvConfig;

    /** Tracking counters */
    unsigned long int CurrentStep;
    unsigned long int CurrentUpdate;

    /** Multi-agent related parameters */
    int MinAgents;
    int MaxAgents;
    bool IsMultiAgent;

    /** Replay buffer capacity and batch size */
    int BufferSize;
    int BatchSize;

    /** Number of steps to keep repeating the same action */
    int ActionRepeat;

    /** Current repetition count */
    int ActionRepeatCounter;

    /** The last-chosen actions */
    TArray<FAction> Actions;
};
