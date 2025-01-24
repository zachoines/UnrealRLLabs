#pragma once

#include "CoreMinimal.h"
#include "Windows/AllowWindowsPlatformTypes.h"
#include "Windows.h"
#include "Windows/HideWindowsPlatformTypes.h"

#include "BaseEnvironment.h"
#include "ExperienceBuffer.h"
#include "EnvironmentConfig.h"
#include "SharedMemoryAgentCommunicator.generated.h"

/**
 * SharedMemoryAgentCommunicator:
 * - Allocates & manages shared memory for exchanging data with a Python RL process
 * - Sends config info, obtains actions, sends experiences, etc.
 */
UCLASS(BlueprintType)
class UNREALRLLABS_API USharedMemoryAgentCommunicator : public UObject
{
    GENERATED_BODY()

public:
    USharedMemoryAgentCommunicator();

    /**
     * Initialize the communicator by:
     *   1) Parsing relevant data from EnvConfig (NumEnvironments, ActionSpace, etc.)
     *   2) Creating shared memory segments / events / mutexes
     *   3) Writing config (in JSON form) to shared memory
     */
    UFUNCTION(BlueprintCallable, Category = "SharedMemory")
    void Init(UEnvironmentConfig* EnvConfig);

    /**
     * Called each step to get actions from the Python side.
     * (We copy states, dones, truncs into shared memory, signal python, block until python sets actions)
     */
    UFUNCTION(BlueprintCallable, Category = "SharedMemory")
    TArray<FAction> GetActions(TArray<FState> States, TArray<float> Dones, TArray<float> Truncs, int NumAgents);

    /**
     * Update the python side with newly collected experiences
     */
    UFUNCTION(BlueprintCallable, Category = "SharedMemory")
    void Update(const TArray<FExperienceBatch>& experiences, int NumAgents);

    /**
     * Debug: print the matrix of actions to log
     */
    UFUNCTION(BlueprintCallable, Category = "SharedMemory")
    void PrintActionsAsMatrix(const TArray<FAction>& Actions);

    virtual ~USharedMemoryAgentCommunicator();

private:
    /** The JSON representation of the config that we share with python. */
    TArray<char> ConfigJSON;

    /** Shared memory + mapped pointers for actions, states, updates, config. */
    void* StatesSharedMemoryHandle = nullptr;
    void* ActionsSharedMemoryHandle = nullptr;
    void* UpdateSharedMemoryHandle = nullptr;
    void* ConfigSharedMemoryHandle = nullptr;

    float* MappedStatesSharedData = nullptr;
    float* MappedActionsSharedData = nullptr;
    float* MappedUpdateSharedData = nullptr;
    int32* MappedConfigSharedData = nullptr;

    /** Mutexes and events for synchronization. */
    void* ActionsMutexHandle = nullptr;
    void* UpdateMutexHandle = nullptr;
    void* StatesMutexHandle = nullptr;
    void* ConfigMutexHandle = nullptr;

    void* ActionReadyEventHandle = nullptr;
    void* ActionReceivedEventHandle = nullptr;
    void* UpdateReadyEventHandle = nullptr;
    void* UpdateReceivedEventHandle = nullptr;
    void* ConfigReadyEventHandle = nullptr;

private:
    /** Creates the shared memory mapping + events. */
    void CreateSharedMemoryAndEvents(
        int32 ActionSizeBytes,
        int32 StatesSizeBytes,
        int32 UpdateSizeBytes,
        int32 ConfigSizeBytes
    );

    /** Write the config data (ConfigJSON) into the config shared memory segment. */
    void WriteConfigToSharedMemory();

    /**
     * Creates a JSON object from the environment config, summarizing:
     *   - Env ID
     *   - IsMultiAgent, MaxAgents
     *   - ActionSpace (discrete + continuous)
     *   - Observations (approx. size)
     *   - Training info: num_envs, buffer_size, batch_size, etc.
     */
    TArray<char> WriteConfigInfoToJson(UEnvironmentConfig* EnvConfig);

    /** Helper to compute total #actions for 1 agent from the discrete/continuous specs. */
    int32 ComputeSingleAgentNumActions(UEnvironmentConfig* EnvConfig) const;

    /** Helper to guess the single-agent obs size or total state size. (You can refine this.) */
    int32 ComputeSingleAgentObsSize(UEnvironmentConfig* EnvConfig) const;
};
