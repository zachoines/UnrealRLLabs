// NOTICE: This file includes modifications generated with the assistance of generative AI.
// Original code structure and logic by the project author.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"

#include "Windows/AllowWindowsPlatformTypes.h"
#include "Windows.h"
#include "Windows/HideWindowsPlatformTypes.h"

#include "BaseEnvironment.h"
#include "ExperienceBuffer.h"
#include "RLTypes.h"
#include "EnvironmentConfig.h"
#include "SharedMemoryAgentCommunicator.generated.h"

/**
 * Communicates with a Python-based RL process via shared memory.
 */
UCLASS(BlueprintType)
class UNREALRLLABS_API USharedMemoryAgentCommunicator : public UObject
{
    GENERATED_BODY()

public:
    USharedMemoryAgentCommunicator();

    /**
     * Initialize the communicator:
     *  1) Reads relevant sizes from EnvConfig
     *  2) Allocates shared memory (actions/states/update)
     *  3) Sets up sync objects (mutexes/events)
     *  4) Stores NumEnvironments so we can reuse later
     */
    UFUNCTION(BlueprintCallable, Category = "SharedMemory")
    void Init(UEnvironmentConfig* EnvConfig);

    /**
     * Request actions from the Python side:
     *   - Writes states/dones/truncs to shared memory
     *   - Waits for the Python process to compute actions
     *   - Reads actions back from shared memory
     */
    UFUNCTION(BlueprintCallable, Category = "SharedMemory")
    TArray<FAction> GetActions(TArray<FState> States, TArray<float> Dones, TArray<float> Truncs, TArray<float> NeedsAction, int NumAgents);

    /**
     * Send a batch of experiences for training updates:
     *   - Writes transitions to shared memory
     *   - Signals the Python side
     *   - Waits for confirmation
     */
    UFUNCTION(BlueprintCallable, Category = "SharedMemory")
    void Update(const TArray<FExperienceBatch>& experiences, int NumAgents);

    virtual ~USharedMemoryAgentCommunicator();

    void WriteTransitionsToFile(const TArray<FExperienceBatch>& experiences, const FString& FilePath);

private:
    // ------------------- SHARED MEMORY HANDLES -------------------
    void* StatesSharedMemoryHandle;
    void* ActionsSharedMemoryHandle;
    void* UpdateSharedMemoryHandle;

    // ------------------- MAPPED POINTERS -------------------------
    float* MappedStatesSharedData;
    float* MappedActionsSharedData;
    float* MappedUpdateSharedData;

    // ------------------- SYNCHRONIZATION -------------------------
    void* ActionsMutexHandle;
    void* UpdateMutexHandle;
    void* StatesMutexHandle;

    void* ActionReadyEventHandle;
    void* ActionReceivedEventHandle;
    void* UpdateReadyEventHandle;
    void* UpdateReceivedEventHandle;
    void* BeginTestEventHandle;
    void* EndTestEventHandle;

    // ------------------- CONFIG / SIZING -------------------------
    UPROPERTY()
    UEnvironmentConfig* LocalEnvConfig;

    int32 NumEnvironments;       // from "train/num_environments"
    int32 BufferSize;            // from "train/buffer_size"
    int32 BatchSize;             // from "train/batch_size"

    /**
     * Maximum single-env state size (for the max # agents if multi-agent).
     * Used for sizing the memory blocks.
     */
    int32 SingleEnvStateSize;

    /**
     * Maximum single-env action size (for the max # agents if multi-agent).
     */
    int32 TotalActionCount;

    // Alloc sizes in bytes
    int32 ActionMAXSize;
    int32 StatesMAXSize;
    int32 UpdateMAXSize;

private:
    // Helper: check if config has "environment/shape/state/agent"
    bool IsMultiAgent() const;

    // Helper: check if config has "environment/shape/state/central"
    bool HasCentralState() const;

    // Helper: sum central + agent obs_size for a given agent count
    int32 ComputeSingleEnvStateSize(int32 NumAgents) const;

    // Helper: read discrete + continuous action arrays from config, sum them up
    int32 ComputeSingleEnvActionSize(int32 NumAgents) const;

public:
    void* GetBeginTestEventHandle() const { return BeginTestEventHandle; }
    void* GetEndTestEventHandle() const { return EndTestEventHandle; }
};
