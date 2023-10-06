#pragma once

#include "CoreMinimal.h"
#include "Windows/AllowWindowsPlatformTypes.h"
#include "Windows.h"
#include "Windows/HideWindowsPlatformTypes.h"
#include "ActionSpace.h"
#include "BaseEnvironment.h"
#include "ExperienceBuffer.h"
#include "SharedMemoryAgentCommunicator.generated.h"

USTRUCT(BlueprintType)
struct FSharedMemoryAgentCommunicatorConfig
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite, Category = "Config")
    int NumEnvironments;

    UPROPERTY(BlueprintReadWrite, Category = "Config")
    int NumActions;

    UPROPERTY(BlueprintReadWrite, Category = "Config")
    int StateSize;

    UPROPERTY(BlueprintReadWrite, Category = "Config")
    int BatchSize;
};

UCLASS(BlueprintType)
class MECANUM_ROBOT_RL_API USharedMemoryAgentCommunicator : public UObject
{
    GENERATED_BODY()

private:
    void* StatesSharedMemoryHandle;
    void* ActionsSharedMemoryHandle;
    void* UpdateSharedMemoryHandle;
    void* ConfigSharedMemoryHandle;

    float* MappedStatesSharedData;
    float* MappedActionsSharedData;
    float* MappedUpdateSharedData;
    int32* MappedConfigSharedData;

    void* ActionsMutexHandle;
    void* UpdateMutexHandle;
    void* StatesMutexHandle;
    void* ConfigMutexHandle; 

    void* ActionReadyEventHandle;
    void* ActionReceivedEventHandle;
    void* UpdateReadyEventHandle;
    void* UpdateReceivedEventHandle;
    void* ConfigReadyEventHandle;

    FSharedMemoryAgentCommunicatorConfig config;

    void WriteConfigToSharedMemory();

public:
    USharedMemoryAgentCommunicator();

    UFUNCTION(BlueprintCallable, Category = "SharedMemory")
    void Init(FSharedMemoryAgentCommunicatorConfig Config);

    UFUNCTION(BlueprintCallable, Category = "SharedMemory")
    TArray<FAction> GetActions(TArray<FState> States);

    UFUNCTION(BlueprintCallable, Category = "SharedMemory")
    void Update(const TArray<FExperienceBatch>& experiences);

    UFUNCTION(BlueprintCallable, Category = "SharedMemory")
    void PrintActionsAsMatrix(const TArray<FAction>& Actions);

    virtual ~USharedMemoryAgentCommunicator();
};
