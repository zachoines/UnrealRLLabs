#pragma once

#include "CoreMinimal.h"
#include "Windows/AllowWindowsPlatformTypes.h"
#include "Windows.h"
#include "Windows/HideWindowsPlatformTypes.h"
#include "ActionSpace.h"
#include "BaseEnvironment.h"
#include "ExperienceBuffer.h"
#include "RLTypes.h"
#include "SharedMemoryAgentCommunicator.generated.h"

UCLASS(BlueprintType)
class MECANUM_ROBOT_RL_API USharedMemoryAgentCommunicator : public UObject
{
    GENERATED_BODY()

public:
    FTrainParams params;
    FEnvInfo info;
    TArray<char> ConfigJSON;
 
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

    void WriteConfigToSharedMemory();

    TArray<char> WriteConfigInfoToJson(const FEnvInfo& EnvInfo, const FTrainParams& TrainParams);

public:
    USharedMemoryAgentCommunicator();

    UFUNCTION(BlueprintCallable, Category = "SharedMemory")
    void Init(FEnvInfo EnvInfo, FTrainParams TrainParams);

    UFUNCTION(BlueprintCallable, Category = "SharedMemory")
    TArray<FAction> GetActions(TArray<FState> States, TArray<float> Dones, TArray<float> Truncs, int NumAgents);

    UFUNCTION(BlueprintCallable, Category = "SharedMemory")
    void Update(const TArray<FExperienceBatch>& experiences, int NumAgents);

    UFUNCTION(BlueprintCallable, Category = "SharedMemory")
    void PrintActionsAsMatrix(const TArray<FAction>& Actions);

    virtual ~USharedMemoryAgentCommunicator();
};
