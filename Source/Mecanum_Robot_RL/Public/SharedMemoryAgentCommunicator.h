#pragma once

#include "CoreMinimal.h"
#include "ExperienceBuffer.h"
#include "SharedMemoryAgentCommunicator.generated.h"

USTRUCT(BlueprintType)
struct FSharedMemoryAgentCommunicatorConfig
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite, Category = "Config")
    int32 NumEnvironments;

    UPROPERTY(BlueprintReadWrite, Category = "Config")
    int32 NumActions;

    UPROPERTY(BlueprintReadWrite, Category = "Config")
    int32 StateSize;

    UPROPERTY(BlueprintReadWrite, Category = "Config")
    int32 TrainingBatchSize;
};

USTRUCT(BlueprintType)
struct FAction
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite, Category = "Action")
    TArray<float> Values;
};

USTRUCT(BlueprintType)
struct FState
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite, Category = "State")
    TArray<float> Values;
};

UCLASS(BlueprintType)
class MECANUM_ROBOT_RL_API USharedMemoryAgentCommunicator : public UObject
{
    GENERATED_BODY()

private:
    void* ActionsSharedMemoryHandle;
    void* UpdateSharedMemoryHandle;
    void* ActionsMutexHandle;
    void* UpdateMutexHandle;
    void* ActionReadyEventHandle;
    void* ActionReceivedEventHandle;
    void* UpdateReadyEventHandle;
    void* UpdateReceivedEventHandle;

public:
    USharedMemoryAgentCommunicator();

    UFUNCTION(BlueprintCallable, Category = "SharedMemory")
    void Init(FSharedMemoryAgentCommunicatorConfig config);

    UFUNCTION(BlueprintCallable, Category = "SharedMemory")
    TArray<FAction> GetActions(TArray<FState> States);

    UFUNCTION(BlueprintCallable, Category = "SharedMemory")
    void Update(const TArray<FExperienceBatch>& experiences);

    virtual ~USharedMemoryAgentCommunicator();
};
