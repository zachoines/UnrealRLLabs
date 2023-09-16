#include "SharedMemoryAgentCommunicator.h"
#include "Windows/AllowWindowsPlatformTypes.h"
#include <Windows.h>
#include "Windows/HideWindowsPlatformTypes.h"

USharedMemoryAgentCommunicator::USharedMemoryAgentCommunicator()
    : ActionsSharedMemoryHandle(NULL), UpdateSharedMemoryHandle(NULL),
    ActionsMutexHandle(NULL), UpdateMutexHandle(NULL),
    ActionReadyEventHandle(NULL), ActionReceivedEventHandle(NULL),
    UpdateReadyEventHandle(NULL), UpdateReceivedEventHandle(NULL),
    MappedActionsSharedData(NULL), MappedStatesSharedData(NULL), MappedUpdateSharedData(NULL)
{
}

void USharedMemoryAgentCommunicator::Init(FSharedMemoryAgentCommunicatorConfig Config)
{
    config = Config;

    int32 ActionTotalSize = config.NumEnvironments * config.NumActions * sizeof(float);
    int32 StatesTotalSize = config.NumEnvironments * config.StateSize * sizeof(float);
    int32 UpdateTotalSize = config.NumEnvironments * config.TrainingBatchSize * (config.StateSize + config.NumActions + 2) * sizeof(float);

    // Creating shared memory for actions
    ActionsSharedMemoryHandle = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        ActionTotalSize,
        TEXT("ActionsSharedMemory")
    );
    if (!ActionsSharedMemoryHandle)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create actions shared memory."));
    }

    // Creating shared memory for states
    StatesSharedMemoryHandle = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        StatesTotalSize,
        TEXT("StatesSharedMemory")
    );
    if (!StatesSharedMemoryHandle)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create states shared memory."));
    }

    // Creating shared memory for update
    UpdateSharedMemoryHandle = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        UpdateTotalSize,
        TEXT("UpdateSharedMemory")
    );
    if (!UpdateSharedMemoryHandle)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create update shared memory."));
    }

    // Initializing synchronization objects
    ActionsMutexHandle = CreateMutex(NULL, false, TEXT("ActionsDataMutex"));
    if (!ActionsMutexHandle)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create actions mutex."));
    }

    StatesMutexHandle = CreateMutex(NULL, false, TEXT("StatesDataMutex"));
    if (!StatesMutexHandle)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create states mutex."));
    }

    UpdateMutexHandle = CreateMutex(NULL, false, TEXT("UpdateDataMutex"));
    if (!UpdateMutexHandle)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create update mutex."));
    }

    // Creating event objects
    ActionReadyEventHandle = CreateEvent(NULL, false, false, TEXT("ActionReadyEvent"));
    ActionReceivedEventHandle = CreateEvent(NULL, false, false, TEXT("ActionReceivedEvent"));
    UpdateReadyEventHandle = CreateEvent(NULL, false, false, TEXT("UpdateReadyEvent"));
    UpdateReceivedEventHandle = CreateEvent(NULL, false, false, TEXT("UpdateReceivedEvent"));

    // Mapping memory regions
    MappedActionsSharedData = (float*)MapViewOfFile(ActionsSharedMemoryHandle, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (!MappedActionsSharedData)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to map view of shared memory for actions."));
    }

    MappedStatesSharedData = (float*)MapViewOfFile(StatesSharedMemoryHandle, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (!MappedStatesSharedData)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to map view of shared memory for states."));
    }

    MappedUpdateSharedData = (float*)MapViewOfFile(UpdateSharedMemoryHandle, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (!MappedUpdateSharedData)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to map view of shared memory for update."));
    }
}

TArray<FAction> USharedMemoryAgentCommunicator::GetActions(TArray<FState> States)
{
    TArray<FAction> Actions;

    float* StateSharedData = MappedStatesSharedData;
    if (!StateSharedData)
    {
        UE_LOG(LogTemp, Error, TEXT("Mapped memory for states is not available."));
        return Actions;
    }

    for (const FState& state : States)
    {
        FMemory::Memcpy(StateSharedData, state.Values.GetData(), state.Values.Num() * sizeof(float));
        StateSharedData += state.Values.Num();
    }

    SetEvent(ActionReadyEventHandle);
    WaitForSingleObject(ActionReceivedEventHandle, INFINITE);

    float* ActionSharedData = MappedActionsSharedData;
    if (!ActionSharedData)
    {
        UE_LOG(LogTemp, Error, TEXT("Mapped memory for actions is not available."));
        return Actions;
    }

    for (int32 i = 0; i < States.Num(); ++i)
    {
        FAction action;
        action.Values.SetNumZeroed(this->config.NumActions);
        FMemory::Memcpy(action.Values.GetData(), ActionSharedData, action.Values.Num() * sizeof(float));
        Actions.Add(action);
        ActionSharedData += action.Values.Num();
    }

    ReleaseMutex(ActionsMutexHandle);

    return Actions;
}

void USharedMemoryAgentCommunicator::Update(const TArray<FExperienceBatch>& experiences)
{
    WaitForSingleObject(UpdateMutexHandle, INFINITE);

    if (!MappedUpdateSharedData)
    {
        UE_LOG(LogTemp, Error, TEXT("Mapped memory for update is not available."));
        ReleaseMutex(UpdateMutexHandle);
        return;
    }

    float* SharedData = MappedUpdateSharedData;
    for (const FExperienceBatch& batch : experiences)
    {
        FMemory::Memcpy(SharedData, batch.Experiences.GetData(), batch.Experiences.Num() * sizeof(float));
        SharedData += batch.Experiences.Num();
    }

    SetEvent(UpdateReadyEventHandle);
    WaitForSingleObject(UpdateReceivedEventHandle, INFINITE);

    ReleaseMutex(UpdateMutexHandle);
}

USharedMemoryAgentCommunicator::~USharedMemoryAgentCommunicator()
{
    if (MappedActionsSharedData)
        UnmapViewOfFile(MappedActionsSharedData);
    if (MappedStatesSharedData)
        UnmapViewOfFile(MappedStatesSharedData);
    if (MappedUpdateSharedData)
        UnmapViewOfFile(MappedUpdateSharedData);

    if (ActionsSharedMemoryHandle)
        CloseHandle(ActionsSharedMemoryHandle);
    if (StatesSharedMemoryHandle)
        CloseHandle(StatesSharedMemoryHandle);
    if (UpdateSharedMemoryHandle)
        CloseHandle(UpdateSharedMemoryHandle);

    if (ActionsMutexHandle)
        CloseHandle(ActionsMutexHandle);
    if (StatesMutexHandle)
        CloseHandle(StatesMutexHandle);
    if (UpdateMutexHandle)
        CloseHandle(UpdateMutexHandle);

    if (ActionReadyEventHandle)
        CloseHandle(ActionReadyEventHandle);
    if (ActionReceivedEventHandle)
        CloseHandle(ActionReceivedEventHandle);
    if (UpdateReadyEventHandle)
        CloseHandle(UpdateReadyEventHandle);
    if (UpdateReceivedEventHandle)
        CloseHandle(UpdateReceivedEventHandle);
}


void USharedMemoryAgentCommunicator::PrintActionsAsMatrix(const TArray<FAction>& Actions)
{
    if (Actions.Num() > 0)
    {
        FString matrixStr = TEXT("[\n");

        for (int32 envIdx = 0; envIdx < Actions.Num(); envIdx++)
        {
            matrixStr += TEXT("   [");

            for (int32 actionIdx = 0; actionIdx < Actions[envIdx].Values.Num(); actionIdx++)
            {
                matrixStr += FString::Printf(TEXT("%s"), *FString::SanitizeFloat(Actions[envIdx].Values[actionIdx]));

                if (actionIdx != Actions[envIdx].Values.Num() - 1)
                {
                    matrixStr += TEXT(", ");
                }
            }

            matrixStr += TEXT("]");
            if (envIdx != Actions.Num() - 1)
            {
                matrixStr += TEXT(",\n");
            }
            else
            {
                matrixStr += TEXT("\n");
            }
        }
        matrixStr += TEXT("]");

        UE_LOG(LogTemp, Warning, TEXT("%s"), *matrixStr);
    }
}