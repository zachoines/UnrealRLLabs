#include "SharedMemoryAgentCommunicator.h"
#include "Windows/AllowWindowsPlatformTypes.h"
#include <Windows.h>
#include "Windows/HideWindowsPlatformTypes.h"

USharedMemoryAgentCommunicator::USharedMemoryAgentCommunicator()
    : ActionsSharedMemoryHandle(NULL), UpdateSharedMemoryHandle(NULL),
    ActionsMutexHandle(NULL), UpdateMutexHandle(NULL),
    ActionReadyEventHandle(NULL), ActionReceivedEventHandle(NULL),
    UpdateReadyEventHandle(NULL), UpdateReceivedEventHandle(NULL)
{
}

void USharedMemoryAgentCommunicator::Init(FSharedMemoryAgentCommunicatorConfig config)
{
    int32 ActionTotalSize = config.NumEnvironments * config.NumActions * sizeof(float);
    int32 UpdateTotalSize = config.NumEnvironments * config.TrainingBatchSize * (config.StateSize + config.NumActions + 2) * sizeof(float);

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

    ActionsMutexHandle = CreateMutex(NULL, false, TEXT("ActionsDataMutex"));
    if (!ActionsMutexHandle)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create actions mutex."));
    }

    UpdateMutexHandle = CreateMutex(NULL, false, TEXT("UpdateDataMutex"));
    if (!UpdateMutexHandle)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create update mutex."));
    }

    ActionReadyEventHandle = CreateEvent(NULL, false, false, TEXT("ActionReadyEvent"));
    ActionReceivedEventHandle = CreateEvent(NULL, false, false, TEXT("ActionReceivedEvent"));
    UpdateReadyEventHandle = CreateEvent(NULL, false, false, TEXT("UpdateReadyEvent"));
    UpdateReceivedEventHandle = CreateEvent(NULL, false, false, TEXT("UpdateReceivedEvent"));
}

TArray<FAction> USharedMemoryAgentCommunicator::GetActions(TArray<FState> States)
{
    TArray<FAction> Actions;

    float* SharedData = (float*)MapViewOfFile(ActionsSharedMemoryHandle, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (!SharedData)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to map view of shared memory for actions."));
        return Actions;
    }

    for (const FState& state : States)
    {
        FMemory::Memcpy(SharedData, state.Values.GetData(), state.Values.Num() * sizeof(float));
        SharedData += state.Values.Num();
    }

    SetEvent(ActionReadyEventHandle);
    WaitForSingleObject(ActionReceivedEventHandle, INFINITE);

    for (int32 i = 0; i < States.Num(); ++i)
    {
        FAction action;
        action.Values.SetNumZeroed(States[i].Values.Num());
        FMemory::Memcpy(action.Values.GetData(), SharedData, action.Values.Num() * sizeof(float));
        Actions.Add(action);
        SharedData += action.Values.Num();
    }

    UnmapViewOfFile(SharedData);
    ReleaseMutex(ActionsMutexHandle);

    return Actions;
}

void USharedMemoryAgentCommunicator::Update(const TArray<FExperienceBatch>& experiences)
{
    float* SharedData = (float*)MapViewOfFile(UpdateSharedMemoryHandle, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (!SharedData)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to map view of shared memory for update."));
        return;
    }

    for (const FExperienceBatch& batch : experiences)
    {
        FMemory::Memcpy(SharedData, batch.Experiences.GetData(), batch.Experiences.Num() * sizeof(float));
        SharedData += batch.Experiences.Num();
    }

    SetEvent(UpdateReadyEventHandle);
    WaitForSingleObject(UpdateReceivedEventHandle, INFINITE);

    UnmapViewOfFile(SharedData);
    ReleaseMutex(UpdateMutexHandle);
}

USharedMemoryAgentCommunicator::~USharedMemoryAgentCommunicator()
{
    if (ActionsSharedMemoryHandle)
        CloseHandle(ActionsSharedMemoryHandle);
    if (UpdateSharedMemoryHandle)
        CloseHandle(UpdateSharedMemoryHandle);
    if (ActionsMutexHandle)
        CloseHandle(ActionsMutexHandle);
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
