#include "SharedMemoryAgentCommunicator.h"
#include "Misc/FileHelper.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"

USharedMemoryAgentCommunicator::USharedMemoryAgentCommunicator()
    : StatesSharedMemoryHandle(nullptr)
    , ActionsSharedMemoryHandle(nullptr)
    , UpdateSharedMemoryHandle(nullptr)
    , MappedStatesSharedData(nullptr)
    , MappedActionsSharedData(nullptr)
    , MappedUpdateSharedData(nullptr)
    , ActionsMutexHandle(nullptr)
    , UpdateMutexHandle(nullptr)
    , StatesMutexHandle(nullptr)
    , ActionReadyEventHandle(nullptr)
    , ActionReceivedEventHandle(nullptr)
    , UpdateReadyEventHandle(nullptr)
    , UpdateReceivedEventHandle(nullptr)
    , LocalEnvConfig(nullptr)
    , NumEnvironments(1)
    , BufferSize(256)
    , BatchSize(256)
    , SingleEnvStateSize(0)
    , TotalActionCount(0)
    , ActionMAXSize(0)
    , StatesMAXSize(0)
    , UpdateMAXSize(0)
{
}

USharedMemoryAgentCommunicator::~USharedMemoryAgentCommunicator()
{
    // Unmap the file views
    if (MappedActionsSharedData)
        UnmapViewOfFile(MappedActionsSharedData);
    if (MappedStatesSharedData)
        UnmapViewOfFile(MappedStatesSharedData);
    if (MappedUpdateSharedData)
        UnmapViewOfFile(MappedUpdateSharedData);

    // Close shared memory handles
    if (ActionsSharedMemoryHandle)
        CloseHandle(ActionsSharedMemoryHandle);
    if (StatesSharedMemoryHandle)
        CloseHandle(StatesSharedMemoryHandle);
    if (UpdateSharedMemoryHandle)
        CloseHandle(UpdateSharedMemoryHandle);

    // Close mutexes
    if (ActionsMutexHandle)
        CloseHandle(ActionsMutexHandle);
    if (StatesMutexHandle)
        CloseHandle(StatesMutexHandle);
    if (UpdateMutexHandle)
        CloseHandle(UpdateMutexHandle);

    // Close events
    if (ActionReadyEventHandle)
        CloseHandle(ActionReadyEventHandle);
    if (ActionReceivedEventHandle)
        CloseHandle(ActionReceivedEventHandle);
    if (UpdateReadyEventHandle)
        CloseHandle(UpdateReadyEventHandle);
    if (UpdateReceivedEventHandle)
        CloseHandle(UpdateReceivedEventHandle);
}

void USharedMemoryAgentCommunicator::Init(UEnvironmentConfig* EnvConfig)
{
    LocalEnvConfig = EnvConfig;
    if (!LocalEnvConfig)
    {
        UE_LOG(LogTemp, Error, TEXT("SharedMemoryAgentCommunicator::Init - EnvConfig is null!"));
        return;
    }

    // 1) Read from config
    if (LocalEnvConfig->HasPath(TEXT("train/buffer_size")))
    {
        BufferSize = LocalEnvConfig->Get(TEXT("train/buffer_size"))->AsInt();
    }
    if (LocalEnvConfig->HasPath(TEXT("train/batch_size")))
    {
        BatchSize = LocalEnvConfig->Get(TEXT("train/batch_size"))->AsInt();
    }
    if (LocalEnvConfig->HasPath(TEXT("train/num_environments")))
    {
        NumEnvironments = LocalEnvConfig->Get(TEXT("train/num_environments"))->AsInt();
    }

    // 2) Compute maximum single-env state & action size 
    int32 MaxAgents = 1;
    if (IsMultiAgent())
    {
        MaxAgents = LocalEnvConfig->Get(TEXT("environment/shape/state/agent/max"))->AsInt();
    }

    SingleEnvStateSize = ComputeSingleEnvStateSize(MaxAgents);
    TotalActionCount = ComputeSingleEnvActionSize(MaxAgents);

    // 3) Derive memory sizes (in bytes)
    // (a) Actions
    ActionMAXSize = NumEnvironments * (TotalActionCount * sizeof(float));

    // (b) States => includes overhead for 6 floats + done/trunc
    // We'll keep a bit of an offset for info floats, but for simplicity:
    {
        int32 InfoSize = 6 * sizeof(float);
        int32 TerminalsSize = NumEnvironments * 2 * sizeof(float); // done + trunc
        StatesMAXSize = NumEnvironments * (SingleEnvStateSize * sizeof(float))
            + InfoSize
            + TerminalsSize;
    }

    // (c) Update => each transition: [state + nextState + action + reward + trunc + done]
    // Single transition = (SingleEnvStateSize*2) + TotalActionCount + 3
    // ( done/trunc as single floats each + reward=1)
    {
        int32 DoneSize = 1;
        int32 TruncSize = 1;
        int32 RewardSize = 1;
        int32 SingleTransitionFloatCount = (SingleEnvStateSize * 2)
            + TotalActionCount
            + (DoneSize + TruncSize + RewardSize);
        UpdateMAXSize = NumEnvironments * BatchSize * SingleTransitionFloatCount * sizeof(float);
    }

    // 4) Create the shared memory sections
    ActionsSharedMemoryHandle = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        ActionMAXSize,
        TEXT("ActionsSharedMemory")
    );
    if (!ActionsSharedMemoryHandle)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create actions shared memory."));
    }

    StatesSharedMemoryHandle = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        StatesMAXSize,
        TEXT("StatesSharedMemory")
    );
    if (!StatesSharedMemoryHandle)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create states shared memory."));
    }

    UpdateSharedMemoryHandle = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        UpdateMAXSize,
        TEXT("UpdateSharedMemory")
    );
    if (!UpdateSharedMemoryHandle)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create update shared memory."));
    }

    // 5) Create mutexes
    ActionsMutexHandle = CreateMutex(NULL, false, TEXT("ActionsDataMutex"));
    StatesMutexHandle = CreateMutex(NULL, false, TEXT("StatesDataMutex"));
    UpdateMutexHandle = CreateMutex(NULL, false, TEXT("UpdateDataMutex"));

    if (!ActionsMutexHandle || !StatesMutexHandle || !UpdateMutexHandle)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create one or more data mutexes."));
    }

    // 6) Create event objects
    ActionReadyEventHandle = CreateEvent(NULL, false, false, TEXT("ActionReadyEvent"));
    ActionReceivedEventHandle = CreateEvent(NULL, false, false, TEXT("ActionReceivedEvent"));
    UpdateReadyEventHandle = CreateEvent(NULL, false, false, TEXT("UpdateReadyEvent"));
    UpdateReceivedEventHandle = CreateEvent(NULL, false, false, TEXT("UpdateReceivedEvent"));

    // 7) Map the memory
    MappedActionsSharedData = (float*)MapViewOfFile(ActionsSharedMemoryHandle, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (!MappedActionsSharedData)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to map actions shared memory."));
    }

    MappedStatesSharedData = (float*)MapViewOfFile(StatesSharedMemoryHandle, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (!MappedStatesSharedData)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to map states shared memory."));
    }

    MappedUpdateSharedData = (float*)MapViewOfFile(UpdateSharedMemoryHandle, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (!MappedUpdateSharedData)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to map update shared memory."));
    }

    UE_LOG(LogTemp, Log, TEXT("USharedMemoryAgentCommunicator::Init complete."));
}

TArray<FAction> USharedMemoryAgentCommunicator::GetActions(
    TArray<FState> States,
    TArray<float> Dones,
    TArray<float> Truncs,
    int NumAgents
)
{
    TArray<FAction> OutActions;

    if (!MappedStatesSharedData)
    {
        UE_LOG(LogTemp, Error, TEXT("GetActions - MappedStatesSharedData is null."));
        return OutActions;
    }
    if (!MappedActionsSharedData)
    {
        UE_LOG(LogTemp, Error, TEXT("GetActions - MappedActionsSharedData is null."));
        return OutActions;
    }

    // Acquire states mutex
    if (WaitForSingleObject(StatesMutexHandle, INFINITE) != WAIT_OBJECT_0)
    {
        UE_LOG(LogTemp, Error, TEXT("GetActions - Failed to lock States mutex."));
        return OutActions;
    }

    // Write 6 info floats
    float fBufferSize = (float)BufferSize;
    float fBatchSize = (float)BatchSize;
    float fNumEnv = (float)NumEnvironments;
    bool  bIsMA = IsMultiAgent();
    float fMultiAgent = bIsMA ? (float)NumAgents : -1.f;

    int32 CurrEnvStateSize = ComputeSingleEnvStateSize(NumAgents);
    int32 CurrEnvActionSize = ComputeSingleEnvActionSize(NumAgents);

    float* p = MappedStatesSharedData;
    p[0] = fBufferSize;
    p[1] = fBatchSize;
    p[2] = fNumEnv;
    p[3] = fMultiAgent;  // if multi-agent => #agents, else -1
    p[4] = (float)CurrEnvStateSize;
    p[5] = (float)CurrEnvActionSize;

    p += 6;

    // Write states
    for (const FState& st : States)
    {
        FMemory::Memcpy(p, st.Values.GetData(), st.Values.Num() * sizeof(float));
        p += st.Values.Num();
    }

    // Write Dones
    FMemory::Memcpy(p, Dones.GetData(), Dones.Num() * sizeof(float));
    p += Dones.Num();

    // Write Truncs
    FMemory::Memcpy(p, Truncs.GetData(), Truncs.Num() * sizeof(float));
    p += Truncs.Num();

    // Release
    ReleaseMutex(StatesMutexHandle);
    SetEvent(ActionReadyEventHandle);

    // Wait for python side
    WaitForSingleObject(ActionReceivedEventHandle, INFINITE);

    // Acquire actions
    if (WaitForSingleObject(ActionsMutexHandle, INFINITE) != WAIT_OBJECT_0)
    {
        UE_LOG(LogTemp, Error, TEXT("GetActions - Failed to lock Actions mutex."));
        return OutActions;
    }

    // Read actions
    float* ActionPtr = MappedActionsSharedData;
    OutActions.Reserve(States.Num());
    for (int32 i = 0; i < States.Num(); i++)
    {
        FAction OneAction;
        OneAction.Values.SetNum(CurrEnvActionSize);

        FMemory::Memcpy(OneAction.Values.GetData(), ActionPtr, CurrEnvActionSize * sizeof(float));
        ActionPtr += CurrEnvActionSize;

        OutActions.Add(OneAction);
    }

    ReleaseMutex(ActionsMutexHandle);

    return OutActions;
}

void USharedMemoryAgentCommunicator::Update(const TArray<FExperienceBatch>& experiences, int NumAgents)
{
    if (WaitForSingleObject(UpdateMutexHandle, INFINITE) != WAIT_OBJECT_0)
    {
        UE_LOG(LogTemp, Error, TEXT("Update - Failed to lock the update mutex."));
        return;
    }

    if (!MappedUpdateSharedData)
    {
        UE_LOG(LogTemp, Error, TEXT("Update - MappedUpdateSharedData is null."));
        ReleaseMutex(UpdateMutexHandle);
        return;
    }

    float* ptr = MappedUpdateSharedData;
    int index = 0;

    // Write transitions => [ state + nextState + action + reward + trunc + done ]
    for (const FExperienceBatch& batch : experiences)
    {
        for (const FExperience& xp : batch.Experiences)
        {
            // 1) State
            FMemory::Memcpy(&ptr[index], xp.State.Values.GetData(), xp.State.Values.Num() * sizeof(float));
            index += xp.State.Values.Num();

            // 2) NextState
            FMemory::Memcpy(&ptr[index], xp.NextState.Values.GetData(), xp.NextState.Values.Num() * sizeof(float));
            index += xp.NextState.Values.Num();

            // 3) Action
            FMemory::Memcpy(&ptr[index], xp.Action.Values.GetData(), xp.Action.Values.Num() * sizeof(float));
            index += xp.Action.Values.Num();

            // 4) reward, trunc, done
            ptr[index++] = xp.Reward;
            ptr[index++] = xp.Trunc ? 1.f : 0.f;
            ptr[index++] = xp.Done ? 1.f : 0.f;
        }
    }

    ReleaseMutex(UpdateMutexHandle);

    // Signal python
    SetEvent(UpdateReadyEventHandle);

    // Wait for python
    WaitForSingleObject(UpdateReceivedEventHandle, INFINITE);
}

void USharedMemoryAgentCommunicator::WriteTransitionsToFile(
    const TArray<FExperienceBatch>& experiences,
    const FString& FilePath
)
{
    // We'll create a CSV file with lines like:
    // state[0], state[1], ..., nextState[0], ..., action[0], ..., reward, trunc, done
    // for each transition.

    FString Output;

    for (const FExperienceBatch& batch : experiences)
    {
        for (const FExperience& xp : batch.Experiences)
        {
            // state
            for (float val : xp.State.Values)
            {
                Output.Append(FString::SanitizeFloat(val));
                Output.AppendChar(',');
            }

            // nextState
            for (float val : xp.NextState.Values)
            {
                Output.Append(FString::SanitizeFloat(val));
                Output.AppendChar(',');
            }

            // action
            for (float aVal : xp.Action.Values)
            {
                Output.Append(FString::SanitizeFloat(aVal));
                Output.AppendChar(',');
            }

            // reward
            Output.Append(FString::SanitizeFloat(xp.Reward));
            Output.AppendChar(',');

            // trunc
            Output.Append(FString::SanitizeFloat(xp.Trunc ? 1.f : 0.f));
            Output.AppendChar(',');

            // done
            Output.Append(FString::SanitizeFloat(xp.Done ? 1.f : 0.f));
            Output.AppendChar('\n');
        }
    }

    // Save to file
    if (!FFileHelper::SaveStringToFile(Output, *FilePath))
    {
        UE_LOG(LogTemp, Warning, TEXT("WriteTransitionsToFile - Could not save to file: %s"), *FilePath);
    }
    else
    {
        UE_LOG(LogTemp, Log, TEXT("WriteTransitionsToFile - Wrote transitions to file: %s"), *FilePath);
    }
}

// -----------------------------------
// Private Helpers
// -----------------------------------

bool USharedMemoryAgentCommunicator::IsMultiAgent() const
{
    return (LocalEnvConfig && LocalEnvConfig->HasPath(TEXT("environment/shape/state/agent")));
}

bool USharedMemoryAgentCommunicator::HasCentralState() const
{
    return (LocalEnvConfig && LocalEnvConfig->HasPath(TEXT("environment/shape/state/central")));
}

int32 USharedMemoryAgentCommunicator::ComputeSingleEnvStateSize(int32 NumAgents) const
{
    int32 Size = 0;
    if (HasCentralState())
    {
        // environment/shape/state/central/obs_size
        Size += LocalEnvConfig->Get(TEXT("environment/shape/state/central/obs_size"))->AsInt();
    }
    if (IsMultiAgent())
    {
        int32 AgentObs = LocalEnvConfig->Get(TEXT("environment/shape/state/agent/obs_size"))->AsInt();
        Size += AgentObs * NumAgents;
    }
    return Size;
}

int32 USharedMemoryAgentCommunicator::ComputeSingleEnvActionSize(int32 NumAgents) const
{
    bool bMA = IsMultiAgent();
    FString ActionPath = bMA ? TEXT("environment/shape/action/agent")
        : TEXT("environment/shape/action/central");

    int32 DiscreteCount = 0;
    if (LocalEnvConfig->HasPath(ActionPath + TEXT("/discrete")))
    {
        UEnvironmentConfig* DiscreteNode = LocalEnvConfig->Get(ActionPath + TEXT("/discrete"));
        TArray<UEnvironmentConfig*> DiscreteArray = DiscreteNode->AsArrayOfConfigs();
        DiscreteCount = DiscreteArray.Num();
    }

    int32 ContinuousCount = 0;
    if (LocalEnvConfig->HasPath(ActionPath + TEXT("/continuous")))
    {
        UEnvironmentConfig* ContinuousNode = LocalEnvConfig->Get(ActionPath + TEXT("/continuous"));
        TArray<UEnvironmentConfig*> ContinuousArray = ContinuousNode->AsArrayOfConfigs();
        ContinuousCount = ContinuousArray.Num();
    }

    int32 SingleAgentActionSize = DiscreteCount + ContinuousCount;
    return bMA ? (SingleAgentActionSize * NumAgents) : SingleAgentActionSize;
}
