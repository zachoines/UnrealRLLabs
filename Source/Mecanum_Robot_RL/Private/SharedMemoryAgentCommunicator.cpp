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
    int32 InfoSize = 6 * sizeof(float);
    int32 TerminalsSize = NumEnvironments * 2 * sizeof(float); // done + trunc per env
    StatesMAXSize = NumEnvironments * (SingleEnvStateSize * sizeof(float))
        + InfoSize
        + TerminalsSize;

    // (c) Update => each transition: [state + nextState + action + reward + trunc + done]
    // Single transition = (SingleEnvStateSize*2) + TotalActionCount + (DoneSize + TruncSize + RewardSize)
    // For multi-agent done/trunc, we could do more logic, but here's a simpler approach with single floats each.
    int32 DoneSize = 1;
    int32 TruncSize = 1;
    int32 RewardSize = 1;
    int32 SingleTransitionFloatCount = (SingleEnvStateSize * 2)
        + TotalActionCount
        + (DoneSize + TruncSize + RewardSize);
    UpdateMAXSize = NumEnvironments * BatchSize * SingleTransitionFloatCount * sizeof(float);

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

    // Done (no extra logs of memory sizes needed)
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

    // Basic safety checks
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

    // 1) Acquire States mutex
    if (WaitForSingleObject(StatesMutexHandle, INFINITE) != WAIT_OBJECT_0)
    {
        UE_LOG(LogTemp, Error, TEXT("GetActions - Failed to lock States mutex."));
        return OutActions;
    }

    // 2) Write 6 info floats at start
    float fBufferSize = (float)BufferSize;
    float fBatchSize = (float)BatchSize;
    float fNumEnv = (float)NumEnvironments;
    bool  bIsMA = IsMultiAgent();
    float fMultiAgent = bIsMA ? (float)NumAgents : -1.f;

    // For the *current* agent count we are using, compute
    int32 CurrEnvStateSize = ComputeSingleEnvStateSize(NumAgents);
    int32 CurrEnvActionSize = ComputeSingleEnvActionSize(NumAgents);

    float* p = MappedStatesSharedData;
    p[0] = fBufferSize;
    p[1] = fBatchSize;
    p[2] = fNumEnv;
    p[3] = fMultiAgent;  // if multi-agent => NumAgents, else -1
    p[4] = (float)CurrEnvStateSize;
    p[5] = (float)CurrEnvActionSize;

    p += 6;

    // 3) Write states
    for (const FState& st : States)
    {
        FMemory::Memcpy(p, st.Values.GetData(), st.Values.Num() * sizeof(float));
        p += st.Values.Num();
    }

    // 4) Write dones
    FMemory::Memcpy(p, Dones.GetData(), Dones.Num() * sizeof(float));
    p += Dones.Num();

    // 5) Write truncs
    FMemory::Memcpy(p, Truncs.GetData(), Truncs.Num() * sizeof(float));
    p += Truncs.Num();

    // Release states mutex
    ReleaseMutex(StatesMutexHandle);

    // 6) Notify python that data is ready
    SetEvent(ActionReadyEventHandle);

    // 7) Wait for python to produce actions
    WaitForSingleObject(ActionReceivedEventHandle, INFINITE);

    // 8) Acquire Actions mutex
    if (WaitForSingleObject(ActionsMutexHandle, INFINITE) != WAIT_OBJECT_0)
    {
        UE_LOG(LogTemp, Error, TEXT("GetActions - Failed to lock Actions mutex."));
        return OutActions;
    }

    // 9) Read actions
    float* ActionPtr = MappedActionsSharedData;
    // We'll produce 1 action per environment => total = NumEnvironments
    // each action has CurrEnvActionSize floats
    OutActions.Reserve(States.Num());
    for (int32 e = 0; e < States.Num(); e++)
    {
        FAction OneAction;
        OneAction.Values.SetNum(CurrEnvActionSize);
        FMemory::Memcpy(OneAction.Values.GetData(), ActionPtr, CurrEnvActionSize * sizeof(float));
        ActionPtr += CurrEnvActionSize;
        OutActions.Add(OneAction);
    }

    // Release action mutex
    ReleaseMutex(ActionsMutexHandle);

    return OutActions;
}

void USharedMemoryAgentCommunicator::Update(const TArray<FExperienceBatch>& experiences, int NumAgents)
{
    // 1) Acquire Update mutex
    if (WaitForSingleObject(UpdateMutexHandle, INFINITE) != WAIT_OBJECT_0)
    {
        UE_LOG(LogTemp, Error, TEXT("Update - Failed to lock update mutex."));
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

    // Write transitions => [ state, nextState, action, reward, trunc, done ]
    for (const FExperienceBatch& batch : experiences)
    {
        for (const FExperience& xp : batch.Experiences)
        {
            // State
            for (float val : xp.State.Values)
            {
                ptr[index++] = val;
            }
            // NextState
            for (float val : xp.NextState.Values)
            {
                ptr[index++] = val;
            }
            // Action
            for (float aVal : xp.Action.Values)
            {
                ptr[index++] = aVal;
            }
            // Reward
            ptr[index++] = xp.Reward;
            // Trunc
            ptr[index++] = xp.Trunc ? 1.f : 0.f;
            // Done
            ptr[index++] = xp.Done ? 1.f : 0.f;
        }
    }

    ReleaseMutex(UpdateMutexHandle);

    // 2) Signal python side
    SetEvent(UpdateReadyEventHandle);

    // 3) Wait for python side to confirm receipt
    WaitForSingleObject(UpdateReceivedEventHandle, INFINITE);
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
    // Are we multi-agent or single-agent?
    bool bMA = IsMultiAgent();

    // If multi-agent => read from "environment/shape/action/agent" 
    // else => from "environment/shape/action/central"
    FString ActionPath = bMA ? TEXT("environment/shape/action/agent")
        : TEXT("environment/shape/action/central");

    // Count discrete
    int32 DiscreteCount = 0;
    if (LocalEnvConfig->HasPath(ActionPath + TEXT("/discrete")))
    {
        UEnvironmentConfig* DiscreteNode = LocalEnvConfig->Get(ActionPath + TEXT("/discrete"));
        TArray<UEnvironmentConfig*> DiscreteArray = DiscreteNode->AsArrayOfConfigs();
        DiscreteCount = DiscreteArray.Num();
    }

    // Count continuous
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