#include "SharedMemoryAgentCommunicator.h"
#include "Misc/FileHelper.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "EnvironmentConfig.h"

// Constructor
USharedMemoryAgentCommunicator::USharedMemoryAgentCommunicator()
    : ActionsSharedMemoryHandle(nullptr)
    , UpdateSharedMemoryHandle(nullptr)
    , ActionsMutexHandle(nullptr)
    , UpdateMutexHandle(nullptr)
    , ActionReadyEventHandle(nullptr)
    , ActionReceivedEventHandle(nullptr)
    , UpdateReadyEventHandle(nullptr)
    , UpdateReceivedEventHandle(nullptr)
    , MappedActionsSharedData(nullptr)
    , MappedStatesSharedData(nullptr)
    , MappedUpdateSharedData(nullptr)
    , ConfigSharedMemoryHandle(nullptr)
    , MappedConfigSharedData(nullptr)
    , ConfigMutexHandle(nullptr)
    , ConfigReadyEventHandle(nullptr)
{
}

// Destructor
USharedMemoryAgentCommunicator::~USharedMemoryAgentCommunicator()
{
    // Unmap memory views
    if (MappedActionsSharedData)   UnmapViewOfFile(MappedActionsSharedData);
    if (MappedStatesSharedData)    UnmapViewOfFile(MappedStatesSharedData);
    if (MappedUpdateSharedData)    UnmapViewOfFile(MappedUpdateSharedData);
    if (MappedConfigSharedData)    UnmapViewOfFile(MappedConfigSharedData);

    // Close handles
    if (ActionsSharedMemoryHandle) CloseHandle(ActionsSharedMemoryHandle);
    if (StatesSharedMemoryHandle)  CloseHandle(StatesSharedMemoryHandle);
    if (UpdateSharedMemoryHandle)  CloseHandle(UpdateSharedMemoryHandle);
    if (ConfigSharedMemoryHandle)  CloseHandle(ConfigSharedMemoryHandle);

    if (ActionsMutexHandle)        CloseHandle(ActionsMutexHandle);
    if (StatesMutexHandle)         CloseHandle(StatesMutexHandle);
    if (UpdateMutexHandle)         CloseHandle(UpdateMutexHandle);
    if (ConfigMutexHandle)         CloseHandle(ConfigMutexHandle);

    if (ActionReadyEventHandle)    CloseHandle(ActionReadyEventHandle);
    if (ActionReceivedEventHandle) CloseHandle(ActionReceivedEventHandle);
    if (UpdateReadyEventHandle)    CloseHandle(UpdateReadyEventHandle);
    if (UpdateReceivedEventHandle) CloseHandle(UpdateReceivedEventHandle);
    if (ConfigReadyEventHandle)    CloseHandle(ConfigReadyEventHandle);
}

// ------------------------------------------------------------
// Init
// ------------------------------------------------------------
void USharedMemoryAgentCommunicator::Init(UEnvironmentConfig* EnvConfig)
{
    if (!EnvConfig)
    {
        UE_LOG(LogTemp, Error, TEXT("USharedMemoryAgentCommunicator::Init - EnvConfig is null."));
        return;
    }

    // 1) Generate the JSON config we share with Python
    ConfigJSON = WriteConfigInfoToJson(EnvConfig);

    // 2) Read required data from EnvConfig to shape memory sizes
    // e.g.:
    int32 NumEnvironments = 1;
    if (EnvConfig->HasPath(TEXT("train/num_environments")))
    {
        NumEnvironments = EnvConfig->Get(TEXT("train/num_environments"))->AsInt();
    }

    int32 BatchSize = 256;
    if (EnvConfig->HasPath(TEXT("train/batch_size")))
    {
        BatchSize = EnvConfig->Get(TEXT("train/batch_size"))->AsInt();
    }

    // If multi-agent, we might do other logic:
    bool bIsMultiAgent = EnvConfig->HasPath(TEXT("environment/shape/agent"));

    int32 MaxAgents = 1;
    if (bIsMultiAgent)
    {
        MaxAgents = EnvConfig->Get(TEXT("environment/shape/agent/max"))->AsInt();
    }

    // Action and observation sizes
    int32 SingleAgentActionCount = ComputeSingleAgentNumActions(EnvConfig);
    int32 SingleAgentObsSize = ComputeSingleAgentObsSize(EnvConfig);

    // Then shape: total actions = MaxAgents * singleAgentActionCount (if multi-agent)
    int32 NumActions = bIsMultiAgent ? MaxAgents * SingleAgentActionCount
        : SingleAgentActionCount;

    // Similarly for states:
    int32 StateSize = bIsMultiAgent ? (MaxAgents * SingleAgentObsSize)
        : SingleAgentObsSize;

    // 3) Convert config size to bytes
    int32 ConfigSizeBytes = ConfigJSON.Num() * sizeof(char);

    // We'll define how large we want shared memory for actions/states/updates
    // For example:
    //   - We'll keep space for up to 'NumEnvironments' states, each of size 'StateSize'
    //   - If we want to add some overhead for done/trunc, etc., do so.

    // Let’s define some approximate sizes:
    int32 InfoSizeBytes = 6 * sizeof(float); // for extra metadata
    int32 TerminalsSizeBytes = NumEnvironments * 2 * sizeof(float); // Dones & Truncs
    int32 ActionSizeBytes = NumEnvironments * NumActions * sizeof(float);
    int32 StatesSizeBytes = (NumEnvironments * StateSize * sizeof(float))
        + InfoSizeBytes + TerminalsSizeBytes;

    // For the update buffer (trajectories):
    // Each transition might contain: State + NextState + Action + reward/done/trunc
    // And we might hold up to BatchSize transitions per environment, etc.
    // This is just an example formula:
    int32 UpdateSizeBytes = NumEnvironments * BatchSize *
        ((StateSize * 2) + NumActions + 3) * sizeof(float);

    // 4) Create the memory + events
    CreateSharedMemoryAndEvents(ActionSizeBytes, StatesSizeBytes, UpdateSizeBytes, ConfigSizeBytes);

    // 5) Write config data into shared memory
    WriteConfigToSharedMemory();
}

// ------------------------------------------------------------
// CreateSharedMemoryAndEvents
// ------------------------------------------------------------
void USharedMemoryAgentCommunicator::CreateSharedMemoryAndEvents(
    int32 ActionSizeBytes,
    int32 StatesSizeBytes,
    int32 UpdateSizeBytes,
    int32 ConfigSizeBytes
)
{
    // 1) Create memory mappings

    // Actions
    ActionsSharedMemoryHandle = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        ActionSizeBytes,
        TEXT("ActionsSharedMemory")
    );
    if (!ActionsSharedMemoryHandle)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create actions shared memory."));
    }

    // States
    StatesSharedMemoryHandle = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        StatesSizeBytes,
        TEXT("StatesSharedMemory")
    );
    if (!StatesSharedMemoryHandle)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create states shared memory."));
    }

    // Update
    UpdateSharedMemoryHandle = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        UpdateSizeBytes,
        TEXT("UpdateSharedMemory")
    );
    if (!UpdateSharedMemoryHandle)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create update shared memory."));
    }

    // Config
    ConfigSharedMemoryHandle = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        ConfigSizeBytes,
        TEXT("ConfigSharedMemory")
    );
    if (!ConfigSharedMemoryHandle)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create configuration shared memory."));
    }

    // 2) Create mutexes
    ActionsMutexHandle = CreateMutex(NULL, false, TEXT("ActionsDataMutex"));
    StatesMutexHandle = CreateMutex(NULL, false, TEXT("StatesDataMutex"));
    UpdateMutexHandle = CreateMutex(NULL, false, TEXT("UpdateDataMutex"));
    ConfigMutexHandle = CreateMutex(NULL, false, TEXT("ConfigDataMutex"));

    if (!ActionsMutexHandle)
        UE_LOG(LogTemp, Error, TEXT("Failed to create actions mutex."));
    if (!StatesMutexHandle)
        UE_LOG(LogTemp, Error, TEXT("Failed to create states mutex."));
    if (!UpdateMutexHandle)
        UE_LOG(LogTemp, Error, TEXT("Failed to create update mutex."));
    if (!ConfigMutexHandle)
        UE_LOG(LogTemp, Error, TEXT("Failed to create config mutex."));

    // 3) Create event objects
    ActionReadyEventHandle = CreateEvent(NULL, false, false, TEXT("ActionReadyEvent"));
    ActionReceivedEventHandle = CreateEvent(NULL, false, false, TEXT("ActionReceivedEvent"));
    UpdateReadyEventHandle = CreateEvent(NULL, false, false, TEXT("UpdateReadyEvent"));
    UpdateReceivedEventHandle = CreateEvent(NULL, false, false, TEXT("UpdateReceivedEvent"));
    ConfigReadyEventHandle = CreateEvent(NULL, false, false, TEXT("ConfigReadyEvent"));

    // 4) Map views of memory
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

    MappedConfigSharedData = (int32*)MapViewOfFile(ConfigSharedMemoryHandle, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (!MappedConfigSharedData)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to map view of shared memory for config."));
    }
}

// ------------------------------------------------------------
// WriteConfigToSharedMemory
// ------------------------------------------------------------
void USharedMemoryAgentCommunicator::WriteConfigToSharedMemory()
{
    if (WaitForSingleObject(ConfigMutexHandle, INFINITE) != WAIT_OBJECT_0)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to lock the configuration mutex."));
        return;
    }

    if (MappedConfigSharedData)
    {
        // Copy the config JSON (as chars) into the shared memory
        FMemory::Memcpy(MappedConfigSharedData, ConfigJSON.GetData(), ConfigJSON.Num() * sizeof(char));

        // Signal that the configuration is ready
        if (!SetEvent(ConfigReadyEventHandle))
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to set the configuration ready event."));
        }
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Mapped memory for configuration is not available."));
    }

    ReleaseMutex(ConfigMutexHandle);
}

// ------------------------------------------------------------
// GetActions
// ------------------------------------------------------------
TArray<FAction> USharedMemoryAgentCommunicator::GetActions(
    TArray<FState> States,
    TArray<float> Dones,
    TArray<float> Truncs,
    int NumAgents
)
{
    TArray<FAction> Actions;

    // We need to write states/dones/truncs into shared memory
    if (!MappedStatesSharedData)
    {
        UE_LOG(LogTemp, Error, TEXT("Mapped memory for states is not available."));
        return Actions;
    }

    // 1) Possibly write some metadata about how many envs, how many agents, etc.
    // If needed, you'd lock the StatesMutex here.
    // This example omits the explicit WaitForSingleObject(StatesMutexHandle, ...) for brevity.

    // e.g. let's say the first few floats store [NumAgents, StatesCount, ...]. 
    // We'll do something simple as an example:
    float* WritePtr = MappedStatesSharedData;
    WritePtr[0] = static_cast<float>(NumAgents);
    WritePtr[1] = static_cast<float>(States.Num()); // # envs
    // etc. Then move forward
    WritePtr += 2;

    // 2) Copy all state vectors
    for (const FState& S : States)
    {
        FMemory::Memcpy(WritePtr, S.Values.GetData(), S.Values.Num() * sizeof(float));
        WritePtr += S.Values.Num();
    }

    // 3) Copy done + trunc arrays
    FMemory::Memcpy(WritePtr, Dones.GetData(), Dones.Num() * sizeof(float));
    WritePtr += Dones.Num();
    FMemory::Memcpy(WritePtr, Truncs.GetData(), Truncs.Num() * sizeof(float));
    WritePtr += Truncs.Num();

    // Signal python that states are ready
    if (!SetEvent(ActionReadyEventHandle))
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to set the ActionReadyEvent."));
    }

    // Wait for python to produce actions
    WaitForSingleObject(ActionReceivedEventHandle, INFINITE);

    // Now read back the actions from MappedActionsSharedData
    if (!MappedActionsSharedData)
    {
        UE_LOG(LogTemp, Error, TEXT("Mapped memory for actions is not available."));
        return Actions;
    }

    float* ReadPtr = MappedActionsSharedData;

    // For each environment, read the entire action array
    for (int32 envIdx = 0; envIdx < States.Num(); envIdx++)
    {
        // e.g. if each agent has X actions, total is (NumAgents * X)
        // But you might have a known value or pass it from python
        // We'll guess each environment has "ActionsPerEnv" from the python side
        // For demonstration, let's say python wrote "NumAgents * 2" actions
        // We'll do something naive:
        int32 ActionsPerEnv = NumAgents * 2;

        FAction action;
        action.Values.SetNum(ActionsPerEnv);

        // Copy the floats
        FMemory::Memcpy(action.Values.GetData(), ReadPtr, ActionsPerEnv * sizeof(float));
        ReadPtr += ActionsPerEnv;

        Actions.Add(action);
    }

    // Release the actions mutex if you locked it (omitted here).
    return Actions;
}

// ------------------------------------------------------------
// Update
// ------------------------------------------------------------
void USharedMemoryAgentCommunicator::Update(const TArray<FExperienceBatch>& experiences, int NumAgents)
{
    // We'll lock the update region
    WaitForSingleObject(UpdateMutexHandle, INFINITE);

    if (!MappedUpdateSharedData)
    {
        UE_LOG(LogTemp, Error, TEXT("Mapped memory for update is not available."));
        ReleaseMutex(UpdateMutexHandle);
        return;
    }

    float* WritePtr = MappedUpdateSharedData;
    int32 index = 0;

    // For each environment's trajectory
    for (const FExperienceBatch& trajectory : experiences)
    {
        // For each transition
        for (const FExperience& T : trajectory.Experiences)
        {
            // Write state
            FMemory::Memcpy(&WritePtr[index], T.State.Values.GetData(), T.State.Values.Num() * sizeof(float));
            index += T.State.Values.Num();

            // Write next state
            FMemory::Memcpy(&WritePtr[index], T.NextState.Values.GetData(), T.NextState.Values.Num() * sizeof(float));
            index += T.NextState.Values.Num();

            // Write action
            FMemory::Memcpy(&WritePtr[index], T.Action.Values.GetData(), T.Action.Values.Num() * sizeof(float));
            index += T.Action.Values.Num();

            // Write reward, trunc, done
            WritePtr[index++] = T.Reward;
            WritePtr[index++] = (float)T.Trunc;
            WritePtr[index++] = (float)T.Done;
        }
    }

    // Done writing
    ReleaseMutex(UpdateMutexHandle);
    // Notify python
    if (!SetEvent(UpdateReadyEventHandle))
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to set UpdateReadyEvent."));
    }
    // Wait for python to confirm it read
    WaitForSingleObject(UpdateReceivedEventHandle, INFINITE);
}

// ------------------------------------------------------------
// PrintActionsAsMatrix
// ------------------------------------------------------------
void USharedMemoryAgentCommunicator::PrintActionsAsMatrix(const TArray<FAction>& Actions)
{
    if (Actions.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("PrintActionsAsMatrix - No actions to print."));
        return;
    }

    FString MatrixStr = TEXT("[\n");
    for (int32 envIdx = 0; envIdx < Actions.Num(); envIdx++)
    {
        MatrixStr += TEXT("   [");
        const FAction& act = Actions[envIdx];
        for (int32 a = 0; a < act.Values.Num(); a++)
        {
            MatrixStr += FString::Printf(TEXT("%.2f"), act.Values[a]);
            if (a < act.Values.Num() - 1)
            {
                MatrixStr += TEXT(", ");
            }
        }
        MatrixStr += TEXT("]");
        if (envIdx < Actions.Num() - 1)
        {
            MatrixStr += TEXT(",\n");
        }
        else
        {
            MatrixStr += TEXT("\n");
        }
    }
    MatrixStr += TEXT("]");

    UE_LOG(LogTemp, Warning, TEXT("Actions:\n%s"), *MatrixStr);
}

// ------------------------------------------------------------
// WriteConfigInfoToJson
// ------------------------------------------------------------
TArray<char> USharedMemoryAgentCommunicator::WriteConfigInfoToJson(UEnvironmentConfig* EnvConfig)
{
    /*
        We create a JSON that Python reads to shape the environment:
        {
          "EnvId": 3,
          "EnvironmentInfo": {
              "MultiAgent": {
                  "IsMultiAgent": true,
                  "NumAgents": 5,
                  "SingleAgentObsSize": ...
              },
              "StateSize": ...
              "ActionSpace": {
                  "DiscreteActions": [...],
                  "ContinuousActions": [...]
              }
          },
          "TrainInfo": {
              "NumEnvironments": ...
              "BufferSize": ...
              "BatchSize": ...
              ...
          }
        }
    */

    TSharedPtr<FJsonObject> RootObject = MakeShareable(new FJsonObject);

    // 1) EnvID: example from environment/id
    int32 EnvID = 0;
    if (EnvConfig->HasPath(TEXT("environment/id")))
    {
        FString EnvIDStr = EnvConfig->Get(TEXT("environment/id"))->AsString();
        EnvID = FCString::Atoi(*EnvIDStr);
    }
    RootObject->SetNumberField(TEXT("EnvId"), EnvID);

    // 2) Build "EnvironmentInfo" object
    TSharedPtr<FJsonObject> EnvInfoObj = MakeShareable(new FJsonObject);

    bool bIsMultiAgent = EnvConfig->HasPath(TEXT("environment/shape/agent"));
    EnvInfoObj->SetBoolField(TEXT("IsMultiAgent"), bIsMultiAgent);

    int32 MaxAgents = 1;
    if (bIsMultiAgent)
    {
        MaxAgents = EnvConfig->Get(TEXT("environment/shape/agent/max"))->AsInt();
    }
    EnvInfoObj->SetNumberField(TEXT("NumAgents"), MaxAgents);

    // single-agent observation size
    int32 SingleAgentObsSize = ComputeSingleAgentObsSize(EnvConfig);
    EnvInfoObj->SetNumberField(TEXT("SingleAgentObsSize"), SingleAgentObsSize);

    // total state size
    // if multi-agent, might be SingleAgentObsSize * MaxAgents, else just SingleAgentObsSize
    int32 StateSize = bIsMultiAgent ? (SingleAgentObsSize * MaxAgents) : SingleAgentObsSize;
    EnvInfoObj->SetNumberField(TEXT("StateSize"), StateSize);

    // 3) ActionSpace
    TSharedPtr<FJsonObject> ActionSpaceObj = MakeShareable(new FJsonObject);

    // Let's parse discrete array from "environment/shape/action/agent/discrete"
    if (EnvConfig->HasPath(TEXT("environment/shape/action/agent/discrete")))
    {
        UEnvironmentConfig* DiscreteNode = EnvConfig->Get(TEXT("environment/shape/action/agent/discrete"));
        TArray<UEnvironmentConfig*> DiscreteArr = DiscreteNode->AsArrayOfConfigs();

        // We'll store them as an array of integers
        TArray<TSharedPtr<FJsonValue>> DiscreteActionsArray;
        for (UEnvironmentConfig* Item : DiscreteArr)
        {
            int32 NumChoices = Item->Get(TEXT("num_choices"))->AsInt();
            DiscreteActionsArray.Add(MakeShareable(new FJsonValueNumber(NumChoices)));
        }
        ActionSpaceObj->SetArrayField(TEXT("DiscreteActions"), DiscreteActionsArray);
    }
    else
    {
        // no discrete array => set empty or do nothing
        ActionSpaceObj->SetArrayField(TEXT("DiscreteActions"), TArray<TSharedPtr<FJsonValue>>());
    }

    // parse continuous from "environment/shape/action/agent/continuous"
    if (EnvConfig->HasPath(TEXT("environment/shape/action/agent/continuous")))
    {
        UEnvironmentConfig* ContinuousNode = EnvConfig->Get(TEXT("environment/shape/action/agent/continuous"));
        TArray<UEnvironmentConfig*> ContArr = ContinuousNode->AsArrayOfConfigs();

        TArray<TSharedPtr<FJsonValue>> ContinuousActionsArray;
        for (UEnvironmentConfig* RangeItem : ContArr)
        {
            TArray<float> RangeValues = RangeItem->AsArrayOfNumbers();
            // e.g. 2 floats: [low, high]
            if (RangeValues.Num() == 2)
            {
                TSharedPtr<FJsonObject> RangeObj = MakeShareable(new FJsonObject);
                RangeObj->SetNumberField(TEXT("Low"), RangeValues[0]);
                RangeObj->SetNumberField(TEXT("High"), RangeValues[1]);
                ContinuousActionsArray.Add(MakeShareable(new FJsonValueObject(RangeObj)));
            }
        }
        ActionSpaceObj->SetArrayField(TEXT("ContinuousActions"), ContinuousActionsArray);
    }
    else
    {
        ActionSpaceObj->SetArrayField(TEXT("ContinuousActions"), TArray<TSharedPtr<FJsonValue>>());
    }

    // Attach ActionSpace to EnvInfo
    EnvInfoObj->SetObjectField(TEXT("ActionSpace"), ActionSpaceObj);

    RootObject->SetObjectField(TEXT("EnvironmentInfo"), EnvInfoObj);

    // 4) TrainInfo object
    TSharedPtr<FJsonObject> TrainInfoObj = MakeShareable(new FJsonObject);

    int32 NumEnvironments = EnvConfig->HasPath(TEXT("train/num_environments"))
        ? EnvConfig->Get(TEXT("train/num_environments"))->AsInt()
        : 1;
    int32 BufferSize = EnvConfig->HasPath(TEXT("train/buffer_size"))
        ? EnvConfig->Get(TEXT("train/buffer_size"))->AsInt()
        : 256;
    int32 BatchSize = EnvConfig->HasPath(TEXT("train/batch_size"))
        ? EnvConfig->Get(TEXT("train/batch_size"))->AsInt()
        : 256;

    TrainInfoObj->SetNumberField(TEXT("NumEnvironments"), NumEnvironments);
    TrainInfoObj->SetNumberField(TEXT("BufferSize"), BufferSize);
    TrainInfoObj->SetNumberField(TEXT("BatchSize"), BatchSize);
    TrainInfoObj->SetNumberField(TEXT("MaxAgents"), MaxAgents);

    RootObject->SetObjectField(TEXT("TrainInfo"), TrainInfoObj);

    // 5) Serialize to string
    FString OutputString;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&OutputString);
    if (!FJsonSerializer::Serialize(RootObject.ToSharedRef(), Writer))
    {
        // If serialization fails, return empty
        return {};
    }

    // convert FString -> TArray<char>
    TArray<char> CharArray;
    CharArray.Append(reinterpret_cast<const char*>(TCHAR_TO_UTF8(*OutputString)), OutputString.Len());

    return CharArray;
}

// ------------------------------------------------------------
// Helper: compute single-agent # actions
// ------------------------------------------------------------
int32 USharedMemoryAgentCommunicator::ComputeSingleAgentNumActions(UEnvironmentConfig* EnvConfig) const
{
    int32 DiscreteCount = 0;
    int32 ContinuousCount = 0;

    if (EnvConfig->HasPath(TEXT("environment/shape/action/agent/discrete")))
    {
        TArray<UEnvironmentConfig*> DiscreteItems = EnvConfig
            ->Get(TEXT("environment/shape/action/agent/discrete"))
            ->AsArrayOfConfigs();
        for (UEnvironmentConfig* Item : DiscreteItems)
        {
            int32 Choices = Item->Get(TEXT("num_choices"))->AsInt();
            // Typically, 1 discrete dimension => 1 action float, but you might store the index
            // We'll just assume each is 1 "slot" in the final action array
            DiscreteCount += 1;
        }
    }

    if (EnvConfig->HasPath(TEXT("environment/shape/action/agent/continuous")))
    {
        TArray<UEnvironmentConfig*> ContItems = EnvConfig
            ->Get(TEXT("environment/shape/action/agent/continuous"))
            ->AsArrayOfConfigs();
        // Each sub-range is 1 float dimension
        ContinuousCount = ContItems.Num();
    }

    return DiscreteCount + ContinuousCount;
}

// ------------------------------------------------------------
// Helper: compute single-agent obs size
// ------------------------------------------------------------
int32 USharedMemoryAgentCommunicator::ComputeSingleAgentObsSize(UEnvironmentConfig* EnvConfig) const
{
    // Example: environment/shape/state/central/obs_size => e.g. 2500
    // plus environment/shape/state/agent/obs_size => e.g. 26
    // Or you might define a single total, etc. 
    // We'll do a naive approach: sum "central.obs_size" + "agent.obs_size" 
    // if both exist.

    int32 CentralSize = 0;
    int32 AgentSize = 0;

    // Central
    if (EnvConfig->HasPath(TEXT("environment/shape/state/central/obs_size")))
    {
        CentralSize = EnvConfig->Get(TEXT("environment/shape/state/central/obs_size"))->AsInt();
    }

    // Agent
    if (EnvConfig->HasPath(TEXT("environment/shape/state/agent/obs_size")))
    {
        AgentSize = EnvConfig->Get(TEXT("environment/shape/state/agent/obs_size"))->AsInt();
    }

    // Return sum
    return CentralSize + AgentSize;
}
