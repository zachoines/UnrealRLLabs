#include "SharedMemoryAgentCommunicator.h"

USharedMemoryAgentCommunicator::USharedMemoryAgentCommunicator()
    : ActionsSharedMemoryHandle(NULL), UpdateSharedMemoryHandle(NULL),
    ActionsMutexHandle(NULL), UpdateMutexHandle(NULL),
    ActionReadyEventHandle(NULL), ActionReceivedEventHandle(NULL),
    UpdateReadyEventHandle(NULL), UpdateReceivedEventHandle(NULL),
    MappedActionsSharedData(NULL), MappedStatesSharedData(NULL), MappedUpdateSharedData(NULL),
    ConfigSharedMemoryHandle(NULL), MappedConfigSharedData(NULL),
    ConfigMutexHandle(NULL), ConfigReadyEventHandle(NULL)
{
}

void USharedMemoryAgentCommunicator::Init(FEnvInfo EnvInfo, FTrainParams TrainParams)
{
    params = TrainParams;
    info = EnvInfo;

    ConfigJSON = WriteConfigInfoToJson(EnvInfo, TrainParams);

    int NumActions;
    int StateSize;
    int DoneSize;
    int TruncSize;
    int RewardSize;
    if (EnvInfo.IsMultiAgent) {
        NumActions = EnvInfo.MaxAgents * EnvInfo.ActionSpace->TotalActions();
        StateSize = EnvInfo.MaxAgents * EnvInfo.SingleAgentObsSize; 
        DoneSize = EnvInfo.MaxAgents;
        TruncSize = EnvInfo.MaxAgents;
        RewardSize = EnvInfo.MaxAgents;
    }
    else {
        NumActions = EnvInfo.ActionSpace->TotalActions();
        StateSize = EnvInfo.StateSize;
        DoneSize = 1;
        TruncSize = 1;
        RewardSize = 1;
    }

    int32 ConfigSize = ConfigJSON.Num() * sizeof(char);
    int32 InfoSize = 6 * sizeof(float);
    int32 ActionMAXSize = params.NumEnvironments * NumActions * sizeof(float);
    int32 StatesMAXSize = params.NumEnvironments * StateSize * sizeof(float);
    int32 UpdateMAXSize = params.NumEnvironments * params.BatchSize * ((StateSize * 2) + NumActions + (DoneSize + TruncSize + RewardSize)) * sizeof(float);

    // Creating shared memory for actions
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

    // Creating shared memory for states
    StatesSharedMemoryHandle = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        StatesMAXSize + InfoSize,
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
        UpdateMAXSize,
        TEXT("UpdateSharedMemory")
    );
    if (!UpdateSharedMemoryHandle)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create update shared memory."));
    }

    // Creating shared memory for configuration
    ConfigSharedMemoryHandle = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        ConfigSize,
        TEXT("ConfigSharedMemory")
    );
    if (!ConfigSharedMemoryHandle)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create configuration shared memory."));
    }

    // Initializing mutuex objects
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

    ConfigMutexHandle = CreateMutex(NULL, false, TEXT("ConfigDataMutex"));
    if (!ConfigMutexHandle)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create config mutex."));
    }

    // Creating event objects
    ActionReadyEventHandle = CreateEvent(NULL, false, false, TEXT("ActionReadyEvent"));
    ActionReceivedEventHandle = CreateEvent(NULL, false, false, TEXT("ActionReceivedEvent"));
    UpdateReadyEventHandle = CreateEvent(NULL, false, false, TEXT("UpdateReadyEvent"));
    UpdateReceivedEventHandle = CreateEvent(NULL, false, false, TEXT("UpdateReceivedEvent"));
    ConfigReadyEventHandle = CreateEvent(NULL, false, false, TEXT("ConfigReadyEvent"));

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

    MappedConfigSharedData = (int32*)MapViewOfFile(ConfigSharedMemoryHandle, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (!MappedConfigSharedData)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to map view of shared memory for config."));
    }

    WriteConfigToSharedMemory();
}

void USharedMemoryAgentCommunicator::WriteConfigToSharedMemory()
{
    if (WaitForSingleObject(ConfigMutexHandle, INFINITE) != WAIT_OBJECT_0)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to lock the configuration mutex."));
        return;
    }

    if (MappedConfigSharedData)
    {
        FMemory::Memcpy(MappedConfigSharedData, ConfigJSON.GetData(), ConfigJSON.Num() * sizeof(char));

        // Signal that the configuration is ready.
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

TArray<FAction> USharedMemoryAgentCommunicator::GetActions(TArray<FState> States, int NumAgents)
{
    TArray<FAction> Actions;

    float* StateSharedData = MappedStatesSharedData;
    if (!StateSharedData)
    {
        UE_LOG(LogTemp, Error, TEXT("Mapped memory for states is not available."));
        return Actions;
    }

    // Write Info
    StateSharedData[0] = params.BufferSize;
    StateSharedData[1] = params.BatchSize;
    StateSharedData[2] = params.NumEnvironments;
    StateSharedData[3] = info.IsMultiAgent ? NumAgents : -1;
    StateSharedData[4] = info.IsMultiAgent ? info.SingleAgentObsSize * NumAgents : info.StateSize;
    StateSharedData[5] = info.IsMultiAgent ? info.ActionSpace->TotalActions() * NumAgents : info.ActionSpace->TotalActions();
    StateSharedData += 6;

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
        action.Values.SetNumZeroed(NumAgents * info.ActionSpace->TotalActions());
        FMemory::Memcpy(action.Values.GetData(), ActionSharedData, action.Values.Num() * sizeof(float));
        Actions.Add(action);
        ActionSharedData += action.Values.Num();
    }

    ReleaseMutex(ActionsMutexHandle);

    return Actions;
}

void USharedMemoryAgentCommunicator::Update(const TArray<FExperienceBatch>& experiences, int NumAgents)
{
    WaitForSingleObject(UpdateMutexHandle, INFINITE);

    if (!MappedUpdateSharedData)
    {
        UE_LOG(LogTemp, Error, TEXT("Mapped memory for update is not available."));
        ReleaseMutex(UpdateMutexHandle);
        return;
    }

    float* SharedData = MappedUpdateSharedData;
    int index = 0;

    // For each enironment's trajectory
    for (const FExperienceBatch& trajectory : experiences)
    {
        // For each transition in that trajectory
        for (const FExperience& Transition : trajectory.Experiences)
        {
            // Write each state
            for (const float& elem : Transition.State.Values) 
            {
                SharedData[index] = elem;
                index++;
            }

            // Write each next state
            for (const float& elem : Transition.NextState.Values)
            {
                SharedData[index] = elem;
                index++;
            }

            // Write each Action
            for (const float& elem : Transition.Action.Values)
            {
                SharedData[index] = elem;
                index++;
            }

            SharedData[index] = (float)Transition.Reward;
            index++;
            SharedData[index] = (float)Transition.Trunc;
            index++;
            SharedData[index] = (float)Transition.Done;
            index++;
        }
    }
    
    ReleaseMutex(UpdateMutexHandle);
    SetEvent(UpdateReadyEventHandle);  
    WaitForSingleObject(UpdateReceivedEventHandle, INFINITE);
}

USharedMemoryAgentCommunicator::~USharedMemoryAgentCommunicator()
{
    if (MappedActionsSharedData)
        UnmapViewOfFile(MappedActionsSharedData);
    if (MappedStatesSharedData)
        UnmapViewOfFile(MappedStatesSharedData);
    if (MappedUpdateSharedData)
        UnmapViewOfFile(MappedUpdateSharedData);
    if (MappedConfigSharedData)
        UnmapViewOfFile(MappedConfigSharedData);

    if (ActionsSharedMemoryHandle)
        CloseHandle(ActionsSharedMemoryHandle);
    if (StatesSharedMemoryHandle)
        CloseHandle(StatesSharedMemoryHandle);
    if (UpdateSharedMemoryHandle)
        CloseHandle(UpdateSharedMemoryHandle);
    if (ConfigSharedMemoryHandle)
        CloseHandle(ConfigSharedMemoryHandle);

    if (ActionsMutexHandle)
        CloseHandle(ActionsMutexHandle);
    if (StatesMutexHandle)
        CloseHandle(StatesMutexHandle);
    if (UpdateMutexHandle)
        CloseHandle(UpdateMutexHandle);
    if (ConfigMutexHandle)
        CloseHandle(ConfigMutexHandle);

    if (ActionReadyEventHandle)
        CloseHandle(ActionReadyEventHandle);
    if (ActionReceivedEventHandle)
        CloseHandle(ActionReceivedEventHandle);
    if (UpdateReadyEventHandle)
        CloseHandle(UpdateReadyEventHandle);
    if (UpdateReceivedEventHandle)
        CloseHandle(UpdateReceivedEventHandle);
    if (ConfigReadyEventHandle)
        CloseHandle(ConfigReadyEventHandle);
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

TArray<char> USharedMemoryAgentCommunicator::WriteConfigInfoToJson(const FEnvInfo& EnvInfo, const FTrainParams& TrainParams)
{
    // Create the root JSON object
    TSharedPtr<FJsonObject> JsonObject = MakeShareable(new FJsonObject);

    // Set the EnvId
    JsonObject->SetNumberField(TEXT("EnvId"), EnvInfo.EnvID);

    // EnvironmentInfo object
    TSharedPtr<FJsonObject> EnvironmentInfoObject = MakeShareable(new FJsonObject);

    // MultiAgent object
    TSharedPtr<FJsonObject> MultiAgentObject = MakeShareable(new FJsonObject);
    MultiAgentObject->SetBoolField(TEXT("IsMultiAgent"), EnvInfo.IsMultiAgent);
    MultiAgentObject->SetNumberField(TEXT("SingleAgentObsSize"), EnvInfo.SingleAgentObsSize);
    MultiAgentObject->SetNumberField(TEXT("NumAgents"), EnvInfo.MaxAgents);
    EnvironmentInfoObject->SetObjectField(TEXT("MultiAgent"), MultiAgentObject);

    EnvironmentInfoObject->SetNumberField(TEXT("StateSize"), EnvInfo.StateSize);

    // ActionSpace object
    TSharedPtr<FJsonObject> ActionSpaceObject = MakeShareable(new FJsonObject);

    // ContinuousActions array
    TArray<TSharedPtr<FJsonValue>> ContinuousActionsArray;
    for (const FContinuousActionSpec& ContinuousAction : EnvInfo.ActionSpace->ContinuousActions)
    {
        TSharedPtr<FJsonObject> ActionObject = MakeShareable(new FJsonObject);
        ActionObject->SetNumberField(TEXT("Low"), ContinuousAction.Low);
        ActionObject->SetNumberField(TEXT("High"), ContinuousAction.High);
        ContinuousActionsArray.Add(MakeShareable(new FJsonValueObject(ActionObject)));
    }
    ActionSpaceObject->SetArrayField(TEXT("ContinuousActions"), ContinuousActionsArray);

    // DiscreteActions array
    TArray<TSharedPtr<FJsonValue>> DiscreteActionsArray;
    for (const FDiscreteActionSpec& DiscreteAction : EnvInfo.ActionSpace->DiscreteActions)
    {
        DiscreteActionsArray.Add(MakeShareable(new FJsonValueNumber(DiscreteAction.NumChoices)));
    }
    ActionSpaceObject->SetArrayField(TEXT("DiscreteActions"), DiscreteActionsArray);

    EnvironmentInfoObject->SetObjectField(TEXT("ActionSpace"), ActionSpaceObject);

    // Add EnvironmentInfo to the root object
    JsonObject->SetObjectField(TEXT("EnvironmentInfo"), EnvironmentInfoObject);

    // TrainInfo object
    TSharedPtr<FJsonObject> TrainInfoObject = MakeShareable(new FJsonObject);
    TrainInfoObject->SetNumberField(TEXT("BufferSize"), TrainParams.BufferSize);
    TrainInfoObject->SetNumberField(TEXT("BatchSize"), TrainParams.BatchSize);
    TrainInfoObject->SetNumberField(TEXT("NumEnvironments"), TrainParams.NumEnvironments);
    TrainInfoObject->SetNumberField(TEXT("MaxAgents"), TrainParams.MaxAgents);

    // Add TrainInfo to the root object
    JsonObject->SetObjectField(TEXT("TrainInfo"), TrainInfoObject);

    // Serialize the JSON object to string
    FString OutputString;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&OutputString);
    if (FJsonSerializer::Serialize(JsonObject.ToSharedRef(), Writer))
    {
        // Convert FString to TArray<char>
        TArray<char> CharArray;
        CharArray.Append(reinterpret_cast<const char*>(TCHAR_TO_UTF8(*OutputString)), OutputString.Len());
        return CharArray;
    }

    // Return an empty array if serialization fails
    return TArray<char>();
}