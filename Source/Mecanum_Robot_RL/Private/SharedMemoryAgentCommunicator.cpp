// Copyright Epic Games, Inc. All Rights Reserved.
// NOTICE: This file includes modifications generated with the assistance of generative AI.
// Original code structure and logic by the project author.

#include "SharedMemoryAgentCommunicator.h"
#include "Misc/FileHelper.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h" // Required for FJsonValueObject if used directly, but UEnvironmentConfig abstracts it

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
    , SingleEnvStateSize(0) // This will be calculated based on new central state structure
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
    int32 MaxAgents = 1; // Default to 1 if not multi-agent
    if (IsMultiAgent() && LocalEnvConfig->HasPath(TEXT("environment/shape/state/agent/max"))) // Check path before getting
    {
        MaxAgents = LocalEnvConfig->Get(TEXT("environment/shape/state/agent/max"))->AsInt();
    }

    // SingleEnvStateSize will now be calculated based on the sum of enabled central components + agent states
    SingleEnvStateSize = ComputeSingleEnvStateSize(MaxAgents);
    TotalActionCount = ComputeSingleEnvActionSize(MaxAgents);

    UE_LOG(LogTemp, Log, TEXT("SharedMemoryAgentCommunicator::Init - Calculated SingleEnvStateSize: %d, TotalActionCount: %d for MaxAgents: %d"), SingleEnvStateSize, TotalActionCount, MaxAgents);

    // 3) Derive memory sizes (in bytes)
    ActionMAXSize = NumEnvironments * (TotalActionCount * sizeof(float));

    // StatesMAXSize: Includes overhead for info floats (6) + done/trunc per environment (2)
    int32 InfoFloatsSize = 6 * sizeof(float);
    int32 TerminalsPerEnvSize = 2 * sizeof(float); // done + trunc
    StatesMAXSize = (NumEnvironments * (SingleEnvStateSize * sizeof(float))) + InfoFloatsSize + (NumEnvironments * TerminalsPerEnvSize);

    // UpdateMAXSize: Based on (state + nextState + action + reward + trunc + done) per transition
    int32 SingleTransitionFloatCount = (SingleEnvStateSize * 2) + TotalActionCount + 3; // 3 for reward, trunc, done
    UpdateMAXSize = NumEnvironments * BatchSize * SingleTransitionFloatCount * sizeof(float);

    UE_LOG(LogTemp, Log, TEXT("SharedMemoryAgentCommunicator::Init - ActionMAXSize: %d B, StatesMAXSize: %d B, UpdateMAXSize: %d B"), ActionMAXSize, StatesMAXSize, UpdateMAXSize);


    // 4) Create the shared memory sections
    ActionsSharedMemoryHandle = CreateFileMapping(
        INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, ActionMAXSize > 0 ? ActionMAXSize : 1, TEXT("ActionsSharedMemory")); // Ensure size is > 0
    if (!ActionsSharedMemoryHandle) { UE_LOG(LogTemp, Error, TEXT("Failed to create actions shared memory. Error: %d"), GetLastError()); }

    StatesSharedMemoryHandle = CreateFileMapping(
        INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, StatesMAXSize > 0 ? StatesMAXSize : 1, TEXT("StatesSharedMemory"));
    if (!StatesSharedMemoryHandle) { UE_LOG(LogTemp, Error, TEXT("Failed to create states shared memory. Error: %d"), GetLastError()); }

    UpdateSharedMemoryHandle = CreateFileMapping(
        INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, UpdateMAXSize > 0 ? UpdateMAXSize : 1, TEXT("UpdateSharedMemory"));
    if (!UpdateSharedMemoryHandle) { UE_LOG(LogTemp, Error, TEXT("Failed to create update shared memory. Error: %d"), GetLastError()); }

    // 5) Create mutexes
    ActionsMutexHandle = CreateMutex(NULL, false, TEXT("ActionsDataMutex"));
    StatesMutexHandle = CreateMutex(NULL, false, TEXT("StatesDataMutex"));
    UpdateMutexHandle = CreateMutex(NULL, false, TEXT("UpdateDataMutex"));
    if (!ActionsMutexHandle || !StatesMutexHandle || !UpdateMutexHandle) { UE_LOG(LogTemp, Error, TEXT("Failed to create one or more data mutexes. Error: %d"), GetLastError()); }

    // 6) Create event objects
    ActionReadyEventHandle = CreateEvent(NULL, false, false, TEXT("ActionReadyEvent"));
    ActionReceivedEventHandle = CreateEvent(NULL, false, false, TEXT("ActionReceivedEvent"));
    UpdateReadyEventHandle = CreateEvent(NULL, false, false, TEXT("UpdateReadyEvent"));
    UpdateReceivedEventHandle = CreateEvent(NULL, false, false, TEXT("UpdateReceivedEvent"));
    if (!ActionReadyEventHandle || !ActionReceivedEventHandle || !UpdateReadyEventHandle || !UpdateReceivedEventHandle) { UE_LOG(LogTemp, Error, TEXT("Failed to create one or more events. Error: %d"), GetLastError()); }

    // 7) Map the memory
    if (ActionsSharedMemoryHandle && ActionMAXSize > 0) {
        MappedActionsSharedData = (float*)MapViewOfFile(ActionsSharedMemoryHandle, FILE_MAP_ALL_ACCESS, 0, 0, ActionMAXSize);
        if (!MappedActionsSharedData) { UE_LOG(LogTemp, Error, TEXT("Failed to map actions shared memory. Error: %d"), GetLastError()); }
    }
    else if (ActionMAXSize == 0) {
        UE_LOG(LogTemp, Warning, TEXT("ActionMAXSize is 0, skipping mapping for ActionsSharedMemory."));
    }


    if (StatesSharedMemoryHandle && StatesMAXSize > 0) {
        MappedStatesSharedData = (float*)MapViewOfFile(StatesSharedMemoryHandle, FILE_MAP_ALL_ACCESS, 0, 0, StatesMAXSize);
        if (!MappedStatesSharedData) { UE_LOG(LogTemp, Error, TEXT("Failed to map states shared memory. Error: %d"), GetLastError()); }
    }
    else if (StatesMAXSize == 0) {
        UE_LOG(LogTemp, Warning, TEXT("StatesMAXSize is 0, skipping mapping for StatesSharedMemory."));
    }

    if (UpdateSharedMemoryHandle && UpdateMAXSize > 0) {
        MappedUpdateSharedData = (float*)MapViewOfFile(UpdateSharedMemoryHandle, FILE_MAP_ALL_ACCESS, 0, 0, UpdateMAXSize);
        if (!MappedUpdateSharedData) { UE_LOG(LogTemp, Error, TEXT("Failed to map update shared memory. Error: %d"), GetLastError()); }
    }
    else if (UpdateMAXSize == 0) {
        UE_LOG(LogTemp, Warning, TEXT("UpdateMAXSize is 0, skipping mapping for UpdateSharedMemory."));
    }

    UE_LOG(LogTemp, Log, TEXT("USharedMemoryAgentCommunicator::Init complete."));
}

TArray<FAction> USharedMemoryAgentCommunicator::GetActions(
    TArray<FState> States,
    TArray<float> Dones,
    TArray<float> Truncs,
    int CurrentNumAgents // Renamed from NumAgents to avoid conflict with member
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

    if (WaitForSingleObject(StatesMutexHandle, INFINITE) != WAIT_OBJECT_0)
    {
        UE_LOG(LogTemp, Error, TEXT("GetActions - Failed to lock States mutex."));
        return OutActions;
    }

    // Calculate state and action sizes for the current number of agents
    int32 CurrentStepEnvStateSize = ComputeSingleEnvStateSize(CurrentNumAgents);
    int32 CurrentStepActionSize = ComputeSingleEnvActionSize(CurrentNumAgents);

    // Write 6 info floats
    MappedStatesSharedData[0] = static_cast<float>(BufferSize);
    MappedStatesSharedData[1] = static_cast<float>(BatchSize);
    MappedStatesSharedData[2] = static_cast<float>(NumEnvironments);
    MappedStatesSharedData[3] = IsMultiAgent() ? static_cast<float>(CurrentNumAgents) : -1.f;
    MappedStatesSharedData[4] = static_cast<float>(CurrentStepEnvStateSize);
    MappedStatesSharedData[5] = static_cast<float>(CurrentStepActionSize);

    float* p = MappedStatesSharedData + 6; // Start writing after the info floats

    // Write states
    for (const FState& st : States)
    {
        if (st.Values.Num() != CurrentStepEnvStateSize) {
            UE_LOG(LogTemp, Error, TEXT("GetActions - State size mismatch for an environment! Expected: %d, Got: %d. Check UStateManager and JSON config consistency."), CurrentStepEnvStateSize, st.Values.Num());
            // Handle error: skip this state, return empty, or use a default? For now, we'll continue writing but this is problematic.
        }
        FMemory::Memcpy(p, st.Values.GetData(), st.Values.Num() * sizeof(float));
        p += st.Values.Num();
    }

    // Write Dones
    FMemory::Memcpy(p, Dones.GetData(), Dones.Num() * sizeof(float));
    p += Dones.Num();

    // Write Truncs
    FMemory::Memcpy(p, Truncs.GetData(), Truncs.Num() * sizeof(float));
    // p += Truncs.Num(); // Not needed as it's the last write to this section

    ReleaseMutex(StatesMutexHandle);
    SetEvent(ActionReadyEventHandle);

    WaitForSingleObject(ActionReceivedEventHandle, INFINITE);

    if (WaitForSingleObject(ActionsMutexHandle, INFINITE) != WAIT_OBJECT_0)
    {
        UE_LOG(LogTemp, Error, TEXT("GetActions - Failed to lock Actions mutex."));
        return OutActions;
    }

    float* ActionPtr = MappedActionsSharedData;
    OutActions.Reserve(States.Num());
    for (int32 i = 0; i < States.Num(); i++)
    {
        FAction OneAction;
        OneAction.Values.SetNum(CurrentStepActionSize);
        if (CurrentStepActionSize > 0) { // Avoid memcpy if size is 0
            FMemory::Memcpy(OneAction.Values.GetData(), ActionPtr, CurrentStepActionSize * sizeof(float));
            ActionPtr += CurrentStepActionSize;
        }
        OutActions.Add(OneAction);
    }

    ReleaseMutex(ActionsMutexHandle);
    return OutActions;
}

void USharedMemoryAgentCommunicator::Update(const TArray<FExperienceBatch>& experiences, int CurrentNumAgents)
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
    int32 index = 0;

    // Calculate sizes based on the current number of agents for this batch of experiences
    int32 CurrentStepEnvStateSize = ComputeSingleEnvStateSize(CurrentNumAgents);
    int32 CurrentStepActionSize = ComputeSingleEnvActionSize(CurrentNumAgents);

    for (const FExperienceBatch& batch : experiences)
    {
        for (const FExperience& xp : batch.Experiences)
        {
            // Verify sizes before writing
            if (xp.State.Values.Num() != CurrentStepEnvStateSize || xp.NextState.Values.Num() != CurrentStepEnvStateSize) {
                UE_LOG(LogTemp, Error, TEXT("Update - Experience state/nextState size mismatch! Expected: %d, Got State: %d, NextState: %d. Check UStateManager and JSON config consistency."),
                    CurrentStepEnvStateSize, xp.State.Values.Num(), xp.NextState.Values.Num());
                // Decide error handling: skip this experience, or pad/truncate (more complex)
                continue;
            }
            if (xp.Action.Values.Num() != CurrentStepActionSize) {
                UE_LOG(LogTemp, Error, TEXT("Update - Experience action size mismatch! Expected: %d, Got: %d."), CurrentStepActionSize, xp.Action.Values.Num());
                continue;
            }

            FMemory::Memcpy(&ptr[index], xp.State.Values.GetData(), xp.State.Values.Num() * sizeof(float));
            index += xp.State.Values.Num();

            FMemory::Memcpy(&ptr[index], xp.NextState.Values.GetData(), xp.NextState.Values.Num() * sizeof(float));
            index += xp.NextState.Values.Num();

            if (CurrentStepActionSize > 0) { // Only copy if there are actions
                FMemory::Memcpy(&ptr[index], xp.Action.Values.GetData(), xp.Action.Values.Num() * sizeof(float));
                index += xp.Action.Values.Num();
            }

            ptr[index++] = xp.Reward;
            ptr[index++] = xp.Trunc ? 1.f : 0.f;
            ptr[index++] = xp.Done ? 1.f : 0.f;
        }
    }

    ReleaseMutex(UpdateMutexHandle);
    SetEvent(UpdateReadyEventHandle);
    WaitForSingleObject(UpdateReceivedEventHandle, INFINITE);
}

void USharedMemoryAgentCommunicator::WriteTransitionsToFile(
    const TArray<FExperienceBatch>& experiences,
    const FString& FilePath
)
{
    FString Output;
    for (const FExperienceBatch& batch : experiences)
    {
        for (const FExperience& xp : batch.Experiences)
        {
            for (float val : xp.State.Values) { Output.Append(FString::SanitizeFloat(val)); Output.AppendChar(','); }
            for (float val : xp.NextState.Values) { Output.Append(FString::SanitizeFloat(val)); Output.AppendChar(','); }
            for (float aVal : xp.Action.Values) { Output.Append(FString::SanitizeFloat(aVal)); Output.AppendChar(','); }
            Output.Append(FString::SanitizeFloat(xp.Reward)); Output.AppendChar(',');
            Output.Append(FString::SanitizeFloat(xp.Trunc ? 1.f : 0.f)); Output.AppendChar(',');
            Output.Append(FString::SanitizeFloat(xp.Done ? 1.f : 0.f)); Output.AppendChar('\n');
        }
    }
    if (!FFileHelper::SaveStringToFile(Output, *FilePath)) { UE_LOG(LogTemp, Warning, TEXT("WriteTransitionsToFile - Could not save to file: %s"), *FilePath); }
    else { UE_LOG(LogTemp, Log, TEXT("WriteTransitionsToFile - Wrote transitions to file: %s"), *FilePath); }
}

// -----------------------------------
// Private Helpers
// -----------------------------------

bool USharedMemoryAgentCommunicator::IsMultiAgent() const
{
    return (LocalEnvConfig && LocalEnvConfig->HasPath(TEXT("environment/shape/state/agent")));
}

// MODIFIED: No longer just checks for path, but for a valid array structure if the new format is intended.
// However, ComputeSingleEnvStateSize effectively determines if central state contributes to size.
bool USharedMemoryAgentCommunicator::HasCentralState() const
{
    if (!LocalEnvConfig || !LocalEnvConfig->HasPath(TEXT("environment/shape/state/central")))
    {
        return false;
    }
    // Check if "central" is an array (new format) or an object (old format with "obs_size")
    UEnvironmentConfig* CentralNode = LocalEnvConfig->Get(TEXT("environment/shape/state/central"));
    if (CentralNode && CentralNode->IsValid()) {
        if (CentralNode->InternalJsonValue->Type == EJson::Array) {
            // New format: array of components. Consider it "has central state" if array is not empty.
            // Actual contribution to size is determined by enabled components in ComputeSingleEnvStateSize.
            return CentralNode->AsArrayOfConfigs().Num() > 0;
        }
        else if (CentralNode->InternalJsonValue->Type == EJson::Object && CentralNode->HasPath(TEXT("obs_size"))) {
            // Old format: object with obs_size.
            return CentralNode->Get(TEXT("obs_size"))->AsInt() > 0;
        }
    }
    return false; // Path exists but content is not a recognized central state definition
}


int32 USharedMemoryAgentCommunicator::ComputeSingleEnvStateSize(int32 NumAgentsToConsider) const
{
    int32 TotalSize = 0;
    if (!LocalEnvConfig) return 0;

    if (LocalEnvConfig->HasPath(TEXT("environment/shape/state/central")))
    {
        UEnvironmentConfig* CentralConfigNode = LocalEnvConfig->Get(TEXT("environment/shape/state/central"));
        if (CentralConfigNode && CentralConfigNode->IsValid())
        {
            if (CentralConfigNode->InternalJsonValue->Type == EJson::Array) // New format
            {
                TArray<UEnvironmentConfig*> CentralComponents = CentralConfigNode->AsArrayOfConfigs();
                for (UEnvironmentConfig* CompConfig : CentralComponents)
                {
                    if (CompConfig && CompConfig->IsValid() && CompConfig->GetOrDefaultBool(TEXT("enabled"), false))
                    {
                        FString CompType = CompConfig->GetOrDefaultString(TEXT("type"), TEXT(""));
                        UEnvironmentConfig* ShapeConfig = CompConfig->Get(TEXT("shape"));

                        if (!ShapeConfig || !ShapeConfig->IsValid()) {
                            UE_LOG(LogTemp, Warning, TEXT("ComputeSingleEnvStateSize: Central component '%s' is enabled but missing valid 'shape' object."), *CompConfig->GetOrDefaultString(TEXT("name"), TEXT("Unknown")));
                            continue;
                        }

                        if (CompType.Equals(TEXT("matrix2d"), ESearchCase::IgnoreCase))
                        {
                            int32 h = ShapeConfig->GetOrDefaultInt(TEXT("h"), 0);
                            int32 w = ShapeConfig->GetOrDefaultInt(TEXT("w"), 0);
                            // int32 c = ShapeConfig->GetOrDefaultInt(TEXT("c"), 1); // Assuming c=1 or channels handled by flat size
                            TotalSize += (h * w);
                        }
                        else if (CompType.Equals(TEXT("vector"), ESearchCase::IgnoreCase))
                        {
                            TotalSize += ShapeConfig->GetOrDefaultInt(TEXT("size"), 0);
                        }
                        // ***** ADD THIS ELSE IF BLOCK for "sequence" *****
                        else if (CompType.Equals(TEXT("sequence"), ESearchCase::IgnoreCase))
                        {
                            int32 max_length = ShapeConfig->GetOrDefaultInt(TEXT("max_length"), 0);
                            int32 feature_dim = ShapeConfig->GetOrDefaultInt(TEXT("feature_dim"), 0);
                            TotalSize += (max_length * feature_dim);
                        }
                        // ***** END OF ADDED BLOCK *****
                    }
                }
            }
            else if (CentralConfigNode->InternalJsonValue->Type == EJson::Object && CentralConfigNode->HasPath(TEXT("obs_size"))) // Fallback to old format
            {
                UE_LOG(LogTemp, Warning, TEXT("ComputeSingleEnvStateSize: 'central' is in old object format. Using 'obs_size'. Please update config."));
                TotalSize += CentralConfigNode->Get(TEXT("obs_size"))->AsInt();
            }
            else
            {
                UE_LOG(LogTemp, Warning, TEXT("ComputeSingleEnvStateSize: 'environment/shape/state/central' exists but is not a valid array or recognized old object format. Central state size contributions will be 0."));
            }
        }
    }

    if (IsMultiAgent() && LocalEnvConfig->HasPath(TEXT("environment/shape/state/agent/obs_size")))
    {
        int32 AgentObsSize = LocalEnvConfig->Get(TEXT("environment/shape/state/agent/obs_size"))->AsInt();
        TotalSize += AgentObsSize * NumAgentsToConsider;
    }
    return TotalSize;
}

// Computes total action dimensions per agent (if multi-agent) or for the single agent.
// The result is then multiplied by NumAgentsToConsider if multi-agent.
int32 USharedMemoryAgentCommunicator::ComputeSingleEnvActionSize(int32 NumAgentsToConsider) const
{
    if (!LocalEnvConfig) return 0;

    bool bIsActuallyMultiAgent = IsMultiAgent(); // Use the helper that checks config structure
    FString ActionBasePathKey = bIsActuallyMultiAgent ? TEXT("environment/shape/action/agent") : TEXT("environment/shape/action/central");

    if (!LocalEnvConfig->HasPath(ActionBasePathKey)) {
        // If the primary path (agent or central) doesn't exist, try the other as a fallback, or assume 0.
        // This handles cases where config might be transitioning.
        FString FallbackPathKey = bIsActuallyMultiAgent ? TEXT("environment/shape/action/central") : TEXT("environment/shape/action/agent");
        if (LocalEnvConfig->HasPath(FallbackPathKey)) {
            ActionBasePathKey = FallbackPathKey;
            // Log a warning about using fallback if appropriate
            UE_LOG(LogTemp, Warning, TEXT("ComputeSingleEnvActionSize: Primary action path missing, using fallback: %s"), *FallbackPathKey);
        }
        else {
            UE_LOG(LogTemp, Warning, TEXT("ComputeSingleEnvActionSize: No action definition found at '%s' or fallback path."), *ActionBasePathKey);
            return 0;
        }
    }

    UEnvironmentConfig* ActionConfigNode = LocalEnvConfig->Get(ActionBasePathKey);
    if (!ActionConfigNode || !ActionConfigNode->IsValid()) return 0;

    int32 DiscreteActionBranches = 0;
    if (ActionConfigNode->HasPath(TEXT("discrete")))
    {
        UEnvironmentConfig* DiscreteNode = ActionConfigNode->Get(TEXT("discrete"));
        TArray<UEnvironmentConfig*> DiscreteArray = DiscreteNode->AsArrayOfConfigs();
        DiscreteActionBranches = DiscreteArray.Num(); // Each object in array is one discrete branch
    }

    int32 ContinuousActionDims = 0;
    if (ActionConfigNode->HasPath(TEXT("continuous")))
    {
        UEnvironmentConfig* ContinuousNode = ActionConfigNode->Get(TEXT("continuous"));
        TArray<UEnvironmentConfig*> ContinuousArray = ContinuousNode->AsArrayOfConfigs();
        ContinuousActionDims = ContinuousArray.Num(); // Each object in array is one continuous dimension/action
    }

    int32 PerAgentActionComponents = DiscreteActionBranches + ContinuousActionDims;

    // If it's multi-agent (structurally), multiply by the number of agents being considered.
    // If not multi-agent, NumAgentsToConsider should ideally be 1, but this structure handles it.
    return bIsActuallyMultiAgent ? (PerAgentActionComponents * NumAgentsToConsider) : PerAgentActionComponents;
}