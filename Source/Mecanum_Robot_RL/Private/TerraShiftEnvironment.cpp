#include "TerraShiftEnvironment.h"
#include "Engine/World.h"
#include "TimerManager.h"

// Constructor
ATerraShiftEnvironment::ATerraShiftEnvironment() {
    PrimaryActorTick.bCanEverTick = true;
    PrimaryActorTick.TickGroup = TG_PostPhysics;
    EnvInfo.EnvID = 3;
    EnvInfo.IsMultiAgent = true;
    EnvInfo.ActionSpace = CreateDefaultSubobject<UActionSpace>(TEXT("ActionSpace"));

    TerraShiftRoot = CreateDefaultSubobject<USceneComponent>(TEXT("TerraShiftRoot"));
    RootComponent = TerraShiftRoot;

    GridObjectManager = CreateDefaultSubobject<AGridObjectManager>(TEXT("GridObjectManager"));
    WaveSimulator = nullptr;
    Grid = nullptr;
    Intialized = false;

    // Initialize RewardBuffer
    RewardBuffer = 0.0f;

    GoalColors = {
        FLinearColor(0.388f, 0.192f, 0.0f),   // Umber
        FLinearColor(0.576f, 0.353f, 0.243f), // Cacao
        FLinearColor(0.714f, 0.537f, 0.404f), // Mocha
        FLinearColor(0.690f, 0.255f, 0.051f), // Rust
        FLinearColor(0.545f, 0.271f, 0.075f), // Saddle Brown
        FLinearColor(0.824f, 0.412f, 0.118f), // Terracotta
        FLinearColor(0.761f, 0.698f, 0.502f), // Sand
        FLinearColor(0.824f, 0.706f, 0.549f), // Latte
        FLinearColor(0.282f, 0.235f, 0.196f), // Taupe
        FLinearColor(0.941f, 0.894f, 0.835f)  // Plaster
    };
}

void ATerraShiftEnvironment::Tick(float DeltaTime) {
    Super::Tick(DeltaTime);

    if (Intialized) {
        // Update active columns after physics simulation
        UpdateActiveColumns();

        // Update column and GridObject colors
        UpdateColumnGoalObjectColors();

        // Check and respawn GridObjects if necessary
        CheckAndRespawnGridObjects();
    }
}

ATerraShiftEnvironment::~ATerraShiftEnvironment() {
    if (WaveSimulator) {
        delete WaveSimulator;
        WaveSimulator = nullptr;
    }
}

void ATerraShiftEnvironment::InitEnv(FBaseInitParams* BaseParams) {
    TerraShiftParams = static_cast<FTerraShiftEnvironmentInitParams*>(BaseParams);

    MaxAgents = TerraShiftParams->MaxAgents;
    CurrentAgents = BaseParams->NumAgents;
    CurrentStep = 0;

    // Set up observation and action space
    SetupActionAndObservationSpace();

    // Initialize AgentParametersArray to match CurrentAgents
    AgentParametersArray.SetNum(CurrentAgents);
    AgentGoalIndices.SetNum(CurrentAgents);
    AgentHasActiveGridObject.SetNum(CurrentAgents);

    // Set the environment root's world location
    TerraShiftRoot->SetWorldLocation(TerraShiftParams->Location);

    // Spawn the platform at the specified location
    Platform = SpawnPlatform(TerraShiftParams->Location);
    Platform->AttachToActor(this, FAttachmentTransformRules::KeepRelativeTransform);

    // Scale the platform based on the specified PlatformSize
    Platform->SetActorScale3D(FVector(TerraShiftParams->PlatformSize, TerraShiftParams->PlatformSize, 1.0f));

    // Calculate platform dimensions and determine grid cell size
    PlatformWorldSize = Platform->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().BoxExtent * 2.0f * Platform->GetActorScale3D();
    PlatformCenter = Platform->GetActorLocation();
    CellSize = PlatformWorldSize.X / static_cast<float>(TerraShiftParams->GridSize);

    // Initialize GridCenterPoints to match the grid size
    GridCenterPoints.SetNum(TerraShiftParams->GridSize * TerraShiftParams->GridSize);

    // Calculate grid center points
    for (int32 X = 0; X < TerraShiftParams->GridSize; ++X) {
        for (int32 Y = 0; Y < TerraShiftParams->GridSize; ++Y) {
            // Calculate the center location of each column in the grid
            float CenterX = PlatformCenter.X - (PlatformWorldSize.X / 2.0f) + (X * CellSize) + (CellSize / 2.0f);
            float CenterY = PlatformCenter.Y - (PlatformWorldSize.Y / 2.0f) + (Y * CellSize) + (CellSize / 2.0f);
            float CenterZ = PlatformCenter.Z;

            int32 Index = X * TerraShiftParams->GridSize + Y;
            GridCenterPoints[Index] = FVector(CenterX, CenterY, CenterZ);
        }
    }

    // Initialize the grid at the center of the platform, slightly elevated
    FVector GridLocation = PlatformCenter;
    GridLocation.Z += 1.00; // Slightly above the platform

    FActorSpawnParameters SpawnParams;
    SpawnParams.Owner = this;
    Grid = GetWorld()->SpawnActor<AGrid>(AGrid::StaticClass(), GridLocation, FRotator::ZeroRotator, SpawnParams);
    if (Grid) {
        Grid->AttachToActor(Platform, FAttachmentTransformRules::KeepRelativeTransform);
        Grid->InitializeGrid(
            TerraShiftParams->GridSize,
            PlatformWorldSize.X,
            GridLocation
        );
    }

    // Initialize GridObjectManager and attach it to TerraShiftRoot
    GridObjectManager = GetWorld()->SpawnActor<AGridObjectManager>(AGridObjectManager::StaticClass());
    if (GridObjectManager) {
        GridObjectManager->AttachToActor(Platform, FAttachmentTransformRules::KeepRelativeTransform);
        GridObjectManager->SetObjectSize(TerraShiftParams->ObjectSize);
        // Bind to the GridObjectManager's event
        GridObjectManager->OnGridObjectSpawned.AddDynamic(this, &ATerraShiftEnvironment::OnGridObjectSpawned);
    }

    // Initialize wave simulator
    WaveSimulator = new MorletWavelets2D(TerraShiftParams->GridSize, TerraShiftParams->GridSize, CellSize);
    WaveSimulator->Initialize();

    // Initialize agent parameters for the current number of agents
    for (int32 i = 0; i < CurrentAgents; ++i) {
        AgentParameters AgentParam;
        AgentParam.Position = FVector2f(FMath::FRandRange(0.0f, static_cast<float>(TerraShiftParams->GridSize)),
            FMath::FRandRange(0.0f, static_cast<float>(TerraShiftParams->GridSize)));
        AgentParam.Velocity = FVector2f(0.0f, 0.0f);
        AgentParam.Amplitude = FMath::FRandRange(TerraShiftParams->AmplitudeRange.X, TerraShiftParams->AmplitudeRange.Y);
        AgentParam.WaveOrientation = FMath::FRandRange(TerraShiftParams->WaveOrientationRange.X, TerraShiftParams->WaveOrientationRange.Y);
        AgentParam.Wavenumber = FMath::FRandRange(TerraShiftParams->WavenumberRange.X, TerraShiftParams->WavenumberRange.Y);
        AgentParam.Phase = FMath::FRandRange(TerraShiftParams->PhaseRange.X, TerraShiftParams->PhaseRange.Y);
        AgentParam.Sigma = FMath::FRandRange(TerraShiftParams->SigmaRange.X, TerraShiftParams->SigmaRange.Y);
        AgentParam.Time = 0.0f;
        AgentParam.Frequency = AgentParam.Wavenumber * WaveSimulator->GetPhaseVelocity();

        AgentParametersArray[i] = AgentParam;

        // Randomly assign a goal index for each agent
        AgentGoalIndices[i] = FMath::RandRange(0, TerraShiftParams->NumGoals - 1);

        // Set AgentHasActiveGridObject to false initially as grid objects spawn over time
        AgentHasActiveGridObject[i] = false;
    }

    Intialized = true;
}

FState ATerraShiftEnvironment::ResetEnv(int NumAgents) {
    CurrentStep = 0;
    CurrentAgents = NumAgents;

    // Reset AgentParametersArray to match CurrentAgents
    AgentParametersArray.SetNum(CurrentAgents);
    AgentGoalIndices.SetNum(CurrentAgents);
    AgentHasActiveGridObject.SetNum(CurrentAgents);

    // Reset RewardBuffer
    RewardBuffer = 0.0f;

    // Reset goals, grid, and grid objects
    GoalPositionArray.Empty();
    GoalPositionArray.SetNum(TerraShiftParams->NumGoals);

    // Generate random goal positions along the four corners
    for (int32 i = 0; i < TerraShiftParams->NumGoals; ++i) {
        FVector GoalPosition = GenerateRandomCornerGridLocation();
        GoalPositionArray[i] = GoalPosition;
    }

    if (Grid) {
        Grid->ResetGrid();
    }

    if (GridObjectManager) {
        GridObjectManager->ResetGridObjects();
    }

    // Reinitialize agent parameters
    for (int32 i = 0; i < CurrentAgents; ++i) {
        AgentParameters AgentParam;
        AgentParam.Position = FVector2f(FMath::FRandRange(0.0f, static_cast<float>(TerraShiftParams->GridSize)),
            FMath::FRandRange(0.0f, static_cast<float>(TerraShiftParams->GridSize)));
        AgentParam.Velocity = FVector2f(0.0f, 0.0f);
        AgentParam.Amplitude = FMath::FRandRange(TerraShiftParams->AmplitudeRange.X, TerraShiftParams->AmplitudeRange.Y);
        AgentParam.WaveOrientation = FMath::FRandRange(TerraShiftParams->WaveOrientationRange.X, TerraShiftParams->WaveOrientationRange.Y);
        AgentParam.Wavenumber = FMath::FRandRange(TerraShiftParams->WavenumberRange.X, TerraShiftParams->WavenumberRange.Y);
        AgentParam.Phase = FMath::FRandRange(TerraShiftParams->PhaseRange.X, TerraShiftParams->PhaseRange.Y);
        AgentParam.Sigma = FMath::FRandRange(TerraShiftParams->SigmaRange.X, TerraShiftParams->SigmaRange.Y);
        AgentParam.Time = 0.0f;
        AgentParam.Frequency = AgentParam.Wavenumber * WaveSimulator->GetPhaseVelocity();

        AgentParametersArray[i] = AgentParam;

        // Reassign goal indices for each agent
        AgentGoalIndices[i] = FMath::RandRange(0, TerraShiftParams->NumGoals - 1);

        // Set AgentHasActiveGridObject to false initially
        AgentHasActiveGridObject[i] = false;
    }

    // Set active grid objects for the new agents
    SetActiveGridObjects(CurrentAgents);

    // Reset the wave simulator
    WaveSimulator->Reset();

    return State();
}

TArray<float> ATerraShiftEnvironment::AgentGetState(int AgentIndex) {
    TArray<float> State;

    // Ensure AgentIndex is within bounds
    if (!AgentParametersArray.IsValidIndex(AgentIndex)) {
        UE_LOG(LogTemp, Error, TEXT("AgentGetState: Invalid AgentIndex %d"), AgentIndex);
        return State;
    }

    const AgentParameters& Agent = AgentParametersArray[AgentIndex];

    // Get the world position of the grid object for the agent
    FVector ObjectWorldPosition = FVector::ZeroVector;
    if (AgentHasActiveGridObject[AgentIndex] && GridObjectManager) {
        ObjectWorldPosition = GridObjectManager->GetGridObjectWorldLocation(AgentIndex);
    }

    FVector ObjectRelativePosition = Platform->GetActorTransform().InverseTransformPosition(ObjectWorldPosition);

    float AgentPosX_Grid = Agent.Position.X;
    float AgentPosY_Grid = Agent.Position.Y;
    FVector AgentWorldPosition = GridPositionToWorldPosition(FVector2D(AgentPosX_Grid, AgentPosY_Grid));
    FVector AgentRelativePosition = Platform->GetActorTransform().InverseTransformPosition(AgentWorldPosition);

    int AgentGoalIndex = AgentGoalIndices[AgentIndex];
    FVector GoalWorldPosition = GoalPositionArray[AgentGoalIndex];
    FVector GoalRelativePosition = Platform->GetActorTransform().InverseTransformPosition(GoalWorldPosition);

    State.Add(AgentRelativePosition.X);
    State.Add(AgentRelativePosition.Y);
    State.Add(Agent.Velocity.X);
    State.Add(Agent.Velocity.Y);
    State.Add(Agent.Amplitude);
    State.Add(Agent.WaveOrientation);
    State.Add(Agent.Wavenumber);
    State.Add(Agent.Frequency);
    State.Add(Agent.Phase);
    State.Add(Agent.Sigma);
    State.Add(Agent.Time);
    State.Add(ObjectRelativePosition.X);
    State.Add(ObjectRelativePosition.Y);
    State.Add(ObjectRelativePosition.Z);
    State.Add(GoalRelativePosition.X);
    State.Add(GoalRelativePosition.Y);

    // Add a flag indicating if the GridObject is active
    State.Add(AgentHasActiveGridObject[AgentIndex] ? 1.0f : 0.0f);

    return State;
}

void ATerraShiftEnvironment::SetActiveGridObjects(int NumAgents) {
    if (!GridObjectManager) return;

    TArray<FVector> SpawnLocations;

    // Generate random spawn locations above the platform
    for (int i = 0; i < NumAgents; ++i) {
        FVector RandomLocation = GenerateRandomGridLocation();
        SpawnLocations.Add(RandomLocation);
    }

    // Spawn GridObjects at specified locations with a spawn delay
    GridObjectManager->SpawnGridObjects(SpawnLocations, TerraShiftParams->ObjectSize, TerraShiftParams->SpawnDelay);
}

void ATerraShiftEnvironment::Act(FAction Action) {
    const int NumAgentActions = EnvInfo.ActionSpace->ContinuousActions.Num();
    if (Action.Values.Num() != CurrentAgents * NumAgentActions) {
        UE_LOG(LogTemp, Error, TEXT("Action array size mismatch. Expected %d, got %d"), CurrentAgents * NumAgentActions, Action.Values.Num());
        return;
    }

    float DeltaTime = GetWorld()->GetDeltaSeconds();
    TArray<AgentParameters> AgentParametersArrayCopy;

    // Process agent actions and update agent parameters
    for (int i = 0; i < CurrentAgents; ++i) {
        int ActionIndex = i * NumAgentActions;

        float VelocityX = Map(Action.Values[ActionIndex], -1.0f, 1.0f, TerraShiftParams->VelocityRange.X, TerraShiftParams->VelocityRange.Y);
        float VelocityY = Map(Action.Values[ActionIndex + 1], -1.0f, 1.0f, TerraShiftParams->VelocityRange.X, TerraShiftParams->VelocityRange.Y);
        float Amplitude = Map(Action.Values[ActionIndex + 2], -1.0f, 1.0f, TerraShiftParams->AmplitudeRange.X, TerraShiftParams->AmplitudeRange.Y);
        float WaveOrientation = Map(Action.Values[ActionIndex + 3], -1.0f, 1.0f, TerraShiftParams->WaveOrientationRange.X, TerraShiftParams->WaveOrientationRange.Y);
        float Wavenumber = Map(Action.Values[ActionIndex + 4], -1.0f, 1.0f, TerraShiftParams->WavenumberRange.X, TerraShiftParams->WavenumberRange.Y);
        float Phase = Map(Action.Values[ActionIndex + 5], -1.0f, 1.0f, TerraShiftParams->PhaseRange.X, TerraShiftParams->PhaseRange.Y);
        float Sigma = Map(Action.Values[ActionIndex + 6], -1.0f, 1.0f, TerraShiftParams->SigmaRange.X, TerraShiftParams->SigmaRange.Y);

        AgentParameters& AgentParam = AgentParametersArray[i];

        AgentParam.Velocity = FVector2f(VelocityX, VelocityY);
        AgentParam.Amplitude = Amplitude;
        AgentParam.WaveOrientation = WaveOrientation;
        AgentParam.Wavenumber = Wavenumber;
        AgentParam.Phase = Phase;
        AgentParam.Sigma = Sigma;
        AgentParam.Position += AgentParam.Velocity * DeltaTime;

        // Keep agents within grid boundaries
        float GridSize = static_cast<float>(TerraShiftParams->GridSize);
        AgentParam.Position.X = FMath::Fmod(AgentParam.Position.X + GridSize, GridSize);
        AgentParam.Position.Y = FMath::Fmod(AgentParam.Position.Y + GridSize, GridSize);

        AgentParam.Time += DeltaTime;
        AgentParam.Frequency = AgentParam.Wavenumber * WaveSimulator->GetPhaseVelocity();

        AgentParametersArrayCopy.Add(AgentParam);
    }

    // Update wave heights using the current agent parameters
    const Matrix2D& HeightMap = WaveSimulator->Update(AgentParametersArrayCopy);

    if (Grid) {
        Grid->UpdateColumnHeights(HeightMap);
    }
}

void ATerraShiftEnvironment::UpdateActiveColumns() {
    if (!Grid || !GridObjectManager) {
        return; // Safety check
    }

    // Determine currently active columns based on proximity to GridObjects
    TSet<int32> NewActiveColumns = GridObjectManager->GetActiveColumnsInProximity(
        TerraShiftParams->GridSize, GridCenterPoints, PlatformCenter, PlatformWorldSize.X, CellSize
    );

    // Calculate columns that need to be toggled on or off
    TSet<int32> ColumnsToEnable = NewActiveColumns.Difference(ActiveColumns);
    TSet<int32> ColumnsToDisable = ActiveColumns.Difference(NewActiveColumns);

    // Enable or disable physics for relevant columns
    if (Grid && (ColumnsToEnable.Num() > 0 || ColumnsToDisable.Num() > 0)) {
        // Create arrays to match the indices for toggling physics
        TArray<int32> ColumnsToToggle;
        TArray<bool> EnablePhysics;

        // Add columns to enable physics
        for (int32 ColumnIndex : ColumnsToEnable) {
            ColumnsToToggle.Add(ColumnIndex);
            EnablePhysics.Add(true);
        }

        // Add columns to disable physics
        for (int32 ColumnIndex : ColumnsToDisable) {
            ColumnsToToggle.Add(ColumnIndex);
            EnablePhysics.Add(false);
        }

        // Toggle physics for the specified columns
        Grid->TogglePhysicsForColumns(ColumnsToToggle, EnablePhysics);
    }

    // Update the active columns set after toggling
    ActiveColumns = NewActiveColumns;
}

void ATerraShiftEnvironment::UpdateColumnGoalObjectColors() {
    if (!Grid || !GridObjectManager) {
        return;
    }

    // Map goal indices to colors
    TMap<int32, FLinearColor> GoalIndexToColor;
    int32 NumGoals = GoalPositionArray.Num();
    for (int32 i = 0; i < NumGoals; ++i) {
        FLinearColor Color = GoalColors[i % GoalColors.Num()];
        GoalIndexToColor.Add(i, Color);
    }

    // Highlight goal columns with their respective colors
    TSet<int32> GoalColumnIndices;
    for (int32 GoalIndex = 0; GoalIndex < NumGoals; ++GoalIndex) {
        FVector GoalPosition = GoalPositionArray[GoalIndex];
        int32 ClosestColumnIndex = FindClosestColumnIndex(GoalPosition, GridCenterPoints);
        if (ClosestColumnIndex != -1) {
            Grid->SetColumnColor(ClosestColumnIndex, GoalIndexToColor[GoalIndex]);
            GoalColumnIndices.Add(ClosestColumnIndex);
        }
    }

    // Set GridObjects to match their goal colors
    TArray<AGridObject*> GridObjects = GridObjectManager->GetActiveGridObjects();
    for (int32 i = 0; i < GridObjects.Num(); ++i) {
        AGridObject* GridObject = GridObjects[i];
        if (GridObject) {
            int32 GoalIndex = AgentGoalIndices[i];
            if (GoalIndexToColor.Contains(GoalIndex)) {
                FLinearColor GoalColor = GoalIndexToColor[GoalIndex];
                GridObject->SetGridObjectColor(GoalColor);
            }
        }
    }

    // Set active columns (physics enabled) to black
    for (int32 ColumnIndex : ActiveColumns) {
        Grid->SetColumnColor(ColumnIndex, FLinearColor::Black);
    }

    // Set other columns' colors based on their height (heat map from black to white)
    float MinHeight = Grid->GetMinHeight();
    float MaxHeight = Grid->GetMaxHeight();

    for (int32 ColumnIndex = 0; ColumnIndex < Grid->GetTotalColumns(); ++ColumnIndex) {
        if (ActiveColumns.Contains(ColumnIndex) || GoalColumnIndices.Contains(ColumnIndex)) {
            // Skip columns already colored
            continue;
        }
        float Height = Grid->GetColumnHeight(ColumnIndex);
        float HeightRatio = FMath::GetMappedRangeValueClamped(FVector2D(MinHeight, MaxHeight), FVector2D(0.0f, 1.0f), Height);
        FLinearColor Color = FLinearColor::LerpUsingHSV(FLinearColor::Black, FLinearColor::White, HeightRatio);
        Grid->SetColumnColor(ColumnIndex, Color);
    }
}

int32 ATerraShiftEnvironment::FindClosestColumnIndex(const FVector& Position, const TArray<FVector>& ColumnCenters) const {
    int32 ClosestIndex = -1;
    float MinDistance = FLT_MAX;

    for (int32 i = 0; i < ColumnCenters.Num(); ++i) {
        float Distance = FVector::Dist2D(Position, ColumnCenters[i]);
        if (Distance < MinDistance) {
            MinDistance = Distance;
            ClosestIndex = i;
        }
    }

    return ClosestIndex;
}

FState ATerraShiftEnvironment::State() {
    FState CurrentState;
    for (int i = 0; i < CurrentAgents; ++i) {
        CurrentState.Values.Append(AgentGetState(i));
    }
    return CurrentState;
}

void ATerraShiftEnvironment::PostTransition() {
    // Add any logic needed after transitions
}

void ATerraShiftEnvironment::PostStep() {
    CurrentStep += 1;
}

bool ATerraShiftEnvironment::Done() {
    // Check if all agents are inactive
    //for (bool bActive : AgentHasActiveGridObject) {
    //    if (bActive) {
    //        return false; // At least one agent is still active
    //    }
    //}
    //return true; // All agents are inactive
    return false;
}

bool ATerraShiftEnvironment::Trunc() {
    // Check if the environment should be truncated (e.g., max steps reached)
    if (CurrentStep >= TerraShiftParams->MaxSteps) {
        CurrentStep = 0;
        return true;
    }
    return false;
}

float ATerraShiftEnvironment::Reward() {
    // Return the total reward accumulated in the RewardBuffer and reset it
    float TotalReward = RewardBuffer;
    RewardBuffer = 0.0f;
    return TotalReward;
}

void ATerraShiftEnvironment::SetupActionAndObservationSpace() {
    const int NumAgentWaveParameters = 11;
    const int NumAgentStateParameters = NumAgentWaveParameters + 3 + 2 + 1; // Includes position, velocity, wave parameters, object position, goal position, and active flag
    EnvInfo.SingleAgentObsSize = NumAgentStateParameters;
    EnvInfo.StateSize = MaxAgents * EnvInfo.SingleAgentObsSize;

    const int NumAgentActions = 7; // VelocityX, VelocityY, Amplitude, WaveOrientation, Wavenumber, Phase, Sigma
    TArray<FContinuousActionSpec> ContinuousActions;
    for (int32 i = 0; i < NumAgentActions; ++i) {
        FContinuousActionSpec ActionSpec;
        ActionSpec.Low = -1.0f;
        ActionSpec.High = 1.0f;
        ContinuousActions.Add(ActionSpec);
    }

    if (EnvInfo.ActionSpace) {
        EnvInfo.ActionSpace->Init(ContinuousActions, {});
    }
}

AStaticMeshActor* ATerraShiftEnvironment::SpawnPlatform(FVector Location) {
    if (UWorld* World = GetWorld()) {
        UStaticMesh* PlaneMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Plane.Plane"));
        if (PlaneMesh) {
            FActorSpawnParameters SpawnParams;
            AStaticMeshActor* NewPlatform = World->SpawnActor<AStaticMeshActor>(AStaticMeshActor::StaticClass(), Location, FRotator::ZeroRotator, SpawnParams);
            if (NewPlatform) {
                // No Physics or collisions. Just Visual.
                NewPlatform->GetStaticMeshComponent()->SetStaticMesh(PlaneMesh);
                NewPlatform->SetMobility(EComponentMobility::Movable);
                NewPlatform->GetStaticMeshComponent()->SetSimulatePhysics(false);
                NewPlatform->SetActorEnableCollision(false);

                // Load and apply material to the platform
                UMaterial* Material = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/Material/Platform_Material.Platform_Material'"));
                if (Material) {
                    Material->TwoSided = true;
                    NewPlatform->GetStaticMeshComponent()->SetMaterial(0, Material);
                }
            }
            return NewPlatform;
        }
    }
    return nullptr;
}

FVector ATerraShiftEnvironment::GenerateRandomGridLocation() const {
    if (!Grid) {
        return FVector::ZeroVector; // Safety check
    }

    // Generate random X, Y grid coordinates
    int32 X = FMath::RandRange(0, TerraShiftParams->GridSize - 1);
    int32 Y = FMath::RandRange(0, TerraShiftParams->GridSize - 1);

    // Get the column's offsets (X, Y, Z) relative to the grid
    FVector ColumnOffsets = Grid->GetColumnOffsets(X, Y);

    // Calculate the corrective offsets based on the column size at (X, Y)
    FVector2D CorrectiveOffsets = Grid->CalculateEdgeCorrectiveOffsets(X, Y);

    // Get the grid's relative Z-location from its root component
    float GridRelativeZ = Grid->GetRootComponent()->GetRelativeLocation().Z;

    // Add the grid's relative Z-location to the column's Z offset
    float RelativeHeight = GridRelativeZ + ColumnOffsets.Z;

    // Calculate the final spawn location with the relative height and X, Y offsets
    FVector SpawnLocation = Grid->CalculateColumnLocation(X, Y, RelativeHeight);

    // Apply the corrective offsets to ensure the GridObject is fully within the grid
    if (X == 0) { // Left edge
        SpawnLocation.X += CorrectiveOffsets.X;
    }
    else if (X == TerraShiftParams->GridSize - 1) { // Right edge
        SpawnLocation.X -= CorrectiveOffsets.X;
    }

    if (Y == 0) { // Bottom edge
        SpawnLocation.Y += CorrectiveOffsets.Y;
    }
    else if (Y == TerraShiftParams->GridSize - 1) { // Top edge
        SpawnLocation.Y -= CorrectiveOffsets.Y;
    }

    return SpawnLocation;
}

FVector ATerraShiftEnvironment::GenerateRandomCornerGridLocation() const {
    if (!Grid) {
        return FVector::ZeroVector; // Safety check
    }

    // Randomly select one of the four corners
    int32 Corner = FMath::RandRange(0, 3);
    int32 X = 0;
    int32 Y = 0;
    switch (Corner) {
    case 0: // Top-Left corner
        X = 0;
        Y = 0;
        break;
    case 1: // Top-Right corner
        X = TerraShiftParams->GridSize - 1;
        Y = 0;
        break;
    case 2: // Bottom-Left corner
        X = 0;
        Y = TerraShiftParams->GridSize - 1;
        break;
    case 3: // Bottom-Right corner
        X = TerraShiftParams->GridSize - 1;
        Y = TerraShiftParams->GridSize - 1;
        break;
    }

    // Get the column's offsets (X, Y, Z) relative to the grid
    FVector ColumnOffsets = Grid->GetColumnOffsets(X, Y);

    // Get the grid's relative Z-location from its root component
    float GridRelativeZ = Grid->GetRootComponent()->GetRelativeLocation().Z;

    // Add the grid's relative Z-location to the column's Z offset
    float RelativeHeight = GridRelativeZ + ColumnOffsets.Z;

    // Calculate the final goal location
    FVector GoalLocation = Grid->CalculateColumnLocation(X, Y, RelativeHeight);

    return GoalLocation;
}

FVector ATerraShiftEnvironment::GridPositionToWorldPosition(FVector2D GridPosition) {
    float GridHalfSize = (TerraShiftParams->GridSize * CellSize) / 2.0f;

    return FVector(
        PlatformCenter.X + (GridPosition.X * CellSize) - GridHalfSize + (CellSize / 2.0f),
        PlatformCenter.Y + (GridPosition.Y * CellSize) - GridHalfSize + (CellSize / 2.0f),
        PlatformCenter.Z
    );
}

float ATerraShiftEnvironment::Map(float x, float in_min, float in_max, float out_min, float out_max) {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

bool ATerraShiftEnvironment::ObjectOffPlatform(int AgentIndex) {
    if (!GridObjectManager) return true;

    AGridObject* GridObject = GridObjectManager->GetGridObject(AgentIndex);
    if (!GridObject) return true;

    // Get the GridObject's world location
    FVector ObjectWorldPosition = GridObject->MeshComponent->GetComponentLocation();

    // Transform the GridObject's world position into the Platform's local space
    FVector ObjectRelativePosition = Platform->GetActorTransform().InverseTransformPosition(ObjectWorldPosition);

    // Get the GridObject's extent (half-size)
    FVector GridObjectExtent = GridObject->MeshComponent->Bounds.BoxExtent * GridObject->MeshComponent->GetComponentScale();

    // Get the platform's extent (half-size)
    FVector PlatformExtent = Platform->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().BoxExtent * Platform->GetActorScale3D();

    // Calculate the allowable range for the GridObject's center to be over the platform
    float AllowedMinX = -PlatformExtent.X + GridObjectExtent.X;
    float AllowedMaxX = PlatformExtent.X - GridObjectExtent.X;
    float AllowedMinY = -PlatformExtent.Y + GridObjectExtent.Y;
    float AllowedMaxY = PlatformExtent.Y - GridObjectExtent.Y;

    // Check if the GridObject's center is within the allowable range along X and Y
    bool IsWithinX = ObjectRelativePosition.X >= AllowedMinX && ObjectRelativePosition.X <= AllowedMaxX;
    bool IsWithinY = ObjectRelativePosition.Y >= AllowedMinY && ObjectRelativePosition.Y <= AllowedMaxY;

    // If the object is outside the allowable range on either axis, it's more than 50% off
    return !(IsWithinX && IsWithinY);
}

int ATerraShiftEnvironment::Get1DIndexFromPoint(const FIntPoint& Point, int GridSize) const {
    return Point.X * GridSize + Point.Y;
}

void ATerraShiftEnvironment::CheckAndRespawnGridObjects() {
    float GoalThreshold = 0.1f; // Threshold distance to consider a goal reached

    for (int32 AgentIndex = 0; AgentIndex < CurrentAgents; ++AgentIndex) {
        AGridObject* GridObject = GridObjectManager->GetGridObject(AgentIndex);
        if (!GridObject || !GridObject->IsActive()) {
            continue;
        }

        // Get the GridObject's location
        FVector ObjectLocation = GridObject->GetActorLocation();

        // Check if the GridObject has fallen off the platform
        if (ObjectOffPlatform(AgentIndex)) {
            // Respawn the GridObject
            RespawnGridObject(AgentIndex);
            // Update RewardBuffer
            RewardBuffer += -1.0f;
            continue;
        }

        // Check if the GridObject has reached any goal
        bool bReachedGoal = false;
        for (int32 GoalIndex = 0; GoalIndex < GoalPositionArray.Num(); ++GoalIndex) {
            FVector GoalLocation = GoalPositionArray[GoalIndex];
            float DistanceToGoal = FVector::Dist2D(ObjectLocation, GoalLocation);

            if (DistanceToGoal <= GoalThreshold) {
                bReachedGoal = true;
                if (GoalIndex == AgentGoalIndices[AgentIndex]) {
                    // Reached assigned goal
                    // Remove the GridObject
                    GridObject->SetGridObjectActive(false);
                    GridObjectManager->DeleteGridObject(AgentIndex);
                    AgentHasActiveGridObject[AgentIndex] = false;
                    // Update RewardBuffer
                    RewardBuffer += 1.0f;
                }
                else {
                    // Reached wrong goal
                    // Respawn the GridObject
                    RespawnGridObject(AgentIndex);
                    // Update RewardBuffer
                    RewardBuffer += -1.0f;
                }
                break; // Exit the goal checking loop
            }
        }

        // No further action needed if a goal was reached
        if (bReachedGoal) {
            continue;
        }
    }
}

void ATerraShiftEnvironment::RespawnGridObject(int32 AgentIndex) {
    if (!GridObjectManager) return;

    // Set the agent's GridObject as inactive
    if (AgentHasActiveGridObject.IsValidIndex(AgentIndex)) {
        AgentHasActiveGridObject[AgentIndex] = false;
    }

    // Delete the existing GridObject
    GridObjectManager->DeleteGridObject(AgentIndex);

    // Generate a new random spawn location
    FVector NewSpawnLocation = GenerateRandomGridLocation();

    // Assign a new goal index
    AgentGoalIndices[AgentIndex] = FMath::RandRange(0, TerraShiftParams->NumGoals - 1);

    // Spawn a new GridObject at the new location after a delay
    GridObjectManager->RespawnGridObjectAtLocation(AgentIndex, NewSpawnLocation, TerraShiftParams->RespawnDelay);
}

void ATerraShiftEnvironment::OnGridObjectSpawned(int32 Index, AGridObject* NewGridObject) {
    if (AgentHasActiveGridObject.IsValidIndex(Index)) {
        AgentHasActiveGridObject[Index] = true;
    }
}
