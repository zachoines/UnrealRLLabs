#include "TerraShiftEnvironment.h"
#include "Engine/World.h"
#include "TimerManager.h"

// Constructor
ATerraShiftEnvironment::ATerraShiftEnvironment() {
    EnvInfo.EnvID = 3;
    EnvInfo.IsMultiAgent = true;
    EnvInfo.ActionSpace = CreateDefaultSubobject<UActionSpace>(TEXT("ActionSpace"));

    TerraShiftRoot = CreateDefaultSubobject<USceneComponent>(TEXT("TerraShiftRoot"));
    RootComponent = TerraShiftRoot;

    
    GridObjectManager = CreateDefaultSubobject<AGridObjectManager>(TEXT("GridObjectManager"));;
    WaveSimulator = nullptr;
    Grid = nullptr;
}

// Destructor
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
    }
}

FState ATerraShiftEnvironment::ResetEnv(int NumAgents) {
    CurrentStep = 0;
    CurrentAgents = NumAgents;

    // Reset AgentParametersArray to match CurrentAgents
    AgentParametersArray.SetNum(CurrentAgents);
    AgentGoalIndices.SetNum(CurrentAgents);

    // Reset goals, grid, and grid objects
    GoalPositionArray.Empty();
    GoalPositionArray.SetNum(TerraShiftParams->NumGoals);

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
    if (GridObjectManager) {
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

    // Spawn GridObjects at specified locations
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

    // Update active columns based on GridObjects' proximity
    UpdateActiveColumns();
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
    // This function checks if the environment should be marked as done
    // Currently set to always return false (no terminal state)
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
    // Calculate and return the reward for the current step
    // This function can be modified to fit the specific reward calculation logic
    return 0.0f;
}

void ATerraShiftEnvironment::SetupActionAndObservationSpace() {
    const int NumAgentWaveParameters = 11;
    const int NumAgentStateParameters = NumAgentWaveParameters + 3 + 2; // Includes position, velocity, wave parameters, object position, and goal position
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

FIntPoint ATerraShiftEnvironment::GenerateRandomGoalIndex() const {
    // Generate random goal indices along the edges of the grid
    int Side = FMath::RandRange(0, 3);
    FIntPoint GoalIndex2D;
    switch (Side) {
    case 0: GoalIndex2D = FIntPoint(FMath::RandRange(0, TerraShiftParams->GridSize - 1), 0); break;          // Top
    case 1: GoalIndex2D = FIntPoint(FMath::RandRange(0, TerraShiftParams->GridSize - 1), TerraShiftParams->GridSize - 1); break; // Bottom
    case 2: GoalIndex2D = FIntPoint(0, FMath::RandRange(0, TerraShiftParams->GridSize - 1)); break;          // Left
    case 3: GoalIndex2D = FIntPoint(TerraShiftParams->GridSize - 1, FMath::RandRange(0, TerraShiftParams->GridSize - 1)); break; // Right
    }
    return GoalIndex2D;
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
    // Map a value from one range to another
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

bool ATerraShiftEnvironment::ObjectOffPlatform(int AgentIndex) {
    if (!GridObjectManager) return true;

    FVector ObjectPosition = GridObjectManager->GetGridObjectWorldLocation(AgentIndex);
    FVector PlatformExtent = Platform->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().BoxExtent * Platform->GetActorScale3D();

    // Check if the object is within the bounds of the platform
    return ObjectPosition.X < PlatformCenter.X - PlatformExtent.X ||
        ObjectPosition.X > PlatformCenter.X + PlatformExtent.X ||
        ObjectPosition.Y < PlatformCenter.Y - PlatformExtent.Y ||
        ObjectPosition.Y > PlatformCenter.Y + PlatformExtent.Y;
}

int ATerraShiftEnvironment::Get1DIndexFromPoint(const FIntPoint& Point, int GridSize) const {
    // Convert a 2D grid index to a 1D index
    return Point.X * GridSize + Point.Y;
}
