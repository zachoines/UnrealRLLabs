#include "TerraShiftEnvironment.h"

ATerraShiftEnvironment::ATerraShiftEnvironment()
{
    PrimaryActorTick.bCanEverTick = true;
    PrimaryActorTick.TickGroup = TG_PostUpdateWork;
    MaxAgents = 10;
    EnvInfo.EnvID = 3;
    EnvInfo.IsMultiAgent = true;
    EnvInfo.MaxAgents = MaxAgents;
    EnvInfo.ActionSpace = CreateDefaultSubobject<UActionSpace>(TEXT("ActionSpace"));

    TerraShiftRoot = CreateDefaultSubobject<USceneComponent>(TEXT("TerraShiftRoot"));
    RootComponent = TerraShiftRoot;

    WaveSimulator = CreateDefaultSubobject<UMorletWavelets2D>(TEXT("WaveSimulator"));
    Grid = nullptr;
    Initialized = false;

    // Initialize RewardBuffer
    RewardBuffer = 0.0f;

    GoalColors = {
        FLinearColor(1.0f, 0.0f, 0.0f), // Red
        FLinearColor(0.0f, 1.0f, 0.0f), // Green
        FLinearColor(0.0f, 0.0f, 1.0f), // Blue
        FLinearColor(1.0f, 1.0f, 0.0f)  // Yellow
    };

    // Set up observation and action space
    SetupActionAndObservationSpace();
}

ATerraShiftEnvironment::~ATerraShiftEnvironment()
{
 
}

void ATerraShiftEnvironment::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    // This tick is done post update of RLRunner. And this updates environment in time for call to "Transition."
    if (Initialized)
    {
        // Update active columns after physics simulation
        UpdateActiveColumns();

        // Update column and GridObject colors
        UpdateColumnGoalObjectColors();

        // Check and respawn GridObjects if necessary
        CheckAndRespawnGridObjects();

        // Select reward
        UpdateRewardBuffer(); 
    }
}

void ATerraShiftEnvironment::InitEnv(FBaseInitParams* BaseParams)
{
    check(BaseParams != nullptr);
    TerraShiftParams = static_cast<FTerraShiftEnvironmentInitParams*>(BaseParams);
    check(TerraShiftParams != nullptr);

    CurrentAgents = 1; // Will be updated in ResetEnv()
    CurrentStep = 0;

    // Set up the environment's folder path
    EnvironmentFolderPath = GetName();
    SetFolderPath(*EnvironmentFolderPath);

    // Set up observation and action space
    SetupActionAndObservationSpace();

    // Initialize agent-related arrays
    AgentGoalIndices.SetNum(CurrentAgents);
    AgentHasActiveGridObject.SetNum(CurrentAgents);
    GridObjectFallenOffGrid.SetNum(CurrentAgents);
    GridObjectHasReachedGoal.SetNum(CurrentAgents);

    // Set the environment root's world location
    TerraShiftRoot->SetWorldLocation(TerraShiftParams->Location);

    // Spawn the platform at the specified location
    Platform = SpawnPlatform(TerraShiftParams->Location);
    check(Platform != nullptr);

    // Scale the platform based on the specified PlatformSize
    Platform->SetActorScale3D(FVector(TerraShiftParams->PlatformSize));

    // Calculate platform dimensions and determine grid cell size
    PlatformWorldSize = Platform->PlatformMeshComponent->GetStaticMesh()->GetBounds().BoxExtent * 2.0f * Platform->GetActorScale3D();
    PlatformCenter = Platform->GetActorLocation();
    CellSize = PlatformWorldSize.X / static_cast<float>(TerraShiftParams->GridSize);

    // Initialize GridCenterPoints to match the grid size
    const int32 TotalGridPoints = TerraShiftParams->GridSize * TerraShiftParams->GridSize;
    GridCenterPoints.SetNum(TotalGridPoints);

    // Calculate grid center points
    for (int32 X = 0; X < TerraShiftParams->GridSize; ++X)
    {
        for (int32 Y = 0; Y < TerraShiftParams->GridSize; ++Y)
        {
            float CenterX = PlatformCenter.X - (PlatformWorldSize.X / 2.0f) + (X + 0.5f) * CellSize;
            float CenterY = PlatformCenter.Y - (PlatformWorldSize.Y / 2.0f) + (Y + 0.5f) * CellSize;
            float CenterZ = PlatformCenter.Z;

            int32 Index = X * TerraShiftParams->GridSize + Y;
            GridCenterPoints[Index] = FVector(CenterX, CenterY, CenterZ);
        }
    }

    // Initialize the grid at the center of the platform, slightly elevated
    FVector GridLocation = PlatformCenter + FVector(0.0f, 0.0f, 1.0f); // Slightly above the platform

    FActorSpawnParameters SpawnParams;
    SpawnParams.Owner = this;
    Grid = GetWorld()->SpawnActor<AGrid>(AGrid::StaticClass(), GridLocation, FRotator::ZeroRotator, SpawnParams);
    if (Grid)
    {
        Grid->AttachToActor(Platform, FAttachmentTransformRules::KeepRelativeTransform);
        Grid->InitializeGrid(
            TerraShiftParams->GridSize,
            PlatformWorldSize.X,
            GridLocation
        );
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to spawn Grid"));
    }

    // Initialize GridObjectManager and set its folder path
    GridObjectManager = GetWorld()->SpawnActor<AGridObjectManager>(AGridObjectManager::StaticClass());
    if (GridObjectManager)
    {
        GridObjectManager->SetObjectSize(TerraShiftParams->ObjectSize);

        // Set the PlatformActor in GridObjectManager
        GridObjectManager->SetPlatformActor(Platform);

        // Bind to the GridObjectManager's event
        GridObjectManager->OnGridObjectSpawned.AddDynamic(this, &ATerraShiftEnvironment::OnGridObjectSpawned);

        // Set the folder path for GridObjectManager
        GridObjectManager->SetFolderPath(FName(*(EnvironmentFolderPath + "/GridObjectManager")));
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to spawn GridObjectManager"));
    }
 
    check(WaveSimulator != nullptr);

    // Build wave parameter ranges from TerraShiftParams
    FWaveParameterRanges Ranges;
    Ranges.AmplitudeRange = TerraShiftParams->AmplitudeRange;
    Ranges.WaveOrientationRange = TerraShiftParams->WaveOrientationRange;
    Ranges.WavenumberRange = TerraShiftParams->WavenumberRange;
    Ranges.PhaseVelocityRange = TerraShiftParams->PhaseVelocityRange;
    Ranges.PhaseRange = TerraShiftParams->PhaseRange;
    Ranges.SigmaRange = TerraShiftParams->SigmaRange;
    Ranges.VelocityRange = TerraShiftParams->VelocityRange;

    // Now call the new Initialize function
    WaveSimulator->Initialize(
        TerraShiftParams->GridSize,
        TerraShiftParams->GridSize,
        Ranges,
        0.1f
    );

    LastDeltaTime = GetWorld()->GetDeltaSeconds();
    Initialized = true;
}

FState ATerraShiftEnvironment::ResetEnv(int NumAgents)
{
    // Reset tracked variables
    CurrentStep = 0;
    CurrentAgents = NumAgents;
    LastDeltaTime = GetWorld()->GetDeltaSeconds();
    RewardBuffer = 0.0f;

    // Resize agent-related arrays
    AgentGoalIndices.Init(-1, CurrentAgents);
    AgentHasActiveGridObject.Init(false, CurrentAgents);
    GridObjectFallenOffGrid.Init(false, CurrentAgents);
    GridObjectHasReachedGoal.Init(false, CurrentAgents);
    PreviousObjectVelocities.Init(FVector::ZeroVector, CurrentAgents);
    PreviousObjectAcceleration.Init(FVector::ZeroVector, CurrentAgents);
    PreviousDistances.Init(-1.0f, CurrentAgents);
    PreviousPositions.Init(FVector::One() * -1.0, CurrentAgents);

    // Reset columns and grid objects
    GridObjectManager->ResetGridObjects();
    Grid->ResetGrid();

    // Clear existing goal platforms
    for (AGoalPlatform* GoalPlatform : GoalPlatforms)
    {
        GoalPlatform->Destroy();
    }
    GoalPlatforms.Empty();

    // Randomly determine number of current goals between 1 and 4
    int32 NumGoals = FMath::RandRange(1, 4);
    TerraShiftParams->NumGoals = NumGoals;

    // Create goal platforms
    for (int32 i = 0; i < NumGoals; ++i)
    {
        UpdateGoal(i);
    }

    // Reinitialize agent parameters
    for (int32 i = 0; i < CurrentAgents; ++i)
    {
        // Assign goal indices for each agent
        AgentGoalIndices[i] = FMath::RandRange(0, NumGoals - 1);
    }

    // Reset the wave simulator
    WaveSimulator->Reset(CurrentAgents);

    // Set active grid objects for the new agents
    SetActiveGridObjects(CurrentAgents);

    return State();
}

TArray<float> ATerraShiftEnvironment::AgentGetState(int AgentIndex)
{
    TArray<float> State;

    // 1) Retrieve wave state from the simulator
    FAgentWaveState WaveState = WaveSimulator->GetAgentWaveState(AgentIndex);

    // 2) Check if the agent currently has an active GridObject
    bool bHasActiveObject = AgentHasActiveGridObject[AgentIndex];

    // 3) Retrieve object world position and velocity if active
    FVector ObjectWorldPosition = FVector::ZeroVector;
    FVector ObjectWorldVelocity = FVector::ZeroVector;
    FVector ObjectWorldAcceleration = FVector::ZeroVector;
    if (bHasActiveObject && GridObjectManager)
    {
        AGridObject* GridObject = GridObjectManager->GetGridObject(AgentIndex);
        ObjectWorldPosition = GridObject->GetObjectLocation();
        ObjectWorldVelocity = GridObject->MeshComponent->GetPhysicsLinearVelocity();
    }

    // 4) Get object position/velocity/accel in relative coords
    FVector ObjectRelativePosition = Platform->GetActorTransform().InverseTransformPosition(ObjectWorldPosition);
    FVector ObjectRelativeVelocity = Platform->GetActorTransform().InverseTransformVector(ObjectWorldVelocity);
    FVector ObjectRelativeAcceleration = ObjectRelativeVelocity - PreviousObjectVelocities[AgentIndex];

    // 5) Retrieve agent�s grid position from wave simulator
    FVector2f AgentGridPosition = WaveSimulator->GetAgentPosition(AgentIndex);
    FVector   AgentWorldPosition = GridPositionToWorldPosition(FVector2D(AgentGridPosition));
    FVector   AgentRelativePosition = Platform->GetActorTransform().InverseTransformPosition(AgentWorldPosition);

    // 6) Retrieve goal position
    int32 AgentGoalIndex = AgentGoalIndices[AgentIndex];
    FVector GoalRelativePosition = GoalPlatforms[AgentGoalIndex]->GetRelativeLocation();

    // 7) Determine column height at the agent�s position
    float ColumnHeightAtAgent = 0.0f;
    int32 GridX = FMath::Clamp(FMath::FloorToInt(AgentGridPosition.X), 0, TerraShiftParams->GridSize - 1);
    int32 GridY = FMath::Clamp(FMath::FloorToInt(AgentGridPosition.Y), 0, TerraShiftParams->GridSize - 1);
    int32 ColumnIndex = GridX * TerraShiftParams->GridSize + GridY;
    if (ColumnIndex >= 0 && ColumnIndex < Grid->GetTotalColumns())
    {
        ColumnHeightAtAgent = Grid->GetColumnHeight(ColumnIndex);
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("AgentGetState: Computed column index out of range for agent %d"), AgentIndex);
    }

    // 8) Compute distance to the assigned goal (if object is active)
    float DistanceToGoal = -1.0f;
    if (bHasActiveObject)
    {
        AGoalPlatform* AssignedGoal = GoalPlatforms[AgentGoalIndex];
        if (AssignedGoal)
        {
            DistanceToGoal = FVector::Dist(ObjectRelativePosition, GoalRelativePosition);
        }
    }

    // 9) Now add wave parameters from the wave state
    State.Add(WaveState.Velocity.X);
    State.Add(WaveState.Velocity.Y);
    State.Add(WaveState.Amplitude);
    State.Add(WaveState.WaveOrientation);
    State.Add(WaveState.Wavenumber);
    State.Add(WaveState.PhaseVelocity);
    State.Add(WaveState.Phase);
    State.Add(WaveState.Sigma);
    State.Add(WaveState.Time);

    // 10) Add positions (agent, object, goal)
    State.Add(AgentRelativePosition.X);
    State.Add(AgentRelativePosition.Y);
    State.Add(ObjectRelativePosition.X);
    State.Add(ObjectRelativePosition.Y);
    State.Add(ObjectRelativePosition.Z);
    State.Add(GoalRelativePosition.X);
    State.Add(GoalRelativePosition.Y);
    State.Add(GoalRelativePosition.Z);

    // 11) Add object velocity (X, Y, Z)
    State.Add(ObjectRelativeVelocity.X);
    State.Add(ObjectRelativeVelocity.Y);
    State.Add(ObjectRelativeVelocity.Z);

    // 12) Add object�s relative acceleration (X, Y, Z)
    State.Add(ObjectRelativeAcceleration.X);
    State.Add(ObjectRelativeAcceleration.Y);
    State.Add(ObjectRelativeAcceleration.Z);

    // Now add previous measures
    // 13.) Add previous object velocity (X, Y, Z)
    State.Add(PreviousObjectVelocities[AgentIndex].X);
    State.Add(PreviousObjectVelocities[AgentIndex].Y);
    State.Add(PreviousObjectVelocities[AgentIndex].Z);

    // 14.) Add previous object acceleration (X, Y, Z)
    State.Add(PreviousObjectAcceleration[AgentIndex].X);
    State.Add(PreviousObjectAcceleration[AgentIndex].Y);
    State.Add(PreviousObjectAcceleration[AgentIndex].Z);

    // 15.)  Add previous object distance
    State.Add(PreviousDistances[AgentIndex]);

    // 16.) Add previous object position
    State.Add(PreviousPositions[AgentIndex].X);
    State.Add(PreviousPositions[AgentIndex].Y);
    State.Add(PreviousPositions[AgentIndex].Z);

    // Add remaining misc properties

    // 17) Add column height at the agent�s position
    State.Add(ColumnHeightAtAgent);

    // 18) Add a flag indicating if the GridObject is active
    State.Add(bHasActiveObject ? 1.0f : 0.0f);

    // 19) Add the goal index offset (AgentGoalIndex + 1) and distance to goal
    State.Add(bHasActiveObject ? static_cast<float>(AgentGoalIndex + 1) : 0.0f);
    State.Add(DistanceToGoal);

    return State;
}

void ATerraShiftEnvironment::SetActiveGridObjects(int NumAgents)
{
    TArray<FVector> SpawnLocations;

    // Generate random spawn locations above the platform
    for (int32 i = 0; i < NumAgents; ++i)
    {
        FVector RandomLocation = GenerateRandomGridLocation();
        SpawnLocations.Add(RandomLocation);
    }

    // Spawn GridObjects at specified locations with a spawn delay
    GridObjectManager->SpawnGridObjects(SpawnLocations, TerraShiftParams->ObjectSize, TerraShiftParams->SpawnDelay);
}

void ATerraShiftEnvironment::Act(FAction Action)
{
    const int NumAgentActions = EnvInfo.ActionSpace->ContinuousActions.Num();
    if (Action.Values.Num() != CurrentAgents * NumAgentActions)
    {
        UE_LOG(LogTemp, Error, TEXT("Action array size mismatch. Expected %d, got %d"),
            CurrentAgents * NumAgentActions, Action.Values.Num());
        return;
    }

    float DeltaTime = GetWorld()->GetDeltaSeconds();

    // 1) Create an array of FAgentDeltaParameters to pass to the wave simulator
    TArray<FAgentDeltaParameters> DeltaParamsArray;
    DeltaParamsArray.SetNum(CurrentAgents);

    // 2) Fill in the deltas in [-1..1] for each agent
    for (int32 i = 0; i < CurrentAgents; ++i)
    {
        int32 BaseIndex = i * NumAgentActions;

        // Use tanh() and clamp to ensure each delta is in [-1..1]
        DeltaParamsArray[i].Velocity.X = FMath::Clamp(tanh(Action.Values[BaseIndex + 0]), -1.0f, 1.0f);
        DeltaParamsArray[i].Velocity.Y = FMath::Clamp(tanh(Action.Values[BaseIndex + 1]), -1.0f, 1.0f);
        DeltaParamsArray[i].Amplitude = FMath::Clamp(tanh(Action.Values[BaseIndex + 2]), -1.0f, 1.0f);
        DeltaParamsArray[i].WaveOrientation = FMath::Clamp(tanh(Action.Values[BaseIndex + 3]), -1.0f, 1.0f);
        DeltaParamsArray[i].Wavenumber = FMath::Clamp(tanh(Action.Values[BaseIndex + 4]), -1.0f, 1.0f);
        DeltaParamsArray[i].PhaseVelocity = FMath::Clamp(tanh(Action.Values[BaseIndex + 5]), -1.0f, 1.0f);
        DeltaParamsArray[i].Phase = FMath::Clamp(tanh(Action.Values[BaseIndex + 6]), -1.0f, 1.0f);
        DeltaParamsArray[i].Sigma = FMath::Clamp(tanh(Action.Values[BaseIndex + 7]), -1.0f, 1.0f);

        // 3) Accumulate time as usual
        DeltaParamsArray[i].Time = DeltaTime;
    }

    // 4) Let the wave simulator interpret these deltas and generate the new height map
    const FMatrix2D& HeightMap = WaveSimulator->Update(DeltaParamsArray);

    // 5) Update the actual columns
    Grid->UpdateColumnHeights(HeightMap);
}

void ATerraShiftEnvironment::UpdateActiveColumns()
{
    if (!Grid || !GridObjectManager)
    {
        UE_LOG(LogTemp, Warning, TEXT("Grid or GridObjectManager is null in UpdateActiveColumns"));
        return;
    }

    // Determine currently active columns based on proximity to GridObjects
    TSet<int32> NewActiveColumns = GridObjectManager->GetActiveColumnsInProximity(
        TerraShiftParams->GridSize, GridCenterPoints, PlatformCenter, PlatformWorldSize.X, CellSize
    );

    // Calculate columns that need to be toggled on or off
    TSet<int32> ColumnsToEnable = NewActiveColumns.Difference(ActiveColumns);
    TSet<int32> ColumnsToDisable = ActiveColumns.Difference(NewActiveColumns);

    // Enable or disable physics for relevant columns
    if ((ColumnsToEnable.Num() > 0 || ColumnsToDisable.Num() > 0))
    {
        // Create arrays to match the indices for toggling physics
        TArray<int32> ColumnsToToggle;
        TArray<bool> EnablePhysics;

        // Add columns to enable physics
        for (int32 ColumnIndex : ColumnsToEnable)
        {
            ColumnsToToggle.Add(ColumnIndex);
            EnablePhysics.Add(true);
        }

        // Add columns to disable physics
        for (int32 ColumnIndex : ColumnsToDisable)
        {
            ColumnsToToggle.Add(ColumnIndex);
            EnablePhysics.Add(false);
        }

        // Toggle physics for the specified columns
        Grid->TogglePhysicsForColumns(ColumnsToToggle, EnablePhysics);
    }

    // Update the active columns set after toggling
    ActiveColumns = NewActiveColumns;
}

void ATerraShiftEnvironment::UpdateColumnGoalObjectColors()
{
    if (!Grid || !GridObjectManager)
    {
        return;
    }

    // Map goal indices to colors
    TMap<int32, FLinearColor> GoalIndexToColor;
    int32 NumGoals = GoalPlatforms.Num();
    for (int32 i = 0; i < NumGoals; ++i)
    {
        FLinearColor Color = GoalColors[i % GoalColors.Num()];
        GoalIndexToColor.Add(i, Color);
    }

    // Set GridObjects to match their goal colors
    for (int32 AgentIndex = 0; AgentIndex < CurrentAgents; ++AgentIndex)
    {
        if (AgentHasActiveGridObject[AgentIndex])
        {
            AGridObject* GridObject = GridObjectManager->GetGridObject(AgentIndex);
            int32 GoalIndex = AgentGoalIndices[AgentIndex];
            if (GoalIndexToColor.Contains(GoalIndex))
            {
                FLinearColor GoalColor = GoalIndexToColor[GoalIndex];
                GridObject->SetGridObjectColor(GoalColor);
            }
        }
    }

    // Set active columns (physics enabled) to black
    for (int32 ColumnIndex : ActiveColumns)
    {
        Grid->SetColumnColor(ColumnIndex, FLinearColor::Black);
    }

    // Set other columns' colors based on their height
    float MinHeight = Grid->GetMinHeight();
    float MaxHeight = Grid->GetMaxHeight();

    for (int32 ColumnIndex = 0; ColumnIndex < Grid->GetTotalColumns(); ++ColumnIndex)
    {
        if (ActiveColumns.Contains(ColumnIndex))
        {
            // Already colored active columns black
            continue;
        }
        float Height = Grid->GetColumnHeight(ColumnIndex);
        float HeightRatio = FMath::GetMappedRangeValueClamped(FVector2D(MinHeight, MaxHeight), FVector2D(0.0f, 1.0f), Height);
        FLinearColor Color = FLinearColor::LerpUsingHSV(FLinearColor::Black, FLinearColor::White, HeightRatio);
        Grid->SetColumnColor(ColumnIndex, Color);
    }
}

FState ATerraShiftEnvironment::State()
{
    FState CurrentState;
    for (int32 i = 0; i < CurrentAgents; ++i)
    {
        CurrentState.Values.Append(AgentGetState(i));
    }
    return CurrentState;
}

void ATerraShiftEnvironment::PostTransition()
{
    UpdateObjectStats();
}

void ATerraShiftEnvironment::PostStep()
{
    CurrentStep += 1;
}

void ATerraShiftEnvironment::PreTransition() 
{

}

void ATerraShiftEnvironment::PreStep()
{
    
}

bool ATerraShiftEnvironment::Done()
{
    return false;
}

bool ATerraShiftEnvironment::Trunc()
{
    if (CurrentStep >= TerraShiftParams->MaxSteps)
    {
         CurrentStep = 0;
        return true;
    }
    return false;
}

float ATerraShiftEnvironment::Reward()
{
    float rewardTmp = RewardBuffer; 
    RewardBuffer = 0.0f; // Because we are collecting rewards between "n" action repeats, after which "Reward" is finally called.
    return rewardTmp;
}

void ATerraShiftEnvironment::SetupActionAndObservationSpace()
{
    const int NumAgentObs = 37;
    EnvInfo.SingleAgentObsSize = NumAgentObs;
    EnvInfo.StateSize = MaxAgents * EnvInfo.SingleAgentObsSize;

    // 8 continuous actions:
    //   VelocityX, VelocityY, Amplitude, WaveOrientation, Wavenumber,
    //   PhaseVelocity, Phase, Sigma
    const int NumAgentActions = 8;
    TArray<FContinuousActionSpec> ContinuousActions;
    ContinuousActions.Reserve(NumAgentActions);

    for (int32 i = 0; i < NumAgentActions; ++i)
    {
        FContinuousActionSpec ActionSpec;
        ActionSpec.Low = -1.0f;
        ActionSpec.High = 1.0f;
        ContinuousActions.Add(ActionSpec);
    }

    if (EnvInfo.ActionSpace)
    {
        EnvInfo.ActionSpace->Init(ContinuousActions, {});
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("EnvInfo.ActionSpace is null in SetupActionAndObservationSpace"));
    }
}

AMainPlatform* ATerraShiftEnvironment::SpawnPlatform(FVector Location)
{
    if (UWorld* World = GetWorld())
    {
        UStaticMesh* PlaneMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Plane.Plane"));
        if (PlaneMesh)
        {
            FActorSpawnParameters SpawnParams;
            AMainPlatform* NewPlatform = World->SpawnActor<AMainPlatform>(AMainPlatform::StaticClass(), Location, FRotator::ZeroRotator, SpawnParams);
            if (NewPlatform)
            {
                // Initialize the platform with mesh and material
                UMaterial* Material = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/Material/Platform_Material.Platform_Material'"));
                NewPlatform->InitializePlatform(PlaneMesh, Material);

                // Attach the platform to the environment root
                NewPlatform->AttachToActor(this, FAttachmentTransformRules::KeepRelativeTransform);
            }
            else
            {
                UE_LOG(LogTemp, Error, TEXT("Failed to spawn MainPlatform"));
            }
            return NewPlatform;
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to load PlaneMesh"));
        }
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("World is null in SpawnPlatform"));
    }
    return nullptr;
}

FVector ATerraShiftEnvironment::GenerateRandomGridLocation() const
{
    // Generate random X, Y grid coordinates
    int32 X = FMath::RandRange(0, TerraShiftParams->GridSize - 1);
    int32 Y = FMath::RandRange(0, TerraShiftParams->GridSize - 1);

    // Get the index into GridCenterPoints
    int32 Index = X * TerraShiftParams->GridSize + Y;

    // Check if index is valid
    if (!GridCenterPoints.IsValidIndex(Index))
    {
        UE_LOG(LogTemp, Error, TEXT("Invalid grid index in GenerateRandomGridLocation: %d"), Index);
        return FVector::ZeroVector;
    }

    // Get the center point of the grid cell (world coordinates)
    FVector WorldSpawnLocation = GridCenterPoints[Index];

    // Adjust the Z coordinate to be slightly above the grid
    float SpawnHeight = Grid->GetActorLocation().Z + TerraShiftParams->ObjectSize.Z * 2.0f;
    WorldSpawnLocation.Z = SpawnHeight;

    // Return world location directly
    return WorldSpawnLocation;
}

FVector ATerraShiftEnvironment::GridPositionToWorldPosition(FVector2D GridPosition) const
{
    float GridHalfSize = (TerraShiftParams->GridSize * CellSize) / 2.0f;

    return FVector(
        PlatformCenter.X - GridHalfSize + (GridPosition.X * CellSize) + (CellSize / 2.0f),
        PlatformCenter.Y - GridHalfSize + (GridPosition.Y * CellSize) + (CellSize / 2.0f),
        PlatformCenter.Z
    );
}

float ATerraShiftEnvironment::Map(float x, float in_min, float in_max, float out_min, float out_max)
{
    float denominator = in_max - in_min;
    if (FMath::IsNearlyZero(denominator))
    {
        UE_LOG(LogTemp, Warning, TEXT("Division by zero in Map function"));
        return out_min;
    }
    return (x - in_min) * (out_max - out_min) / denominator + out_min;
}

void ATerraShiftEnvironment::CheckAndRespawnGridObjects()
{
    for (int32 AgentIndex = 0; AgentIndex < CurrentAgents; ++AgentIndex)
    {
        if (AgentHasActiveGridObject[AgentIndex]) 
        {
            AGridObject* GridObject = GridObjectManager->GetGridObject(AgentIndex);
            FVector ObjectWorldPosition = GridObject->GetObjectLocation();
            FVector ObjectExtent = GridObject->GetObjectExtent();
            float PlatformZ = Platform->GetActorLocation().Z;
            bool ShouldRespawnGridObject = false;
            
            // Check if the GridObject has reached its goal platform
            if (!GridObjectHasReachedGoal[AgentIndex])
            {
                int32 GoalIndex = AgentGoalIndices[AgentIndex];
                FVector GoalWorldPosition = GoalPlatforms[GoalIndex]->GetActorLocation();
                float DistanceToGoal = FVector::Dist2D(ObjectWorldPosition, GoalWorldPosition);

                if (DistanceToGoal <= FMath::Max(ObjectExtent.X, ObjectExtent.Y))
                {
                    AgentHasActiveGridObject[AgentIndex] = false;
                    GridObjectHasReachedGoal[AgentIndex] = true;
                    ShouldRespawnGridObject = true;
                }
            }

            // Check if the bottom of GridObject has fallen below the platform
            if (!GridObjectFallenOffGrid[AgentIndex]) {
                if ((ObjectWorldPosition.Z - (ObjectExtent.Z / 2)) < PlatformZ)
                {
                    AgentHasActiveGridObject[AgentIndex] = false;
                    GridObjectFallenOffGrid[AgentIndex] = true;
                    ShouldRespawnGridObject = true;
                }
            }

            // Respawn if fallen off the grid or reached goal
            if (ShouldRespawnGridObject)
            {
                RespawnGridObject(AgentIndex);
            }
        }
    }
}

void ATerraShiftEnvironment::UpdateObjectStats() {
    if (GridObjectManager)
    {
        for (int32 i = 0; i < CurrentAgents; ++i)
        {
            if (AgentHasActiveGridObject[i])
            {

                // Get world relative attributes
                AGoalPlatform* AssignedGoal = GoalPlatforms[AgentGoalIndices[i]];
                AGridObject* GridObject = GridObjectManager->GetGridObject(i);
                FVector ObjectWorldVelocity = GridObject->MeshComponent->GetPhysicsLinearVelocity();
                FVector ObjectWorldPosition = GridObject->GetObjectLocation();
                FVector GoalWorldPosition = AssignedGoal->GetActorLocation();
               
                // Convert to relative values
                FVector GoalRelativePosition = Platform->GetActorTransform().InverseTransformPosition(GoalWorldPosition);
                FVector ObjectRelativePosition = Platform->GetActorTransform().InverseTransformPosition(ObjectWorldPosition);
                FVector ObjectRelativeVelocity = Platform->GetActorTransform().InverseTransformVector(ObjectWorldVelocity);
                FVector ObjectRelativeVelocityPrevious = PreviousObjectVelocities[i];
                FVector ObjectRelativeAcceleration = !FMath::IsNearlyZero(LastDeltaTime) ? (ObjectRelativeVelocity - ObjectRelativeVelocityPrevious) / LastDeltaTime : FVector::Zero();
             
                // Store for later use
                PreviousObjectVelocities[i] = ObjectRelativeVelocity;
                PreviousObjectAcceleration[i] = ObjectRelativeAcceleration;
                PreviousDistances[i] = FVector::Dist(ObjectRelativePosition, GoalRelativePosition);
                PreviousPositions[i] = ObjectRelativePosition;
            }
            else
            {
                PreviousObjectVelocities[i] = FVector::ZeroVector;
                PreviousObjectAcceleration[i] = FVector::ZeroVector;
                PreviousDistances[i] = -1;
                PreviousPositions[i] = -1 * FVector::One();
            }
        }
    }

    LastDeltaTime = GetWorld()->GetDeltaSeconds();
}

void ATerraShiftEnvironment::RespawnGridObject(int32 AgentIndex)
{
    GridObjectManager->DeleteGridObject(AgentIndex);

    // Generate a new random spawn location
    FVector NewSpawnLocation = GenerateRandomGridLocation();

    // Assign a new goal index
    int32 NumGoals = GoalPlatforms.Num();
    AgentGoalIndices[AgentIndex] = FMath::RandRange(0, NumGoals - 1);

    // Respawn the GridObject after a delay
    GridObjectManager->RespawnGridObjectAtLocation(AgentIndex, NewSpawnLocation, TerraShiftParams->RespawnDelay);
}

void ATerraShiftEnvironment::OnGridObjectSpawned(int32 Index, AGridObject* NewGridObject)
{
    if (AgentHasActiveGridObject.IsValidIndex(Index))
    {
        AgentHasActiveGridObject[Index] = true;
    }
}

FVector ATerraShiftEnvironment::CalculateGoalPlatformLocation(int32 EdgeIndex)
{
    // EdgeIndex: 0 - Top, 1 - Bottom, 2 - Left, 3 - Right
    FVector Location = FVector::ZeroVector;
    float Offset = (PlatformWorldSize.X / 2.0f) + (PlatformWorldSize.X * TerraShiftParams->ObjectSize.X / 2);

    switch (EdgeIndex)
    {
    case 0: // Top
        Location = FVector(0.0f, Offset, 0.0f);
        break;
    case 1: // Bottom
        Location = FVector(0.0f, -Offset, 0.0f);
        break;
    case 2: // Left
        Location = FVector(-Offset, 0.0f, 0.0f);
        break;
    case 3: // Right
        Location = FVector(Offset, 0.0f, 0.0f);
        break;
    default:
        break;
    }

    return Location;
}

void ATerraShiftEnvironment::UpdateGoal(int32 GoalIndex)
{
    // Calculate goal platform size based on GridObject size
    FVector GoalPlatformScale = TerraShiftParams->ObjectSize;

    // Calculate goal platform location
    FVector GoalLocation = CalculateGoalPlatformLocation(GoalIndex);

    // Get the goal color
    FLinearColor GoalColor = GoalColors[GoalIndex % GoalColors.Num()];

    // Spawn the goal platform using the GoalPlatform class
    FActorSpawnParameters SpawnParams;
    SpawnParams.Owner = this;
    AGoalPlatform* NewGoalPlatform = GetWorld()->SpawnActor<AGoalPlatform>(AGoalPlatform::StaticClass(), Platform->GetActorLocation(), FRotator::ZeroRotator, SpawnParams);
    if (NewGoalPlatform)
    {
        NewGoalPlatform->InitializeGoalPlatform(GoalLocation, GoalPlatformScale, GoalColor, Platform);
        GoalPlatforms.Add(NewGoalPlatform);
    }
}

void ATerraShiftEnvironment::UpdateRewardBuffer()
{
    float StepReward = 0.0f;

    for (int32 AgentIndex = 0; AgentIndex < CurrentAgents; ++AgentIndex)
    {
        bool bActive = AgentHasActiveGridObject[AgentIndex];
        bool bReachedGoal = GridObjectHasReachedGoal[AgentIndex];
        bool bHasFallen = GridObjectFallenOffGrid[AgentIndex];

        // Punish falling off
        if (bHasFallen)
        {
            StepReward -= 2.0f;
        }
        // Reward reaching goal
        else if (bReachedGoal)
        {
            StepReward += 5.0f;
        }
        // If the agent is still active (and hasn't reached goal or fallen)
        else if (bActive)
        {
            // Retrieve the GridObject and velocity
            int32 GoalIndex = AgentGoalIndices[AgentIndex];
            AGridObject* GridObject = GridObjectManager->GetGridObject(AgentIndex);
            AGoalPlatform* AssignedGoal = GoalPlatforms[GoalIndex];
            FVector CurrentObjectRelativePosition = Platform->GetActorTransform().InverseTransformPosition(GridObject->GetObjectLocation());
            FVector CurrentObjectRelativeVelocity = Platform->GetActorTransform().InverseTransformVector(GridObject->MeshComponent->GetPhysicsLinearVelocity());
            FVector CurrentObjectGoalRelativePosition = Platform->GetActorTransform().InverseTransformVector(AssignedGoal->GetActorLocation());
            FVector CurrentObjectGoalRelativeDirection = CurrentObjectGoalRelativePosition - CurrentObjectRelativePosition;

            // Punish positive Z velocity (bouncing; being launched)
            float UpwardVel = CurrentObjectRelativeVelocity.Z;
            if (UpwardVel > 10.0f) // Avoid punishing jitters or slow vertical diplacements
            {
                StepReward -= 0.001f * (UpwardVel - 10.0f);
            }

            // Reward velocity toward the goal, and punish velocity away
            CurrentObjectGoalRelativeDirection.Normalize(KINDA_SMALL_NUMBER);
            float DotToGoal = FVector::DotProduct(
                CurrentObjectRelativeVelocity,
                CurrentObjectGoalRelativeDirection
            );

            StepReward += 0.001f * DotToGoal; // Dot product: positive if velocity is in direction of goal
      

            // Distance improvement reward
            float CurrentDistance = FVector::Distance(CurrentObjectGoalRelativePosition, CurrentObjectRelativePosition);
            float DeltaImprovement = PreviousDistances[AgentIndex] - CurrentDistance;
            StepReward += (0.001f * DeltaImprovement) - 0.01f; // Minus a small constant to avoid doing nothing.
        }

        // Clear or reset flags so they are valid next step
        GridObjectFallenOffGrid[AgentIndex] = false;
        GridObjectHasReachedGoal[AgentIndex] = false;
    }

    RewardBuffer += StepReward;
}