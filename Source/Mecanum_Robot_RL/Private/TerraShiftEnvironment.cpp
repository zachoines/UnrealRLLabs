#include "TerraShiftEnvironment.h"

ATerraShiftEnvironment::ATerraShiftEnvironment()
{
    PrimaryActorTick.bCanEverTick = true;
    PrimaryActorTick.TickGroup = TG_PostUpdateWork;
    MaxAgents = 10;
    TerraShiftRoot = CreateDefaultSubobject<USceneComponent>(TEXT("TerraShiftRoot"));
    RootComponent = TerraShiftRoot;

    WaveSimulator = CreateDefaultSubobject<UMultiAgentFractalWave3D>(TEXT("WaveSimulator"));
    Grid = nullptr;
    Initialized = false;

    GoalColors = {
        FLinearColor(1.0f, 0.0f, 0.0f), // Red
        FLinearColor(0.0f, 1.0f, 0.0f), // Green
        FLinearColor(0.0f, 0.0f, 1.0f), // Blue
        FLinearColor(1.0f, 1.0f, 0.0f)  // Yellow
    };
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
    }
}

void ATerraShiftEnvironment::InitEnv(FBaseInitParams* BaseParams)
{
    // 1) Check params
    check(BaseParams != nullptr);
    TerraShiftParams = static_cast<FTerraShiftEnvironmentInitParams*>(BaseParams);
    check(TerraShiftParams != nullptr);
    check(TerraShiftParams->EnvConfig != nullptr);

    // 2) Reset basic counters
    CurrentAgents = 1;
    CurrentStep = 0;

    // 3) Basic environment folder organization
    EnvironmentFolderPath = GetName();
    SetFolderPath(*EnvironmentFolderPath);

    // 4) Initialize arrays for tracking multi-agent states
    AgentGoalIndices.SetNum(CurrentAgents);
    AgentHasActiveGridObject.SetNum(CurrentAgents);
    GridObjectFallenOffGrid.SetNum(CurrentAgents);
    GridObjectHasReachedGoal.SetNum(CurrentAgents);
    GridObjectShouldCollectEventReward.SetNum(CurrentAgents);
    GridObjectShouldRespawn.SetNum(CurrentAgents);
    GridObjectRespawnTimer.SetNum(CurrentAgents);
    GridObjectRespawnDelays.SetNum(CurrentAgents);

    // 5) Read environment params from config
    UEnvironmentConfig* EnvConfig = TerraShiftParams->EnvConfig;

    if (EnvConfig->HasPath(TEXT("environment/params/PlatformSize")))
    {
        PlatformSize = EnvConfig->Get(TEXT("environment/params/PlatformSize"))->AsNumber();
    }

    if (EnvConfig->HasPath(TEXT("environment/params/MaxColumnHeight")))
    {
        MaxColumnHeight = EnvConfig->Get(TEXT("environment/params/MaxColumnHeight"))->AsNumber();
    }

    if (EnvConfig->HasPath(TEXT("environment/params/ObjectSize")))
    {
        TArray<float> ObjSizeArr = EnvConfig->Get(TEXT("environment/params/ObjectSize"))->AsArrayOfNumbers();
        if (ObjSizeArr.Num() == 3)
        {
            ObjectSize = FVector(ObjSizeArr[0], ObjSizeArr[1], ObjSizeArr[2]);
        }
    }

    if (EnvConfig->HasPath(TEXT("environment/params/ObjectMass")))
    {
        ObjectMass = EnvConfig->Get(TEXT("environment/params/ObjectMass"))->AsNumber();
    }

    if (EnvConfig->HasPath(TEXT("environment/params/GridSize")))
    {
        GridSize = EnvConfig->Get(TEXT("environment/params/GridSize"))->AsInt();
    }

    if (EnvConfig->HasPath(TEXT("environment/params/MaxSteps")))
    {
        MaxSteps = EnvConfig->Get(TEXT("environment/params/MaxSteps"))->AsInt();
    }

    if (EnvConfig->HasPath(TEXT("environment/params/NumGoals")))
    {
       NumGoals = EnvConfig->Get(TEXT("environment/params/NumGoals"))->AsInt();
    }

    if (EnvConfig->HasPath(TEXT("environment/params/SpawnDelay")))
    {
        SpawnDelay = EnvConfig->Get(TEXT("environment/params/SpawnDelay"))->AsNumber();
    }

    if (EnvConfig->HasPath(TEXT("environment/params/MaxAgents")))
    {
        MaxAgents = EnvConfig->Get(TEXT("environment/params/MaxAgents"))->AsInt();
    }

    if (EnvConfig->HasPath(TEXT("environment/params/GoalThreshold")))
    {
        GoalThreshold = EnvConfig->Get(TEXT("environment/params/GoalThreshold"))->AsNumber();
    }

    // 6) Place environment at given location
    TerraShiftRoot->SetWorldLocation(TerraShiftParams->Location);

    // 7) Spawn platform at location with scale
    Platform = SpawnPlatform(TerraShiftParams->Location);
    Platform->SetActorScale3D(FVector(PlatformSize));

    // 8) Calculate platform dimensions & cell size
    PlatformWorldSize = Platform->PlatformMeshComponent->GetStaticMesh()->GetBounds().BoxExtent
        * 2.0f
        * Platform->GetActorScale3D();
    PlatformCenter = Platform->GetActorLocation();
    CellSize = PlatformWorldSize.X / static_cast<float>(GridSize);

    // 9) Initialize GridCenterPoints
    const int32 TotalGridPoints = GridSize * GridSize;
    GridCenterPoints.SetNum(TotalGridPoints);

    for (int32 X = 0; X < GridSize; ++X)
    {
        for (int32 Y = 0; Y < GridSize; ++Y)
        {
            float CenterX = PlatformCenter.X - (PlatformWorldSize.X * 0.5f) + (X + 0.5f) * CellSize;
            float CenterY = PlatformCenter.Y - (PlatformWorldSize.Y * 0.5f) + (Y + 0.5f) * CellSize;
            float CenterZ = PlatformCenter.Z;

            int32 Index = X * GridSize + Y;
            GridCenterPoints[Index] = FVector(CenterX, CenterY, CenterZ);
        }
    }

    // 10) Spawn the Grid slighly above platform (leaves room for column movements)
    FVector GridLocation = PlatformCenter + FVector(0.f, 0.f, MaxColumnHeight); 

    FActorSpawnParameters SpawnParams;
    SpawnParams.Owner = this;
    Grid = GetWorld()->SpawnActor<AGrid>(AGrid::StaticClass(), GridLocation, FRotator::ZeroRotator, SpawnParams);
    if (Grid)
    {
        Grid->AttachToActor(Platform, FAttachmentTransformRules::KeepRelativeTransform);
        Grid->SetColumnMovementBounds(-MaxColumnHeight, MaxColumnHeight);
        Grid->InitializeGrid(GridSize, PlatformWorldSize.X, GridLocation);
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to spawn Grid"));
    }

    // 11) Spawn GridObjectManager
    GridObjectManager = GetWorld()->SpawnActor<AGridObjectManager>(AGridObjectManager::StaticClass());
    if (GridObjectManager)
    {
        GridObjectManager->SetPlatformActor(Platform);
        GridObjectManager->SetFolderPath(FName(*(EnvironmentFolderPath + "/GridObjectManager")));
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to spawn GridObjectManager"));
    }

    // 12) Initialize Wave Simulator
    UEnvironmentConfig* MWConfig = EnvConfig->Get("environment/params/MultiAgentFractalWave");
    WaveSimulator->InitializeFromConfig(MWConfig);

    // 13) Mark as initialized
    LastDeltaTime = GetWorld()->GetDeltaSeconds();
    Initialized = true;
}

FState ATerraShiftEnvironment::ResetEnv(int NumAgents)
{
    // Reset tracked variables
    CurrentStep = 0;
    CurrentAgents = NumAgents;
    LastDeltaTime = GetWorld()->GetDeltaSeconds();

    // Resize agent-related arrays
    AgentGoalIndices.Init(-1, CurrentAgents);
    AgentHasActiveGridObject.Init(false, CurrentAgents);
    GridObjectFallenOffGrid.Init(false, CurrentAgents);
    GridObjectHasReachedGoal.Init(false, CurrentAgents);
    GridObjectShouldCollectEventReward.Init(false, CurrentAgents);
    GridObjectShouldRespawn.Init(true, CurrentAgents);
    GridObjectRespawnTimer.Init(0.0, CurrentAgents);
    GridObjectRespawnDelays.Init(0.0, CurrentAgents);

    PreviousObjectVelocities.Init(FVector::ZeroVector, CurrentAgents);
    PreviousObjectAcceleration.Init(FVector::ZeroVector, CurrentAgents);
    PreviousDistances.Init(-1.0f, CurrentAgents);
    PreviousPositions.Init(FVector::One() * -1.0f, CurrentAgents);

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
    NumGoals = FMath::RandRange(1, 4);

    // Create goal platforms
    for (int32 i = 0; i < NumGoals; ++i)
    {
        UpdateGoal(i);
    }

    for (int32 i = 0; i < CurrentAgents; ++i)
    {
        // Set the spawn time for each
        GridObjectRespawnDelays[i] = ((float)i) * SpawnDelay;

        // Assign a random goal index for each agent
        AgentGoalIndices[i] = FMath::RandRange(0, NumGoals - 1);
    }

    // Pass CurrentAgents so it can resize agent states if needed.
    WaveSimulator->Reset(CurrentAgents);

    return State();
}

TArray<float> ATerraShiftEnvironment::AgentGetState(int AgentIndex)
{
    TArray<float> State;
    FVector ObjectWorldPosition = FVector::ZeroVector;
    FVector ObjectWorldVelocity = FVector::ZeroVector;
    FVector ObjectRelativePosition = FVector::ZeroVector;
    FVector ObjectRelativeVelocity = FVector::ZeroVector;
    FVector ObjectRelativeAcceleration = FVector::ZeroVector;

    // 2) Retrieve object's relative world position/velocity/acceleration if active
    bool bHasActiveObject = AgentHasActiveGridObject[AgentIndex];
    if (bHasActiveObject && GridObjectManager)
    {
        AGridObject* GridObject = GridObjectManager->GetGridObject(AgentIndex);
        ObjectWorldPosition = GridObject->GetObjectLocation();
        ObjectWorldVelocity = GridObject->MeshComponent->GetPhysicsLinearVelocity();

        // 3) Convert object position/velocity to relative coordinates (platform-based)
        ObjectRelativePosition = Platform->GetActorTransform().InverseTransformPosition(ObjectWorldPosition);
        ObjectRelativeVelocity = Platform->GetActorTransform().InverseTransformVector(ObjectWorldVelocity);
        if (PreviousObjectVelocities[AgentIndex] != FVector::ZeroVector) 
        {
            ObjectRelativeAcceleration = ObjectRelativeVelocity - PreviousObjectVelocities[AgentIndex];
        } 
    }

    // 3) Retrieve the goal platform's position in relative coords
    int32 AgentGoalIndex = AgentGoalIndices[AgentIndex];
    FVector GoalRelativePosition = Platform->GetActorTransform().InverseTransformPosition(
        GoalPlatforms[AgentGoalIndex]->GetActorLocation()
    );

    // 4) Distance to the assigned goal (if object is active)
    float DistanceToGoal = -1.0f;
    if (bHasActiveObject)
    {
        AGoalPlatform* AssignedGoal = GoalPlatforms[AgentGoalIndex];
        if (AssignedGoal)
        {
            DistanceToGoal = FVector::Dist(ObjectRelativePosition, GoalRelativePosition);
        }
    }

    // 5) Add object and goal positions (X,Y,Z)
    State.Add(ObjectRelativePosition.X);
    State.Add(ObjectRelativePosition.Y);
    State.Add(ObjectRelativePosition.Z);

    State.Add(GoalRelativePosition.X);
    State.Add(GoalRelativePosition.Y);
    State.Add(GoalRelativePosition.Z);

    // 6) Add object velocity (X,Y,Z)
    State.Add(ObjectRelativeVelocity.X);
    State.Add(ObjectRelativeVelocity.Y);
    State.Add(ObjectRelativeVelocity.Z);

    // 7) Add object acceleration (X,Y,Z)
    State.Add(ObjectRelativeAcceleration.X);
    State.Add(ObjectRelativeAcceleration.Y);
    State.Add(ObjectRelativeAcceleration.Z);

    // 8) Add previous object velocity (X, Y, Z)
    State.Add(PreviousObjectVelocities[AgentIndex].X);
    State.Add(PreviousObjectVelocities[AgentIndex].Y);
    State.Add(PreviousObjectVelocities[AgentIndex].Z);

    // 9.) Add previous object acceleration (X, Y, Z)
    State.Add(PreviousObjectAcceleration[AgentIndex].X);
    State.Add(PreviousObjectAcceleration[AgentIndex].Y);
    State.Add(PreviousObjectAcceleration[AgentIndex].Z);

    // 10)  Add previous object distance to the assigned goal
    State.Add(PreviousDistances[AgentIndex]);

    // 11) Add previous object position (X,Y,Z)
    State.Add(PreviousPositions[AgentIndex].X);
    State.Add(PreviousPositions[AgentIndex].Y);
    State.Add(PreviousPositions[AgentIndex].Z);

    // 12) Flag: is the GridObject active?
    State.Add(bHasActiveObject ? 1.0f : 0.0f);

    // 13) Add the (GoalIndex + 1) if active, else 0, plus distance to goal
    State.Add(bHasActiveObject ? static_cast<float>(AgentGoalIndex + 1) : 0.0f);
    State.Add(DistanceToGoal);
    State.Add(LastDeltaTime);

    // 14) Agent Wave State
    State.Append(WaveSimulator->GetAgentStateVariables(AgentIndex));
    // State.Append(WaveSimulator->GetAgentFractalImage(AgentIndex));
    return State;
}

void ATerraShiftEnvironment::Act(FAction Action)
{
    // We'll define 9 floats per agent in the range [-1..1]:
    //   [0..2] => dPos.X, dPos.Y, dPos.Z
    //   [3] => dPitch
    //   [4] => dYaw
    //   [5] => dBaseFreq
    //   [6] => dLacunarity
    //   [7] => dGain
    //   [8] => dBlendWeight
    const int32 ValuesPerAgent = 9;

    if (Action.Values.Num() != CurrentAgents * ValuesPerAgent)
    {
        UE_LOG(LogTemp, Error, TEXT("Act: mismatch (#values=%d, expected=%d)"),
            Action.Values.Num(), CurrentAgents * ValuesPerAgent);
        return;
    }

    TArray<FFractalAgentAction> FractalActions;
    FractalActions.SetNum(CurrentAgents);

    // parse
    for (int32 i = 0; i < CurrentAgents; i++)
    {
        const int32 BaseIndex = i * ValuesPerAgent;

        // Each input is in [-1..1].
        float dx = Action.Values[BaseIndex + 0];
        float dy = Action.Values[BaseIndex + 1];
        float dz = Action.Values[BaseIndex + 2];

        float dPitch = Action.Values[BaseIndex + 3];
        float dYaw = Action.Values[BaseIndex + 4];
        float dBaseFreq = Action.Values[BaseIndex + 5];
        float dLacunarity = Action.Values[BaseIndex + 6];
        float dGain = Action.Values[BaseIndex + 7];
        float dBlendWeight = Action.Values[BaseIndex + 8];

        FFractalAgentAction& FA = FractalActions[i];
        FA.dPos = FVector(dx, dy, dz);
        FA.dPitch = dPitch;
        FA.dYaw = dYaw;
        FA.dBaseFreq = dBaseFreq;
        FA.dLacunarity = dLacunarity;
        FA.dGain = dGain;
        FA.dBlendWeight = dBlendWeight;
    }

    // Step fractal wave environment
    if (!WaveSimulator)
    {
        UE_LOG(LogTemp, Warning, TEXT("Act: WaveSimulator is null."));
        return;
    }

    WaveSimulator->Step(FractalActions, 0.1f);

    // final wave => NxN in [-1..1]
    const FMatrix2D& WaveMap = WaveSimulator->GetWave();
    // apply to columns
    Grid->UpdateColumnHeights(WaveMap);
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
        GridSize, GridCenterPoints, PlatformCenter, PlatformWorldSize.X, CellSize
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
    NumGoals = GoalPlatforms.Num();
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
    /*for (int32 ColumnIndex : ActiveColumns)
    {
        Grid->SetColumnColor(ColumnIndex, FLinearColor::Black);
    }*/

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
    CurrentState.Values.Append(WaveSimulator->GetWave().Data);
    for (int32 i = 0; i < CurrentAgents; ++i)
    {
        CurrentState.Values.Append(AgentGetState(i));
    }
    return CurrentState;
}

void ATerraShiftEnvironment::PreTransition()
{
    // Check and respawn GridObjects if necessary
    UpdateGridObjectFlags();
}

void ATerraShiftEnvironment::PostTransition()
{
    // Handle objects marked for respawn
    RespawnGridObjects();
    
    // Store previous object information
    UpdateObjectStats();
}

void ATerraShiftEnvironment::PostStep()
{
    CurrentStep += 1;
}

void ATerraShiftEnvironment::PreStep()
{
    
}

bool ATerraShiftEnvironment::Done()
{
    // Grid object has fallen off grid
    if (GridObjectFallenOffGrid.Contains(true)) {
        return true;
    }

    // If there are no more GridObjects left to handle, then the environment is done
    /*if (CurrentStep > 0 && (!AgentHasActiveGridObject.Contains(true) && !GridObjectShouldRespawn.Contains(true))) {
        return true;
    }*/

    return false;
}

bool ATerraShiftEnvironment::Trunc()
{
    if (CurrentStep >= MaxSteps)
    {
        return true;
    }
    return false;
}

float ATerraShiftEnvironment::Reward()
{
    float StepReward = 0.0f;
    float TotalDistanceImprovements = 0.0;
    for (int32 AgentIndex = 0; AgentIndex < CurrentAgents; ++AgentIndex)
    {
        // (A) Punish falling off platform
        if (GridObjectShouldCollectEventReward[AgentIndex] && GridObjectFallenOffGrid[AgentIndex])
        {
            StepReward -= 10.0f;
            GridObjectShouldCollectEventReward[AgentIndex] = false;
        } 

        // (B) Reward reaching goal
        else if (GridObjectShouldCollectEventReward[AgentIndex] && GridObjectHasReachedGoal[AgentIndex])
        {
            StepReward += 10.0f;
            GridObjectShouldCollectEventReward[AgentIndex] = false;
        }
        
        // (C) If still in play
        else if (AgentHasActiveGridObject[AgentIndex])
        {   
            AGridObject* GridObject = GridObjectManager->GetGridObject(AgentIndex);
            AGoalPlatform* AssignedGoal = GoalPlatforms[AgentGoalIndices[AgentIndex]];
            FVector ObjLocalPos = Platform->GetActorTransform().InverseTransformPosition(
                GridObject->GetObjectLocation()
            );
            FVector GoalLocalPos = Platform->GetActorTransform().InverseTransformPosition(
                AssignedGoal->GetActorLocation()
            );

            // 1) DISTANCE IMPROVEMENT
            if (PreviousDistances[AgentIndex] > 0.0f)
            {
                float newDist = FVector::Distance(
                    ObjLocalPos,
                    GoalLocalPos
                );

                float improvementFraction = (PreviousDistances[AgentIndex] - newDist);
                StepReward += 10.0 * FMath::Abs(improvementFraction) > 1e-2 ? improvementFraction: -0.0001;
            }

            // 2) VELOCITY TOWARD GOAL
            //{
            //    FVector ObjectLocalVel = Platform->GetActorTransform().InverseTransformVector(
            //        GridObject->MeshComponent->GetPhysicsLinearVelocity()
            //    );

            //    FVector toGoal = (GoalLocalPos - ObjLocalPos);
            //    float distGoal = toGoal.Size();
            //    if (distGoal > SMALL_NUMBER)
            //    {
            //        // Dot product => + if going toward, - if away
            //        float dotToGoal = FVector::DotProduct(ObjectLocalVel, toGoal);
            //        if (FMath::Abs(dotToGoal) > MaxDot) {
            //            MaxDot = FMath::Abs(dotToGoal);
            //        }
            //        StepReward += FMath::Abs(dotToGoal) > 1e-2 ? dotToGoal : -0.0001;
            //    }
            //}
        }
    }

    return StepReward;
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
    int32 X = FMath::RandRange(GridSize / 4, GridSize - (GridSize / 4));
    int32 Y = FMath::RandRange(GridSize / 4, GridSize - (GridSize / 4));
    
    /*int32 X = GridSize / 2;
    int32 Y = GridSize / 2;*/

    // Get the index into GridCenterPoints
    int32 Index = X * GridSize + Y;

    // Check if index is valid
    if (!GridCenterPoints.IsValidIndex(Index))
    {
        UE_LOG(LogTemp, Error, TEXT("Invalid grid index in GenerateRandomGridLocation: %d"), Index);
        return FVector::ZeroVector;
    }

    // Get the center point of the grid cell (world coordinates)
    FVector WorldSpawnLocation = GridCenterPoints[Index];
    WorldSpawnLocation.Z = Grid->GetActorLocation().Z;

    // Return world location directly
    return WorldSpawnLocation;
}

FVector ATerraShiftEnvironment::GridPositionToWorldPosition(FVector2D GridPosition) const
{
    float GridHalfSize = (GridSize * CellSize) / 2.0f;

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

void ATerraShiftEnvironment::UpdateGridObjectFlags()
{
    for (int32 AgentIndex = 0; AgentIndex < CurrentAgents; ++AgentIndex)
    {
        if (AgentHasActiveGridObject[AgentIndex])
        {
            AGridObject* GridObject = GridObjectManager->GetGridObject(AgentIndex);
            FVector ObjectWorldPosition = GridObject->GetObjectLocation();
            FVector ObjectExtent = GridObject->MeshComponent->Bounds.BoxExtent;
            float PlatformZ = Platform->GetActorLocation().Z;
            bool ShouldRespawnGridObject = false;

            // 1) Check if the GridObject has reached its goal platform
            if (!GridObjectHasReachedGoal[AgentIndex])
            {
                int32 GoalIndex = AgentGoalIndices[AgentIndex];
                FVector GoalWorldPosition = GoalPlatforms[GoalIndex]->GetActorLocation();
                float DistanceToGoal = FVector::Dist(ObjectWorldPosition, GoalWorldPosition);

                if (DistanceToGoal <= (ObjectExtent.GetAbsMax() * GoalThreshold))
                {
                    AgentHasActiveGridObject[AgentIndex] = false;
                    GridObjectHasReachedGoal[AgentIndex] = true;
                    GridObjectShouldCollectEventReward[AgentIndex] = true;
                    ShouldRespawnGridObject = true;
                    GridObjectManager->DisableGridObject(AgentIndex);
                }
            }
           
            if (!GridObjectFallenOffGrid[AgentIndex])
            {
                // 2) Check if the bottom of GridObject has fallen below the platform
                float HalfExtent = (ObjectExtent.Z * 0.5f);
                float BottomZ = ObjectWorldPosition.Z - HalfExtent;
                if (BottomZ < PlatformZ)
                {
                    AgentHasActiveGridObject[AgentIndex] = false;
                    GridObjectFallenOffGrid[AgentIndex] = true;
                    GridObjectShouldCollectEventReward[AgentIndex] = true;
                    ShouldRespawnGridObject = true;
                    GridObjectManager->DisableGridObject(AgentIndex);
                }    

                // 3) Check if the GridObject is "too high" above the platform
                float TopZ = ObjectWorldPosition.Z + HalfExtent;
                float ZDiff = TopZ - PlatformZ;
                float ZMax = ObjectExtent.Z * 10.0; //arbitrary buts seems to work
                if (ZDiff > ZMax) 
                {
                    AgentHasActiveGridObject[AgentIndex] = false;
                    GridObjectFallenOffGrid[AgentIndex] = true;
                    ShouldRespawnGridObject = true;
                    GridObjectManager->DisableGridObject(AgentIndex);
                }
            }

            // 5) Respawn if object is out of bounds or reached goal
            if (ShouldRespawnGridObject)
            {
                GridObjectManager->DisableGridObject(AgentIndex);
                GridObjectShouldRespawn[AgentIndex] = true;
            }
        }
    }
}

void ATerraShiftEnvironment::UpdateObjectStats() {
    LastDeltaTime = GetWorld()->GetDeltaSeconds();
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
                PreviousPositions[i] = FVector::ZeroVector;
            }

            if (GridObjectShouldRespawn[i]) {
                GridObjectRespawnTimer[i] += LastDeltaTime;
            }
        }
    } 
}

void ATerraShiftEnvironment::RespawnGridObjects() 
{
    for (int32 AgentIndex = 0; AgentIndex < CurrentAgents; ++AgentIndex)
    {
        if 
        (
            GridObjectShouldRespawn[AgentIndex] && 
            GridObjectRespawnTimer[AgentIndex] > 
            GridObjectRespawnDelays[AgentIndex]
        )
        {
            // Generate a new random spawn location
            FVector NewSpawnLocation = GenerateRandomGridLocation();

            NumGoals = GoalPlatforms.Num();
            AgentGoalIndices[AgentIndex] = FMath::RandRange(0, NumGoals - 1);

            GridObjectManager->SpawnGridObjectAtIndex(
                AgentIndex,
                NewSpawnLocation,
                ObjectSize,
                ObjectMass
            );

            AgentHasActiveGridObject[AgentIndex] = true;
            GridObjectHasReachedGoal[AgentIndex] = false;
            GridObjectFallenOffGrid[AgentIndex] = false;
            GridObjectShouldRespawn[AgentIndex] = false;
            GridObjectRespawnTimer[AgentIndex] = 0.0;
        }
    }
}

FVector ATerraShiftEnvironment::CalculateGoalPlatformLocation(int32 EdgeIndex)
{
    // EdgeIndex: 0 - Top, 1 - Bottom, 2 - Left, 3 - Right
    FVector Location = FVector::ZeroVector;
    float Offset = (PlatformWorldSize.X / 2.0f) + (PlatformWorldSize.X * ObjectSize.X / 2);

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
    FVector GoalPlatformScale = ObjectSize;

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
