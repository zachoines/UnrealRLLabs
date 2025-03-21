#include "TerraShiftEnvironment.h"

ATerraShiftEnvironment::ATerraShiftEnvironment()
{
    PrimaryActorTick.bCanEverTick = true;
    PrimaryActorTick.TickGroup = TG_PostUpdateWork;
    MaxAgents = 5;
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

    CurrentObjectVelocities.Init(FVector::ZeroVector, CurrentAgents);
    CurrentObjectAcceleration.Init(FVector::ZeroVector, CurrentAgents);
    CurrentDistances.Init(-1.0f, CurrentAgents);
    CurrentPositions.Init(FVector::One() * -1.0f, CurrentAgents);

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
    // NumGoals = Math::RandRange(1, 4);

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
    
    // 1) Retrieve object's relative world position/velocity/acceleration if active
    FVector ObjectRelativePosition = CurrentPositions[AgentIndex];
    FVector ObjectRelativeVelocity = CurrentObjectVelocities[AgentIndex];
    FVector ObjectRelativeAcceleration = CurrentObjectAcceleration[AgentIndex];

    FVector PreviousObjectRelativePosition = PreviousPositions[AgentIndex];
    FVector PreviousObjectRelativeVelocity = PreviousObjectVelocities[AgentIndex];
    FVector PreviousObjectRelativeAcceleration = PreviousObjectAcceleration[AgentIndex];

    float DistanceToGoal = CurrentDistances[AgentIndex];
    float PreviousDistanceToGoal = PreviousDistances[AgentIndex];

    // 2) Retrieve the goal platform's position in relative coords
    int32 AgentGoalIndex = AgentGoalIndices[AgentIndex];
    FVector GoalRelativePosition = Platform->GetActorTransform().InverseTransformPosition(
        GoalPlatforms[AgentGoalIndex]->GetActorLocation()
    );


    // 3.) Scale state
    DistanceToGoal /= PlatformWorldSize.X;
    ObjectRelativePosition /= PlatformWorldSize.X;
    GoalRelativePosition /= PlatformWorldSize.X;
    ObjectRelativeVelocity /= PlatformWorldSize.X;
    ObjectRelativeAcceleration /= 980.0; // Scale by gravity
    
    PreviousDistanceToGoal /= PlatformWorldSize.X;
    PreviousObjectRelativePosition /= PlatformWorldSize.X;
    PreviousObjectRelativeVelocity /= PlatformWorldSize.X;
    PreviousObjectRelativeAcceleration /= 980.0; // Scale by gravity


    // 4) Add object and goal positions (X,Y,Z)
    State.Add(ObjectRelativePosition.X);
    State.Add(ObjectRelativePosition.Y);
    State.Add(ObjectRelativePosition.Z);

    State.Add(GoalRelativePosition.X);
    State.Add(GoalRelativePosition.Y);
    State.Add(GoalRelativePosition.Z);

    // 5) Add object velocity (X,Y,Z)
    State.Add(ObjectRelativeVelocity.X);
    State.Add(ObjectRelativeVelocity.Y);
    State.Add(ObjectRelativeVelocity.Z);

    // 6) Add object acceleration (X,Y,Z)
    State.Add(ObjectRelativeAcceleration.X);
    State.Add(ObjectRelativeAcceleration.Y);
    State.Add(ObjectRelativeAcceleration.Z);

    // 7) Add previous object velocity (X, Y, Z)
    /*State.Add(PreviousObjectRelativeVelocity.X);
    State.Add(PreviousObjectRelativeVelocity.Y);
    State.Add(PreviousObjectRelativeVelocity.Z);*/

    // 9.) Add previous object acceleration (X, Y, Z)
    /*State.Add(PreviousObjectRelativeAcceleration.X);
    State.Add(PreviousObjectRelativeAcceleration.Y);
    State.Add(PreviousObjectRelativeAcceleration.Z);*/

    // 10)  Add previous object distance to the assigned goal
    // State.Add(PreviousDistanceToGoal);

    // 11) Add previous object position (X,Y,Z)
    State.Add(PreviousObjectRelativePosition.X);
    State.Add(PreviousObjectRelativePosition.Y);
    State.Add(PreviousObjectRelativePosition.Z);

    // 7) Flag: is the GridObject active?
    State.Add(AgentHasActiveGridObject[AgentIndex] ? 1.0f : 0.0f);

    // 8) Add the (GoalIndex + 1) if active, else 0, plus distance to goal
    State.Add(AgentHasActiveGridObject[AgentIndex] ? static_cast<float>(AgentGoalIndex + 1) : 0.0f);
    State.Add(DistanceToGoal);
    State.Add(LastDeltaTime);

    // 9) Agent Wave State
    State.Append(WaveSimulator->GetAgentStateVariables(AgentIndex));
    
    // State.Append(WaveSimulator->GetAgentFractalImage(AgentIndex));
    
    return State;
}

void ATerraShiftEnvironment::Act(FAction Action)
{
    // Now we define 7 floats per agent in the range [-1..1]:
    //   [0] => dPitch
    //   [1] => dYaw
    //   [2] => dRoll
    //   [3] => dBaseFreq
    //   [4] => dLacunarity
    //   [5] => dGain
    //   [6] => dBlendWeight
    //   [7] => dFov
    //   [8] => dSampleDist

    const int32 ValuesPerAgent = 7;

    if (Action.Values.Num() != CurrentAgents * ValuesPerAgent)
    {
        UE_LOG(LogTemp, Error, TEXT("Act: mismatch (#values=%d, expected=%d)"),
            Action.Values.Num(), CurrentAgents * ValuesPerAgent);
        return;
    }

    // Prepare array of fractal actions
    TArray<FFractalAgentAction> FractalActions;
    FractalActions.SetNum(CurrentAgents);

    // Parse inputs
    for (int32 i = 0; i < CurrentAgents; i++)
    {
        const int32 BaseIndex = i * ValuesPerAgent;

        float dPitch = Action.Values[BaseIndex + 0];
        float dYaw = Action.Values[BaseIndex + 1];
        float dRoll = Action.Values[BaseIndex + 2];
        float dBaseFreq = Action.Values[BaseIndex + 3];
        float dBlendWeight = Action.Values[BaseIndex + 4];
        float dSampleDist = Action.Values[BaseIndex + 5];
        float dFov = Action.Values[BaseIndex + 6];
        // float dLacunarity = Action.Values[BaseIndex + 6];
        // float dGain = Action.Values[BaseIndex + 7];
        

        FFractalAgentAction& FA = FractalActions[i];
        FA.dPitch = FMath::Clamp(dPitch, -1.0f, 1.0f);
        FA.dYaw = FMath::Clamp(dYaw, -1.0f, 1.0f);
        FA.dRoll = FMath::Clamp(dRoll, - 1.0f, 1.0f);
        FA.dBaseFreq = FMath::Clamp(dBaseFreq, -1.0f, 1.0f);
        FA.dBlendWeight = FMath::Clamp(dBlendWeight, -1.0f, 1.0f);
        FA.dSampleDist = FMath::Clamp(dSampleDist, -1.0f, 1.0f);
        FA.dFOV = FMath::Clamp(dFov, -1.0f, 1.0f);
        // FA.dLacunarity = FMath::Clamp(dLacunarity, -1.0f, 1.0f);
        // FA.dGain = FMath::Clamp(dGain, -1.0f, 1.0f);
        
        
    }

    // Step fractal wave environment
    if (!WaveSimulator)
    {
        UE_LOG(LogTemp, Warning, TEXT("Act: WaveSimulator is null."));
        return;
    }

    WaveSimulator->Step(FractalActions, GetWorld()->GetDeltaSeconds());

    // final wave => NxN in [-1..1]
    const FMatrix2D& WaveMap = WaveSimulator->GetWave();

    // Apply wave to your grid or columns
    Grid->UpdateColumnHeights(WaveMap * MaxColumnHeight);
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
        //if (ActiveColumns.Contains(ColumnIndex))
        //{
        //    // Already colored active columns black
        //    continue;
        //}
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

    
    // Store previous object information
    UpdateObjectStats();


    // Handle objects marked for respawn. Needs to happen before we call 
    RespawnGridObjects();
}

void ATerraShiftEnvironment::PostTransition()
{

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
    // If there are no more GridObjects left to handle, then the environment is done
    if (CurrentStep > 0 && (!AgentHasActiveGridObject.Contains(true) && !GridObjectShouldRespawn.Contains(true))) {
        return true;
    }

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
    if (LastDeltaTime < KINDA_SMALL_NUMBER) {
        // If dt is extremely small, skip 
        return 0.0f;
    }

    for (int32 AgentIndex = 0; AgentIndex < CurrentAgents; ++AgentIndex)
    {
        // ------------------ (A) Handle event-based rewards first ------------------
        if (GridObjectShouldCollectEventReward[AgentIndex] && GridObjectFallenOffGrid[AgentIndex])
        {
            StepReward += FALL_OFF_PENALTY;
            GridObjectShouldCollectEventReward[AgentIndex] = false;
            continue;
        }
        else if (GridObjectShouldCollectEventReward[AgentIndex] && GridObjectHasReachedGoal[AgentIndex])
        {
            StepReward += REACH_GOAL_REWARD;
            GridObjectShouldCollectEventReward[AgentIndex] = false;
            continue;
        }

        if (!AgentHasActiveGridObject[AgentIndex]) {
            StepReward += STEP_PENALTY;
            continue;
        }

        AGridObject* GridObject = GridObjectManager->GetGridObject(AgentIndex);
        AGoalPlatform* AssignedGoal = GoalPlatforms[AgentGoalIndices[AgentIndex]];
        if (!GridObject || !AssignedGoal)
        {
            continue;
        }

        FVector ObjectWorldPos = GridObject->GetObjectLocation();
        FVector ObjectWorldVel = GridObject->MeshComponent->GetPhysicsLinearVelocity();
        FVector ObjLocalPos = Platform->GetActorTransform().InverseTransformPosition(ObjectWorldPos);
        FVector VelLocal = Platform->GetActorTransform().InverseTransformVector(ObjectWorldVel);

        FVector GoalLocalPos = Platform->GetActorTransform().InverseTransformPosition(
            AssignedGoal->GetActorLocation()
        );

        float subReward = 0.0f;

        // ------------------------------------------------------------
        // Velocity to Goal (3D alignment)
        // ------------------------------------------------------------
        if (bUseVelAlignment)
        {
            FVector toGoal3D = GoalLocalPos - ObjLocalPos;
            float dist3D = toGoal3D.Size();
            float speed3D = VelLocal.Size();

            // Only do alignment calc if we're not basically at the goal
            if (dist3D > KINDA_SMALL_NUMBER)
            {
                const FVector goalDir = toGoal3D.GetSafeNormal();

                // Dot product => "speed along the goal direction"
                float speedAlongGoal = FVector::DotProduct(VelLocal, goalDir);
                float clampedSpeed = FMath::Clamp(speedAlongGoal, VelAlign_Min, VelAlign_Max) / VelAlign_Max;
                float velReward = VelAlign_Scale * clampedSpeed;
                subReward += velReward;
            }
        }

        // ------------------------------------------------------------
        // Distance Improvement (XY)
        // ------------------------------------------------------------
        if (bUseXYDistanceImprovement && PreviousDistances[AgentIndex] > 0.0f)
        {
            float prevDist = PreviousDistances[AgentIndex];  // stored from last step
            float currDist = FVector::Distance(ObjLocalPos, GoalLocalPos);

            if (prevDist > 0.f && currDist > 0.f)
            {
                float distDelta = (prevDist - currDist) / PlatformWorldSize.X;
                float clampedDelta = FMath::Clamp(distDelta, DistImprove_Min, DistImprove_Max);

                float distReward = DistImprove_Scale * clampedDelta;
                subReward += distReward;
            }
        }

        // ------------------------------------------------------------
        // 2) Z Acceleration Penalty
        // ------------------------------------------------------------
        if (bUseZAccelerationPenalty && PreviousObjectVelocities[AgentIndex] != FVector::ZeroVector)
        {

            FVector aTotal = (VelLocal - PreviousObjectVelocities[AgentIndex]) / LastDeltaTime;

            // Example: only positive Z => launching up
            float positiveZ = (aTotal.Z > 0.f) ? aTotal.Z : 0.f;
            float clampedZ = ThresholdAndClamp(positiveZ, ZAccel_Min, ZAccel_Max);

            // penalty => subtract
            subReward -= (ZAccel_Scale * clampedZ);
        }

        // accumulate subReward to global StepReward
        StepReward += subReward;
    }

    return StepReward * LastDeltaTime;
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

    float spawnCollisionRadius = CellSize * 4.0f;

    // 1) Build distance matrix
    FMatrix2D distMatrix = ComputeCollisionDistanceMatrix();

    // 2) We'll gather all cell indices that are "free enough"
    TArray<int32> FreeIndices;
    FreeIndices.Reserve(GridSize * GridSize);

    // define the margin in #cells or could do minCell, maxCell approach
    int32 marginCells = GridSize / 4;
    int32 minCell = marginCells;
    int32 maxCell = GridSize - marginCells - 1;

    for (int32 X = minCell; X <= maxCell; ++X)
    {
        for (int32 Y = minCell; Y <= maxCell; ++Y)
        {
            float dist = distMatrix[X][Y];
            if (dist > spawnCollisionRadius)
            {
                // This cell is far enough from any object
                int32 idx = X * GridSize + Y;
                FreeIndices.Add(idx);
            }
        }
    }

    if (FreeIndices.Num() == 0)
    {
        // fallback => no free cell found
        UE_LOG(LogTemp, Warning, TEXT("No free cell found in linear algebra approach. Using fallback."));
        FVector fallback = PlatformCenter;
        fallback.Z = Grid->GetActorLocation().Z + 20.f;
        return fallback;
    }

    // 3) pick random cell from FreeIndices
    int32 chosenIdx = FreeIndices[FMath::RandRange(0, FreeIndices.Num() - 1)];
    FVector location = GridCenterPoints[chosenIdx];
    location.Z = Grid->GetActorLocation().Z + (CellSize * MaxColumnHeight); // slightly above wave peak
    return location;
}

FMatrix2D ATerraShiftEnvironment::ComputeCollisionDistanceMatrix() const
{
    // Create matrix => (GridSize x GridSize), initially large distances
    FMatrix2D distMatrix(GridSize, GridSize, 1e6f);

    // For each active object
    for (int32 i = 0; i < CurrentAgents; ++i)
    {
        if (AgentHasActiveGridObject[i])
        {
            AGridObject* Obj = GridObjectManager->GetGridObject(i);
            if (!Obj) continue;

            FVector ObjLocation = Obj->GetObjectLocation();
            float ObjRadius = Obj->MeshComponent->Bounds.SphereRadius;
            float EffectiveRadius = ObjRadius * 1.5f;

            // (Similar logic to GetActiveColumnsInProximity)
            // Map bounding box to cell indices
            float HalfSize = PlatformWorldSize.X * 0.5f;
            float GridOriginX = PlatformCenter.X - HalfSize;
            float GridOriginY = PlatformCenter.Y - HalfSize;

            FVector MinCorner = ObjLocation - FVector(EffectiveRadius, EffectiveRadius, 0.f);
            FVector MaxCorner = ObjLocation + FVector(EffectiveRadius, EffectiveRadius, 0.f);

            int32 MinXIndex = FMath::FloorToInt((MinCorner.X - GridOriginX) / CellSize);
            int32 MaxXIndex = FMath::FloorToInt((MaxCorner.X - GridOriginX) / CellSize);
            int32 MinYIndex = FMath::FloorToInt((MinCorner.Y - GridOriginY) / CellSize);
            int32 MaxYIndex = FMath::FloorToInt((MaxCorner.Y - GridOriginY) / CellSize);

            // Clamp
            MinXIndex = FMath::Clamp(MinXIndex, 0, GridSize - 1);
            MaxXIndex = FMath::Clamp(MaxXIndex, 0, GridSize - 1);
            MinYIndex = FMath::Clamp(MinYIndex, 0, GridSize - 1);
            MaxYIndex = FMath::Clamp(MaxYIndex, 0, GridSize - 1);

            // For each cell in that bounding region
            for (int32 X = MinXIndex; X <= MaxXIndex; ++X)
            {
                for (int32 Y = MinYIndex; Y <= MaxYIndex; ++Y)
                {
                    int32 Index = X * GridSize + Y;
                    const FVector& CellCenter = GridCenterPoints[Index];

                    // Actual 2D distance in XY plane
                    float dist = FVector::Dist2D(ObjLocation, CellCenter);

                    // If dist < stored value => update matrix
                    if (dist < distMatrix[X][Y])
                    {
                        distMatrix[X][Y] = dist;
                    }
                }
            }
        }
    }

    return distMatrix;
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
    // Precompute bounding box data
    float HalfX = PlatformWorldSize.X * 0.5f;
    float HalfY = PlatformWorldSize.Y * 0.5f;

    // Example margin factor (e.g., 1 or 2 times CellSize)
    float MarginXY = CellSize * 1.5f;

    // Vertical bounds: let’s allow from (PlatformZ) up to (Grid->GetActorLocation().Z + some margin)
    // Because Grid is placed at PlatformZ + MaxColumnHeight, we can allow it a bit above that.
    float MinZ = Platform->GetActorLocation().Z - (CellSize * 2.0f);  // a bit below platform
    float MaxZ = Grid->GetActorLocation().Z + (CellSize * MaxColumnHeight * 2.0f);      // a bit above the grid

    for (int32 AgentIndex = 0; AgentIndex < CurrentAgents; ++AgentIndex)
    {
        if (AgentHasActiveGridObject[AgentIndex])
        {
            AGridObject* GridObject = GridObjectManager->GetGridObject(AgentIndex);
            FVector ObjectWorldPosition = GridObject->GetObjectLocation();
            FVector ObjectExtent = GridObject->MeshComponent->Bounds.BoxExtent;

            bool ShouldRespawnGridObject = false;

            // 1) Check if the GridObject has reached its goal platform
            if (!GridObjectHasReachedGoal[AgentIndex])
            {
                int32 GoalIndex = AgentGoalIndices[AgentIndex];
                FVector GoalWorldPosition = GoalPlatforms[GoalIndex]->GetActorLocation();
                float DistanceToGoal = FVector::Dist(ObjectWorldPosition, GoalWorldPosition);

                // If within threshold => reached goal => disable, do NOT respawn
                if (DistanceToGoal <= (ObjectExtent.GetAbsMax() * GoalThreshold))
                {
                    AgentHasActiveGridObject[AgentIndex] = false;
                    GridObjectHasReachedGoal[AgentIndex] = true;
                    GridObjectShouldCollectEventReward[AgentIndex] = true;
                    ShouldRespawnGridObject = false;  // no respawn => it's done
                    GridObjectManager->DisableGridObject(AgentIndex);
                }
            }

            // 2) If NOT already flagged as 'fallen off'
            if (!GridObjectFallenOffGrid[AgentIndex])
            {
                // We only do bounding-box check if it hasn't reached the goal or already failed
                if (!GridObjectHasReachedGoal[AgentIndex])
                {
                    // Horizontal check
                    float dx = FMath::Abs(ObjectWorldPosition.X - PlatformCenter.X);
                    float dy = FMath::Abs(ObjectWorldPosition.Y - PlatformCenter.Y);

                    // Vertical check
                    float zPos = ObjectWorldPosition.Z;

                    bool bOutOfBounds = false;

                    if (dx > (HalfX + MarginXY) || dy > (HalfY + MarginXY))
                    {
                        bOutOfBounds = true;
                    }
                    else if (zPos < MinZ || zPos > MaxZ)
                    {
                        bOutOfBounds = true;
                    }

                    if (bOutOfBounds)
                    {
                        // Mark as fallen
                        AgentHasActiveGridObject[AgentIndex] = false;
                        GridObjectFallenOffGrid[AgentIndex] = true;
                        GridObjectShouldCollectEventReward[AgentIndex] = true;
                        ShouldRespawnGridObject = false;   // no respawn => it's done
                        GridObjectManager->DisableGridObject(AgentIndex);
                    }
                }
            }

            // 3) If object is out of bounds or flagged for respawn, mark to respawn
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
                FVector ObjectRelativeAcceleration = !FMath::IsNearlyZero(LastDeltaTime) ? (ObjectRelativeVelocity - CurrentObjectVelocities[i]) / LastDeltaTime : FVector::Zero();
             
                // Update currenent/previous states for later use in Rewards/Dones/State/etc.
                PreviousObjectVelocities[i] = CurrentObjectVelocities[i];
                PreviousObjectAcceleration[i] = CurrentObjectAcceleration[i];
                PreviousDistances[i] = CurrentDistances[i];
                PreviousPositions[i] = CurrentPositions[i];

                CurrentObjectVelocities[i] = ObjectRelativeVelocity;
                CurrentObjectAcceleration[i] = ObjectRelativeAcceleration;
                CurrentDistances[i] = FVector::Dist(ObjectRelativePosition, GoalRelativePosition);
                CurrentPositions[i] = ObjectRelativePosition;
            }
            else
            {
                PreviousObjectVelocities[i] = FVector::ZeroVector;
                PreviousObjectAcceleration[i] = FVector::ZeroVector;
                PreviousDistances[i] = -1;
                PreviousPositions[i] = FVector::ZeroVector;

                CurrentObjectVelocities[i] = FVector::ZeroVector;
                CurrentObjectAcceleration[i] = FVector::ZeroVector;
                CurrentDistances[i] = -1;
                CurrentPositions[i] = FVector::ZeroVector;
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
        if (GridObjectShouldRespawn[AgentIndex] && 
            GridObjectRespawnTimer[AgentIndex] > GridObjectRespawnDelays[AgentIndex])
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
            GridObjectFallenOffGrid[AgentIndex]  = false;
            GridObjectShouldRespawn[AgentIndex]  = false;
            GridObjectRespawnTimer[AgentIndex]   = 0.0f;
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

float ATerraShiftEnvironment::ThresholdAndClamp(float value, float minVal, float maxVal)
{
    // 1) If abs(value) < minVal => return 0
    if (FMath::Abs(value) < minVal)
    {
        return 0.0;
    }

    // 2) Clamp magnitude to maxVal
    float clamped = FMath::Clamp(value, -maxVal, maxVal);
    return clamped;
}