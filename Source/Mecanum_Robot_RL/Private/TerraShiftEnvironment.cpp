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

    // Replaced UMorletWavelets2D with UDiscreteFourier2D
    WaveSimulator = CreateDefaultSubobject<UDiscreteFourier2D>(TEXT("WaveSimulator"));
    Grid = nullptr;
    Initialized = false;

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
    GridObjectShouldRespawn.SetNum(CurrentAgents);
    GridObjectRespawnTimer.SetNum(CurrentAgents);
    GridObjectRespawnDelays.SetNum(CurrentAgents);

    // Set the environment root's world location
    TerraShiftRoot->SetWorldLocation(TerraShiftParams->Location);

    // Spawn the platform at the specified location
    Platform = SpawnPlatform(TerraShiftParams->Location);
    check(Platform != nullptr);

    // Scale the platform based on the specified PlatformSize
    Platform->SetActorScale3D(FVector(TerraShiftParams->PlatformSize));

    // Calculate platform dimensions and determine grid cell size
    PlatformWorldSize = Platform->PlatformMeshComponent->GetStaticMesh()->GetBounds().BoxExtent
        * 2.0f
        * Platform->GetActorScale3D();
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
            float CenterX = PlatformCenter.X - (PlatformWorldSize.X / 2.0f)
                + (X + 0.5f) * CellSize;
            float CenterY = PlatformCenter.Y - (PlatformWorldSize.Y / 2.0f)
                + (Y + 0.5f) * CellSize;
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
        // Set the PlatformActor in GridObjectManager
        GridObjectManager->SetPlatformActor(Platform);

        // Set the folder path for GridObjectManager
        GridObjectManager->SetFolderPath(FName(*(EnvironmentFolderPath + "/GridObjectManager")));
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to spawn GridObjectManager"));
    }

    // We expect WaveSimulator to be valid
    check(WaveSimulator != nullptr);

    // Now initialize the discrete Fourier simulator
    WaveSimulator->Initialize(
        TerraShiftParams->GridSize,
        TerraShiftParams->GridSize,
        TerraShiftParams->MaxAgents,
        TerraShiftParams->K
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

    // Resize agent-related arrays
    AgentGoalIndices.Init(-1, CurrentAgents);
    AgentHasActiveGridObject.Init(false, CurrentAgents);
    GridObjectFallenOffGrid.Init(false, CurrentAgents);
    GridObjectHasReachedGoal.Init(false, CurrentAgents);
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
    int32 NumGoals = FMath::RandRange(1, 4);
    TerraShiftParams->NumGoals = NumGoals;

    // Create goal platforms
    for (int32 i = 0; i < NumGoals; ++i)
    {
        UpdateGoal(i);
    }

    for (int32 i = 0; i < CurrentAgents; ++i)
    {
        // Set the spawn time for each
        GridObjectRespawnDelays[i] = ((float)i) * TerraShiftParams->SpawnDelay;

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

    // 1) Retrieve discrete Fourier info for this agent
    TArray<float> FourierState = WaveSimulator->GetAgentFourierState(AgentIndex);
    State.Append(FourierState);

    // 2) Check if the agent currently has an active GridObject
    bool bHasActiveObject = AgentHasActiveGridObject[AgentIndex];

    // 3) Retrieve object world position and velocity if active
    FVector ObjectWorldPosition = FVector::ZeroVector;
    FVector ObjectWorldVelocity = FVector::ZeroVector;
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

    // 5) Retrieve goal position
    int32 AgentGoalIndex = AgentGoalIndices[AgentIndex];
    FVector GoalRelativePosition = Platform->GetActorTransform().InverseTransformPosition(GoalPlatforms[AgentGoalIndex]->GetActorLocation());

    // 6) Compute distance to the assigned goal (if object is active)
    float DistanceToGoal = -1.0f;
    if (bHasActiveObject)
    {
        AGoalPlatform* AssignedGoal = GoalPlatforms[AgentGoalIndex];
        if (AssignedGoal)
        {
            DistanceToGoal = FVector::Dist(ObjectRelativePosition, GoalRelativePosition);
        }
    }

    // 7) Add positions (object, goal)
    State.Add(ObjectRelativePosition.X);
    State.Add(ObjectRelativePosition.Y);
    State.Add(ObjectRelativePosition.Z);
    State.Add(GoalRelativePosition.X);
    State.Add(GoalRelativePosition.Y);
    State.Add(GoalRelativePosition.Z);

    // 8) Add object velocity (X, Y, Z)
    State.Add(ObjectRelativeVelocity.X);
    State.Add(ObjectRelativeVelocity.Y);
    State.Add(ObjectRelativeVelocity.Z);

    // 9) Add object’s relative acceleration (X, Y, Z)
    if (PreviousObjectVelocities[AgentIndex] == FVector::ZeroVector)
    {
        State.Add(ObjectRelativeAcceleration.X);
        State.Add(ObjectRelativeAcceleration.Y);
        State.Add(ObjectRelativeAcceleration.Z);
    }
    else 
    {
        State.Add(0.0);
        State.Add(0.0);
        State.Add(0.0);
    }
    

    // 10) Add previous object velocity (X, Y, Z)
    State.Add(PreviousObjectVelocities[AgentIndex].X);
    State.Add(PreviousObjectVelocities[AgentIndex].Y);
    State.Add(PreviousObjectVelocities[AgentIndex].Z);

    // 11) Add previous object acceleration (X, Y, Z)
    State.Add(PreviousObjectAcceleration[AgentIndex].X);
    State.Add(PreviousObjectAcceleration[AgentIndex].Y);
    State.Add(PreviousObjectAcceleration[AgentIndex].Z);

    // 12)  Add previous object distance
    State.Add(PreviousDistances[AgentIndex]);

    // 13) Add previous object position
    State.Add(PreviousPositions[AgentIndex].X);
    State.Add(PreviousPositions[AgentIndex].Y);
    State.Add(PreviousPositions[AgentIndex].Z);

    // 14) Add a flag indicating if the GridObject is active
    State.Add(bHasActiveObject ? 1.0f : 0.0f);

    // 15) Add the goal index offset (AgentGoalIndex + 1) and distance to goal
    State.Add(bHasActiveObject ? static_cast<float>(AgentGoalIndex + 1) : 0.0f);
    State.Add(DistanceToGoal);

    return State;
}

void ATerraShiftEnvironment::Act(FAction Action)
{
    // Example: Suppose each agent has 6 discrete actions:
    //   [0] = dPhaseX, [1] = dPhaseY, [2] = dFreqScale
    //   [3] = rowIndex, [4] = colIndex, [5] = deltaValue
    // This is just one possible interpretation.
    const int NumAgentActions = 6;

    // Verify correct number of inputs
    if (Action.Values.Num() != CurrentAgents * NumAgentActions)
    {
        UE_LOG(LogTemp, Error, TEXT("Action array size mismatch. Expected %d, got %d"),
            CurrentAgents * NumAgentActions, Action.Values.Num());
        return;
    }

    // Prepare an array of FAgentFourierDelta (one per agent)
    TArray<FAgentFourierDelta> FourierDeltas;
    FourierDeltas.SetNum(CurrentAgents);

    // We'll also retrieve DeltaTime if we need it
    float DeltaTime = GetWorld()->GetDeltaSeconds();

    // Retrieve K from TerraShiftParams for building the partial (2K x 2K) updates
    const int32 KVal = TerraShiftParams->K; // e.g. 8 by default
    int32 Dim = 2 * KVal;

    for (int32 i = 0; i < CurrentAgents; ++i)
    {
        // Step 1: Extract delta increments
        int32 BaseIndex = i * NumAgentActions;

        float dPhaseX = 0.0;
        if (FMath::FloorToInt(Action.Values[BaseIndex + 0]) == 0)
        {
            dPhaseX += TerraShiftParams->PhaseXDeltaRange.X;
        }
        else if (FMath::FloorToInt(Action.Values[BaseIndex + 0]) == 2)
        {
            dPhaseX += TerraShiftParams->PhaseXDeltaRange.Y;
        }

        float dPhaseY = 0.0;
        if (FMath::FloorToInt(Action.Values[BaseIndex + 1]) == 0)
        {
            dPhaseY += TerraShiftParams->PhaseYDeltaRange.X;
        }
        else if (FMath::FloorToInt(Action.Values[BaseIndex + 1]) == 2)
        {
            dPhaseY += TerraShiftParams->PhaseYDeltaRange.Y;
        }


        float dFreqScale = 0.0;
        if (FMath::FloorToInt(Action.Values[BaseIndex + 2]) == 0)
        {
            dFreqScale += TerraShiftParams->FreqScaleDeltaRange.X;
        }
        else if (FMath::FloorToInt(Action.Values[BaseIndex + 2]) == 2)
        {
            dFreqScale += TerraShiftParams->FreqScaleDeltaRange.Y;
        }


        float deltaVal = 0.0;
        if (FMath::FloorToInt(Action.Values[BaseIndex + 5]) == 0)
        {
            deltaVal += TerraShiftParams->MatrixDeltaRange.X;
        }
        else if (FMath::FloorToInt(Action.Values[BaseIndex + 5]) == 2)
        {
            deltaVal += TerraShiftParams->MatrixDeltaRange.Y;
        }

        // interpret rowChoice, colChoice as indices in [0..(2K-1)]
        int32 rowIndex = FMath::Clamp(FMath::FloorToInt(Action.Values[BaseIndex + 3]), 0, Dim - 1);
        int32 colIndex = FMath::Clamp(FMath::FloorToInt(Action.Values[BaseIndex + 4]), 0, Dim - 1);

        // build the partial matrix update => zero matrix except for one cell
        FMatrix2D DeltaA(Dim, Dim, 0.0f);
        DeltaA[rowIndex][colIndex] = deltaVal;

        // fill out FAgentFourierDelta
        FAgentFourierDelta& FD = FourierDeltas[i];
        FD.dPhaseX = dPhaseX;
        FD.dPhaseY = dPhaseY;
        FD.dFreqScale = dFreqScale;
        FD.DeltaA = DeltaA;
    }

    // Step 6: Pass these deltas to the discrete Fourier simulator
    //         We'll get back a new NxN height map
    const FMatrix2D& HeightMap = WaveSimulator->Update(FourierDeltas);

    // Step 7: Update the actual grid columns
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
    for (int32 AgentIndex = 0; AgentIndex < CurrentAgents; ++AgentIndex) 
    {
        if (GridObjectFallenOffGrid[AgentIndex] || GridObjectHasReachedGoal[AgentIndex])
        {
            return true;
        }
    }

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
    float StepReward = 0.0f;

    for (int32 AgentIndex = 0; AgentIndex < CurrentAgents; ++AgentIndex)
    {
        bool bActive = AgentHasActiveGridObject[AgentIndex];
        bool bReachedGoal = GridObjectHasReachedGoal[AgentIndex];
        bool bHasFallen = GridObjectFallenOffGrid[AgentIndex];

        // (A) Punish falling off platform => Episode terminates
        if (bHasFallen)
        {
            // Larger penalty since it fully ends the agent's run
            StepReward -= 2.0f;
        }
        // (B) Reached goal => Respawn object
        else if (bReachedGoal)
        {
            // Medium reward for success, 
            // but environment doesn't reset, so not too large
            StepReward += 2.0f;
        }
        else if (bActive) // Still in play
        {
            AGridObject* GridObject = GridObjectManager->GetGridObject(AgentIndex);
            AGoalPlatform* AssignedGoal = GoalPlatforms[AgentGoalIndices[AgentIndex]];

            // 1) DISTANCE IMPROVEMENT
            if (PreviousDistances[AgentIndex] > 0.0f)
            {
                // newDist in local platform coords
                float newDist = FVector::Distance(
                    Platform->GetActorTransform().InverseTransformPosition(GridObject->GetObjectLocation()),
                    Platform->GetActorTransform().InverseTransformPosition(AssignedGoal->GetActorLocation())
                );

                // improvement fraction = (oldDist - newDist) / platform_size
                float improvementFraction =
                    (PreviousDistances[AgentIndex] - newDist) / (PlatformWorldSize.X + KINDA_SMALL_NUMBER);

                // scale the fraction by "improvement_reward_scale"
                float improvement_reward_scale = 1.0f; // tune as needed
                StepReward += improvementFraction * improvement_reward_scale;
            }

            // 2) VELOCITY TOWARD GOAL
            {
                // We'll compute a dot product in local coords 
                // between velocity and direction to the goal
                FVector ObjectLocalVel = Platform->GetActorTransform().InverseTransformVector(
                    GridObject->MeshComponent->GetPhysicsLinearVelocity()
                );

                FVector ObjLocalPos = Platform->GetActorTransform().InverseTransformPosition(
                    GridObject->GetObjectLocation()
                );
                FVector GoalLocalPos = Platform->GetActorTransform().InverseTransformPosition(
                    AssignedGoal->GetActorLocation()
                );
                FVector toGoal = (GoalLocalPos - ObjLocalPos);
                float distGoal = toGoal.Size();
                if (distGoal > KINDA_SMALL_NUMBER)
                {
                    toGoal /= distGoal; // unit direction
                }

                // Dot product => + if going toward, - if away
                float dotToGoal = FVector::DotProduct(ObjectLocalVel, toGoal);

                // scale by a "velocity_direction_scale," e.g. 0.01
                // so 1 m/s in the correct direction => +0.001 reward
                float velocity_direction_scale = 0.001f;
                StepReward += (dotToGoal * velocity_direction_scale);
            }

            // 3) TINY STEP PENALTY
            // to ensure the agent doesn't stall infinitely
            StepReward -= 0.001f;
        }
    }

    return StepReward;
}

void ATerraShiftEnvironment::SetupActionAndObservationSpace()
{
    // 1) Calculate SingleAgentObsSize based on updated AgentGetState logic.
    //    - 3 + (2K x 2K) from DiscreteFourier => phaseX, phaseY, freqScale + flatten(A)
    //    - 25 additional floats from environment (object pos, velocity, etc.)
    //    => total = 28 + (2K*2K)

    // Retrieve K from TerraShiftParams (fallback to 8 if TerraShiftParams is null).
    int32 KVal = (TerraShiftParams) ? TerraShiftParams->K : 8;
    int32 Dim = 2 * KVal; // dimension of the AgentA matrix
    // Fourier part: 3 + (Dim*Dim)
    int32 FourierCount = 3 + (Dim * Dim);
    // Env part: 25
    int32 EnvCount = 25;

    EnvInfo.SingleAgentObsSize = FourierCount + EnvCount;
    // e.g., (3 + 16*16) + 25 = 28 + 256 = 284 if K=8

    // The overall environment state size (for multi-agent) 
    EnvInfo.StateSize = MaxAgents * EnvInfo.SingleAgentObsSize;

    // 2) Define the 6 discrete actions:
    //    0) dPhaseX    -> X possible choices
    //    1) dPhaseY    -> ...
    //    2) dFreqScale -> ...
    //    3) rowIndex   -> up to (2K)
    //    4) colIndex   -> up to (2K)
    //    5) deltaVal   -> ...
    // The exact number of choices is up to your design.

    TArray<FDiscreteActionSpec> DiscreteActions;
    DiscreteActions.SetNum(6);

    // We'll define:
    //  - dPhaseX has 3 possible increments
    //  - dPhaseY has 3
    //  - dFreqScale has 3
    //  - rowIndex in [0..(2K-1)] => 2K choices
    //  - colIndex in [0..(2K-1)] => 2K choices
    //  - deltaVal has 3 possible increments
    int32 NumRowChoices = Dim; // e.g., 16 if K=8
    int32 NumColChoices = Dim;

    DiscreteActions[0].NumChoices = 3;     // dPhaseX
    DiscreteActions[1].NumChoices = 3;     // dPhaseY
    DiscreteActions[2].NumChoices = 3;     // dFreqScale
    DiscreteActions[3].NumChoices = NumRowChoices; // rowIndex
    DiscreteActions[4].NumChoices = NumColChoices; // colIndex
    DiscreteActions[5].NumChoices = 3;     // deltaVal

    // 3) Initialize the action space with no continuous actions, only discrete:
    if (EnvInfo.ActionSpace)
    {
        EnvInfo.ActionSpace->Init({}, DiscreteActions);
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

                if (DistanceToGoal <= ObjectExtent.GetMax())
                {
                    AgentHasActiveGridObject[AgentIndex] = false;
                    GridObjectHasReachedGoal[AgentIndex] = true;
                    ShouldRespawnGridObject = true;
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
                    ShouldRespawnGridObject = true;
                }    

                // 3) Check if the GridObject is "too high" above the platform
                float TopZ = ObjectWorldPosition.Z + HalfExtent;
                float ZDiff = TopZ - PlatformZ;
                float ZMax = ObjectExtent.Z * 5.0; //arbitrary buts seems to work
                if (ZDiff > ZMax) 
                {
                    AgentHasActiveGridObject[AgentIndex] = false;
                    GridObjectFallenOffGrid[AgentIndex] = true;
                    ShouldRespawnGridObject = true;
                }
                
                // 4.) Check if Gridobject is to far off the grid

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

            int32 NumGoals = GoalPlatforms.Num();
            AgentGoalIndices[AgentIndex] = FMath::RandRange(0, NumGoals - 1);

            GridObjectManager->SpawnGridObjectAtIndex(
                AgentIndex,
                NewSpawnLocation,
                TerraShiftParams->ObjectSize,
                TerraShiftParams->ObjectMass
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
