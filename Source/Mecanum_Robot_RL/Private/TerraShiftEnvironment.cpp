#include "TerraShiftEnvironment.h"

ATerraShiftEnvironment::ATerraShiftEnvironment()
{
    EnvInfo.EnvID = 3;
    EnvInfo.IsMultiAgent = true;

    TerraShiftRoot = CreateDefaultSubobject<USceneComponent>(TEXT("TerraShiftRoot"));
    RootComponent = TerraShiftRoot;

    EnvInfo.ActionSpace = CreateDefaultSubobject<UActionSpace>(TEXT("ActionSpace"));
}

void ATerraShiftEnvironment::InitEnv(FBaseInitParams* BaseParams)
{
    TerraShiftParams = static_cast<FTerraShiftEnvironmentInitParams*>(BaseParams);

    MaxAgents = TerraShiftParams->MaxAgents;
    CurrentAgents = BaseParams->NumAgents;
    CurrentStep = 0;

    TerraShiftRoot->SetWorldLocation(TerraShiftParams->Location);

    Platform = SpawnPlatform(
        TerraShiftParams->Location,
        FVector(TerraShiftParams->GroundPlaneSize, TerraShiftParams->GroundPlaneSize, TerraShiftParams->GroundPlaneSize)
    );

    Columns.SetNum(TerraShiftParams->GridSize * TerraShiftParams->GridSize);
    GridCenterPoints.SetNum(TerraShiftParams->GridSize * TerraShiftParams->GridSize);

    FVector PlatformWorldSize = Platform->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().BoxExtent * 2.0f * TerraShiftParams->GroundPlaneSize;
    FVector PlatformCenter = Platform->GetActorLocation();

    float CellWidth = (TerraShiftParams->GroundPlaneSize / static_cast<float>(TerraShiftParams->GridSize)) - 1e-2;
    float CellHeight = CellWidth;

    for (int i = 0; i < TerraShiftParams->GridSize; ++i)
    {
        for (int j = 0; j < TerraShiftParams->GridSize; ++j)
        {
            AColumn* Column = GetWorld()->SpawnActor<AColumn>(AColumn::StaticClass());
            if (Column)
            {
                FVector GridCenter = PlatformCenter + FVector(
                    (i - TerraShiftParams->GridSize / 2.0f + 0.5f) * (PlatformWorldSize.X / TerraShiftParams->GridSize),
                    (j - TerraShiftParams->GridSize / 2.0f + 0.5f) * (PlatformWorldSize.Y / TerraShiftParams->GridSize),
                    CellHeight / 2
                );

                Column->InitColumn(FVector(CellWidth, CellWidth, CellHeight), GridCenter, TerraShiftParams->MaxColumnHeight);

                Columns[Get1DIndexFromPoint(FIntPoint(i, j), TerraShiftParams->GridSize)] = Column;
                GridCenterPoints[Get1DIndexFromPoint(FIntPoint(i, j), TerraShiftParams->GridSize)] = GridCenter;
            }
        }
    }

    EnvInfo.SingleAgentObsSize = 6;
    EnvInfo.StateSize = MaxAgents * EnvInfo.SingleAgentObsSize;

    TArray<FDiscreteActionSpec> DiscreteActions;
    DiscreteActions.Add({ 8 }); // Column direction actions
    DiscreteActions.Add({ 3 }); // Column height actions

    if (EnvInfo.ActionSpace)
    {
        EnvInfo.ActionSpace->Init({}, DiscreteActions);
    }

    UMaterial* DefaultObjectMaterial = LoadObject<UMaterial>(this, TEXT("Material'/Game/Material/Manip_Object_Material.Manip_Object_Material'"));
    UStaticMesh* DefaultObjectMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Sphere.Sphere"));

    // Spawn the maximum number of grid objects (MaxAgents)
    for (int i = 0; i < MaxAgents; ++i)
    {
        AGridObject* GridObject = GetWorld()->SpawnActor<AGridObject>(AGridObject::StaticClass());
        GridObject->InitializeGridObject(TerraShiftParams->ObjectSize, DefaultObjectMesh, DefaultObjectMaterial);
        Objects.Add(GridObject);

        // Initially hide all grid objects; only activate when required
        GridObject->SetGridObjectActive(false);
    }

    LastColumnIndexArray.SetNum(MaxAgents);
    AgentGoalIndices.SetNum(MaxAgents);

    SetActiveGridObjects(CurrentAgents);  
}

FState ATerraShiftEnvironment::ResetEnv(int NumAgents)
{
    CurrentStep = 0;
    CurrentAgents = NumAgents;

    // Reset goals and columns
    GoalPositionArray.SetNum(TerraShiftParams->NumGoals);
    TArray<FIntPoint> GoalIndices2D;
    GoalIndices2D.SetNum(TerraShiftParams->NumGoals);

    for (int i = 0; i < TerraShiftParams->NumGoals; ++i)
    {
        int Side;
        FIntPoint GoalIndex2D;
        do
        {
            Side = FMath::RandRange(0, 3);
            switch (Side)
            {
            case 0: // Top
                GoalIndex2D = FIntPoint(FMath::RandRange(0, TerraShiftParams->GridSize - 1), 0);
                break;
            case 1: // Bottom
                GoalIndex2D = FIntPoint(FMath::RandRange(0, TerraShiftParams->GridSize - 1), TerraShiftParams->GridSize - 1);
                break;
            case 2: // Left
                GoalIndex2D = FIntPoint(0, FMath::RandRange(0, TerraShiftParams->GridSize - 1));
                break;
            case 3: // Right
                GoalIndex2D = FIntPoint(TerraShiftParams->GridSize - 1, FMath::RandRange(0, TerraShiftParams->GridSize - 1));
                break;
            }
        } while (GoalIndices2D.Contains(GoalIndex2D));

        GoalIndices2D[i] = GoalIndex2D;
        GoalPositionArray[i] = GridCenterPoints[Get1DIndexFromPoint(GoalIndex2D, TerraShiftParams->GridSize)];
    }

    // Reset columns aned set goal locations
    for (int i = 0; i < Columns.Num(); ++i)
    {
        Columns[i]->ResetColumn();
    }

    for (int i = 0; i < TerraShiftParams->NumGoals; ++i)
    {
        FLinearColor GoalColor = GoalColors[i % GoalColors.Num()];
        Columns[Get1DIndexFromPoint(GoalIndices2D[i], TerraShiftParams->GridSize)]->SetColumnColor(GoalColor);
    }

    SetActiveGridObjects(CurrentAgents);  // Activate the grid objects based on the current number of agents

    return State();
}

void ATerraShiftEnvironment::SetActiveGridObjects(int NumAgents)
{
    for (int i = 0; i < MaxAgents; ++i)
    {
        if (i < NumAgents)
        {
            // Get the extent of the object, scaled by TerraShiftParams->ObjectSize
            FVector GridObjectExtent = Objects[i]->GetObjectExtent() * TerraShiftParams->ObjectSize;

            // Use GetComponentBounds to get platform bounds considering scale
            FVector PlatformOrigin, PlatformExtent;
            Platform->GetStaticMeshComponent()->GetLocalBounds(PlatformOrigin, PlatformExtent);
            PlatformExtent *= Platform->GetActorScale3D(); // Scale it to match platform's world size

            FVector PlatformCenter = Platform->GetActorLocation();

            // Compute a random location above and within the platform bounds
            FVector RandomLocation = PlatformCenter + FVector(
                FMath::RandRange(-PlatformExtent.X + GridObjectExtent.X, PlatformExtent.X - GridObjectExtent.X),
                FMath::RandRange(-PlatformExtent.Y + GridObjectExtent.Y, PlatformExtent.Y - GridObjectExtent.Y),
                PlatformCenter.Z + PlatformExtent.Z + GridObjectExtent.Z + (TerraShiftParams->MaxColumnHeight * 2.0) // Ensure it's above the platform
            );

            SetSpawnGridObject(
                i,
                static_cast<float>(i) * TerraShiftParams->SpawnDelay,
                RandomLocation
            );
        }
        else
        {
            Objects[i]->SetGridObjectActive(false); // Deactivate unused agents
        }
    }
}

void ATerraShiftEnvironment::SetSpawnGridObject(int AgentIndex, float Delay, FVector Location)
{
    FTimerHandle TimerHandle;
    GetWorldTimerManager().SetTimer(
        TimerHandle,
        [this, AgentIndex, Location]() {
            Objects[AgentIndex]->SetActorLocationAndActivate(Location);
        },
        Delay,
        false
    );
}

AStaticMeshActor* ATerraShiftEnvironment::SpawnPlatform(FVector Location, FVector Size)
{
    if (UWorld* World = GetWorld())
    {
        UStaticMesh* PlaneMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Plane.Plane"));
        if (PlaneMesh)
        {
            FActorSpawnParameters SpawnParams;
            AStaticMeshActor* NewPlatform = World->SpawnActor<AStaticMeshActor>(AStaticMeshActor::StaticClass(), Location, FRotator::ZeroRotator, SpawnParams);
            if (NewPlatform)
            {
                NewPlatform->GetStaticMeshComponent()->SetStaticMesh(PlaneMesh);
                NewPlatform->GetStaticMeshComponent()->SetWorldScale3D(Size);
                NewPlatform->SetMobility(EComponentMobility::Static);
                NewPlatform->GetStaticMeshComponent()->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
                NewPlatform->GetStaticMeshComponent()->SetCollisionResponseToAllChannels(ECollisionResponse::ECR_Block);
            }
            return NewPlatform;
        }
    }
    return nullptr;
}



int ATerraShiftEnvironment::SelectColumn(int AgentIndex, int Direction) const
{
    if (LastColumnIndexArray[AgentIndex] == INDEX_NONE)
    {
        FVector AgentPosition = Objects[AgentIndex]->GetActorLocation();
        float MinDistance = MAX_FLT;
        int ClosestColumnIndex = INDEX_NONE;

        for (int i = 0; i < Columns.Num(); ++i)
        {
            float Distance = FVector::Dist2D(AgentPosition, Columns[i]->GetActorLocation());
            if (Distance < MinDistance)
            {
                MinDistance = Distance;
                ClosestColumnIndex = i;
            }
        }
        return ClosestColumnIndex;
    }
    else
    {
        int GridSize = TerraShiftParams->GridSize;
        int Row = LastColumnIndexArray[AgentIndex] / GridSize;
        int Col = LastColumnIndexArray[AgentIndex] % GridSize;

        switch (Direction)
        {
        case 0: // N
            Row = FMath::Clamp(Row - 1, 0, GridSize - 1);
            break;
        case 1: // NE
            Row = FMath::Clamp(Row - 1, 0, GridSize - 1);
            Col = FMath::Clamp(Col + 1, 0, GridSize - 1);
            break;
        case 2: // E
            Col = FMath::Clamp(Col + 1, 0, GridSize - 1);
            break;
        case 3: // SE
            Row = FMath::Clamp(Row + 1, 0, GridSize - 1);
            Col = FMath::Clamp(Col + 1, 0, GridSize - 1);
            break;
        case 4: // S
            Row = FMath::Clamp(Row + 1, 0, GridSize - 1);
            break;
        case 5: // SW
            Row = FMath::Clamp(Row + 1, 0, GridSize - 1);
            Col = FMath::Clamp(Col - 1, 0, GridSize - 1);
            break;
        case 6: // W
            Col = FMath::Clamp(Col - 1, 0, GridSize - 1);
            break;
        case 7: // NW
            Row = FMath::Clamp(Row - 1, 0, GridSize - 1);
            Col = FMath::Clamp(Col - 1, 0, GridSize - 1);
            break;
        default:
            break;
        }

        return Row * GridSize + Col;
    }
}

TArray<float> ATerraShiftEnvironment::AgentGetState(int AgentIndex)
{
    TArray<float> State;
    if (LastColumnIndexArray[AgentIndex] != INDEX_NONE)
    {
        FVector ColumnLocation = Columns[LastColumnIndexArray[AgentIndex]]->GetActorLocation();
        State.Add(ColumnLocation.X);
        State.Add(ColumnLocation.Y);
        State.Add(ColumnLocation.Z);
    }

    FVector AgentPosition = Objects[AgentIndex]->GetActorLocation();
    State.Add(AgentPosition.X);
    State.Add(AgentPosition.Y);
    State.Add(AgentPosition.Z);

    return State;
}

int ATerraShiftEnvironment::Get1DIndexFromPoint(const FIntPoint& Point, int GridSize) const
{
    return Point.X * GridSize + Point.Y;
}


void ATerraShiftEnvironment::PostStep()
{
    CurrentStep += 1;
}

FState ATerraShiftEnvironment::State()
{
    FState CurrentState;
    for (int i = 0; i < Objects.Num(); ++i)
    {
        CurrentState.Values += AgentGetState(i);
    }
    return CurrentState;
}

bool ATerraShiftEnvironment::Done()
{
    for (int i = 0; i < Objects.Num(); ++i)
    {
        if (ObjectOffPlatform(i)) {
            return true;
        }
        if (ObjectReachedWrongGoal(i))
        {
            return true;
        }
    }

    return false;
}

bool ATerraShiftEnvironment::Trunc()
{
    if (CurrentStep > TerraShiftParams->MaxSteps)
    {
        CurrentStep = 0;
        return true;
    }

    return false;
}

float ATerraShiftEnvironment::Reward()
{
    float TotalRewards = 0.0f;

    for (int i = 0; i < Objects.Num(); ++i)
    {
        FVector ObjectPosition = Objects[i]->GetActorLocation();
        FVector GoalPosition = GoalPositionArray[AgentGoalIndices[i]];

        float DistanceToGoal = FVector::Dist(ObjectPosition, GoalPosition);

        // Provide a reward based on the inverse of the distance to the goal, with small epsilon to avoid division by zero
        float RewardForAgent = 1.0f / (DistanceToGoal + 1e-5);

        TotalRewards += RewardForAgent;
    }

    return TotalRewards;
}

void ATerraShiftEnvironment::PostTransition()
{
    // Handle any cleanup or state transitions post-action
}

float ATerraShiftEnvironment::Map(float x, float in_min, float in_max, float out_min, float out_max) const
{
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

bool ATerraShiftEnvironment::ObjectReachedWrongGoal(int AgentIndex) const
{
    if (!Objects.IsValidIndex(AgentIndex) || !Objects[AgentIndex])
    {
        // If the object is off the grid or invalid, skip further checks
        return false;
    }

    FVector ObjectPosition = Objects[AgentIndex]->GetActorLocation();
    FVector ObjectExtent = Objects[AgentIndex]->GetObjectExtent();

    if (ObjectExtent.IsZero())
    {
        // If the object doesn't have a valid extent, return false
        return false;
    }

    for (int i = 0; i < GoalPositionArray.Num(); ++i)
    {
        // Skip checking the correct goal for this agent
        if (i == AgentGoalIndices[AgentIndex])
        {
            continue;
        }

        FVector GoalPosition = GoalPositionArray[i];

        // Check if the object is within the wrong goal area
        if (ObjectPosition.X + ObjectExtent.X > GoalPosition.X - ObjectExtent.X &&
            ObjectPosition.X - ObjectExtent.X < GoalPosition.X + ObjectExtent.X &&
            ObjectPosition.Y + ObjectExtent.Y > GoalPosition.Y - ObjectExtent.Y &&
            ObjectPosition.Y - ObjectExtent.Y < GoalPosition.Y + ObjectExtent.Y)
        {
            return true;
        }
    }

    return false;
}

void ATerraShiftEnvironment::Act(FAction Action)
{
    // Ensure the action array has the correct size for the current number of agents
    if (Action.Values.Num() != Objects.Num() * 2) {
        UE_LOG(LogTemp, Error, TEXT("Action array size mismatch. Expected %d, got %d"), Objects.Num() * 2, Action.Values.Num());
        return;
    }

    for (int i = 0; i < Objects.Num(); ++i)
    {
        // Extract direction and height action for each agent
        int Direction = FMath::RoundToInt(Action.Values[i * 2]);
        int HeightAction = FMath::RoundToInt(Action.Values[i * 2 + 1]);

        // Select the appropriate column for the agent to influence
        int SelectedColumnIndex = SelectColumn(i, Direction);
        LastColumnIndexArray[i] = SelectedColumnIndex; // Update the last selected column index

        // Apply the action on the selected column through the Column class
        switch (HeightAction)
        {
        case 0: // accelerate
            Columns[SelectedColumnIndex]->SetColumnAcceleration(TerraShiftParams->ColumnAccelConstant);
            break;
        case 1: // decelerate
            Columns[SelectedColumnIndex]->SetColumnAcceleration(-TerraShiftParams->ColumnAccelConstant);
            break;
        case 2: // noop (no operation)
            break;
        default:
            UE_LOG(LogTemp, Warning, TEXT("Invalid HeightAction value"));
            break;
        }
    }
}

bool ATerraShiftEnvironment::ObjectOffPlatform(int AgentIndex) const
{
    if (!Objects.IsValidIndex(AgentIndex) || !Objects[AgentIndex])
    {
        // If the object index is invalid or the object is null, consider it off the platform
        return true;
    }

    FVector ObjectPosition = Objects[AgentIndex]->GetActorLocation();
    FVector PlatformExtent = Platform->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().BoxExtent * Platform->GetActorScale3D();
    FVector PlatformCenter = Platform->GetActorLocation();

    // Check if the object is within the bounds of the platform
    if (ObjectPosition.X < PlatformCenter.X - PlatformExtent.X ||
        ObjectPosition.X > PlatformCenter.X + PlatformExtent.X ||
        ObjectPosition.Y < PlatformCenter.Y - PlatformExtent.Y ||
        ObjectPosition.Y > PlatformCenter.Y + PlatformExtent.Y)
    {
        return true;
    }

    return false;
}

