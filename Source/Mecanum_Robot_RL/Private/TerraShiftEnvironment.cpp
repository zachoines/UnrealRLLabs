#include "TerraShiftEnvironment.h"

ATerraShiftEnvironment::ATerraShiftEnvironment()
{
    // Setup Env Info
    EnvInfo.EnvID = 3;
    EnvInfo.IsMultiAgent = true;

    // Initialize Scene components
    TerraShiftRoot = CreateDefaultSubobject<USceneComponent>(TEXT("TerraShiftRoot"));
    RootComponent = TerraShiftRoot;

    // Create a default sub-object for ActionSpace
    EnvInfo.ActionSpace = CreateDefaultSubobject<UActionSpace>(TEXT("ActionSpace"));
}

void ATerraShiftEnvironment::InitEnv(FBaseInitParams* BaseParams)
{
    TerraShiftParams = static_cast<FTerraShiftEnvironmentInitParams*>(BaseParams);

    // Set the number of current agents from the initialization parameters
    CurrentAgents = TerraShiftParams->NumAgents;
    CurrentStep = 0;

    // Set TerraShiftRoot at the specified location.
    TerraShiftRoot->SetWorldLocation(TerraShiftParams->Location);

    // Spawn the platform
    Platform = SpawnPlatform(
        TerraShiftParams->Location,
        FVector(TerraShiftParams->GroundPlaneSize, TerraShiftParams->GroundPlaneSize, TerraShiftParams->GroundPlaneSize)
    );

    // Prepare for column spawning
    Columns.SetNum(TerraShiftParams->GridSize * TerraShiftParams->GridSize);
    GridCenterPoints.SetNum(TerraShiftParams->GridSize * TerraShiftParams->GridSize);

    // Initialize column velocities to zero
    ColumnVelocities.SetNum(TerraShiftParams->GridSize * TerraShiftParams->GridSize);
    for (int i = 0; i < ColumnVelocities.Num(); ++i) {
        ColumnVelocities[i] = 0.0f;
    }

    FVector PlatformWorldSize = Platform->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().BoxExtent * 2.0f * TerraShiftParams->GroundPlaneSize;
    FVector PlatformCenter = Platform->GetActorLocation();

    float CellWidth = (TerraShiftParams->GroundPlaneSize / static_cast<float>(TerraShiftParams->GridSize)) - 1e-2;
    float CellHeight = CellWidth;

    for (int i = 0; i < TerraShiftParams->GridSize; ++i)
    {
        for (int j = 0; j < TerraShiftParams->GridSize; ++j)
        {
            // Spawn each column.
            AStaticMeshActor* Column = SpawnColumn(
                FVector(CellWidth, CellWidth, CellHeight),
                *FString::Printf(TEXT("ColumnMesh_%d_%d"), i, j)
            );

            FVector ColumnWorldSize = Column->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().BoxExtent * 2.0f * CellHeight;
            FVector GridCenter = PlatformCenter + FVector(
                (i - TerraShiftParams->GridSize / 2.0f + 0.5f) * (PlatformWorldSize.X / TerraShiftParams->GridSize),
                (j - TerraShiftParams->GridSize / 2.0f + 0.5f) * (PlatformWorldSize.Y / TerraShiftParams->GridSize),
                ColumnWorldSize.Z / 2
            );
            Column->SetActorLocation(GridCenter);

            Columns[Get1DIndexFromPoint(FIntPoint(i, j), TerraShiftParams->GridSize)] = Column;
            SetColumnColor(Get1DIndexFromPoint(FIntPoint(i, j), TerraShiftParams->GridSize), FLinearColor(FLinearColor::White));
            GridCenterPoints[Get1DIndexFromPoint(FIntPoint(i, j), TerraShiftParams->GridSize)] = GridCenter;

            SetColumnHeight(Get1DIndexFromPoint(FIntPoint(i, j), TerraShiftParams->GridSize), 0.5f);
        }
    }

    // Spawn the chute at a location above the tallest possible column
    FVector PlatformExtent = Platform->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().BoxExtent * Platform->GetActorScale3D();
    FVector ChuteLocation = Platform->GetActorLocation() + FVector(0, 0, PlatformExtent.Z + TerraShiftParams->MaxColumnHeight + TerraShiftParams->ChuteHeight + 4 * TerraShiftParams->MaxColumnHeight);
    Chute = SpawnChute(ChuteLocation);

    // Set the single agent observation size
    EnvInfo.SingleAgentObsSize = 6;

    // Set the state size and action space
    EnvInfo.StateSize = CurrentAgents * EnvInfo.SingleAgentObsSize;

    TArray<FDiscreteActionSpec> DiscreteActions;
    DiscreteActions.Add({ 8 }); // columnChangeDirection: 8 directions
    DiscreteActions.Add({ 3 }); // columnHeight: 3 options

    if (EnvInfo.ActionSpace)
    {
        EnvInfo.ActionSpace->Init({}, DiscreteActions);
    }

    // Initialize agents (GridObjects) 
    for (int i = 0; i < CurrentAgents; ++i)
    {
        Objects.Add(InitializeGridObject());
    }

    // Initialize arrays for LastColumnIndex and AgentGoalIndices
    LastColumnIndexArray.SetNum(CurrentAgents);
    AgentGoalIndices.SetNum(CurrentAgents);
}

FState ATerraShiftEnvironment::ResetEnv(int NumAgents)
{
    CurrentStep = 0;
    CurrentAgents = NumAgents;

    // Handle potential change in the number of agents and their associated data
    if (Objects.Num() != NumAgents) {
        // Destroy existing objects if there are more than needed
        while (Objects.Num() > NumAgents) {
            AStaticMeshActor* LastObject = Objects.Pop();
            if (LastObject) {
                LastObject->Destroy();
            }
        }

        // Spawn new objects if there are fewer than needed
        while (Objects.Num() < NumAgents) {
            Objects.Add(InitializeGridObject());
        }

        // Resize arrays for LastColumnIndex and GoalPosition
        LastColumnIndexArray.SetNum(NumAgents);
        AgentGoalIndices.SetNum(NumAgents);
    }

    // Reset columns
    for (int i = 0; i < Columns.Num(); ++i)
    {
        SetColumnHeight(i, 0.5f);
        SetColumnColor(i, FLinearColor(0.0f, 0.0f, 0.0f));
        ColumnVelocities[i] = 0.0f;
    }

    // Calculate CellWidth 
    float CellWidth = (TerraShiftParams->GroundPlaneSize / static_cast<float>(TerraShiftParams->GridSize)) - 1e-2;
    int GridSize = TerraShiftParams->GridSize;

    // Reset goal positions to be on the edges of the grid
    GoalPositionArray.SetNum(TerraShiftParams->NumGoals); // Set the number of goals based on config
    TArray<FIntPoint> GoalIndices2D;
    GoalIndices2D.SetNum(TerraShiftParams->NumGoals);

    // Select distinct goal positions on the edges
    for (int i = 0; i < TerraShiftParams->NumGoals; ++i) {
        int Side;
        FIntPoint GoalIndex2D;
        do {
            Side = FMath::RandRange(0, 3);
            switch (Side)
            {
            case 0: // Top
                GoalIndex2D = FIntPoint(FMath::RandRange(0, GridSize - 1), 0);
                break;
            case 1: // Bottom
                GoalIndex2D = FIntPoint(FMath::RandRange(0, GridSize - 1), GridSize - 1);
                break;
            case 2: // Left
                GoalIndex2D = FIntPoint(0, FMath::RandRange(0, GridSize - 1));
                break;
            case 3: // Right
                GoalIndex2D = FIntPoint(GridSize - 1, FMath::RandRange(0, GridSize - 1));
                break;
            }
        } while (GoalIndices2D.Contains(GoalIndex2D)); // Ensure the goal is unique

        GoalIndices2D[i] = GoalIndex2D;
        GoalPositionArray[i] = GridCenterPoints[Get1DIndexFromPoint(GoalIndex2D, GridSize)];
    }

    // Color the goal columns
    for (int i = 0; i < TerraShiftParams->NumGoals; ++i) {
        // Cycle through the GoalColors array
        FLinearColor GoalColor = GoalColors[i % GoalColors.Num()];
        SetColumnColor(Get1DIndexFromPoint(GoalIndices2D[i], GridSize), GoalColor);
    }

    // Reset the positions of all objects and randomly assign goals
    for (int i = 0; i < Objects.Num(); ++i)
    {
        Objects[i]->SetActorLocation(Chute->GetActorLocation());

        // Randomly assign a goal to this agent
        AgentGoalIndices[i] = FMath::RandRange(0, TerraShiftParams->NumGoals - 1);

        // Reset LastColumnIndex for this agent
        LastColumnIndexArray[i] = INDEX_NONE;
    }

    // Randomly change the position of the chute within platform bounds
    if (Chute) {
        // Calculate a new random location within the platform bounds
        FVector PlatformCenter = Platform->GetActorLocation();
        FVector PlatformExtent = Platform->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().BoxExtent * Platform->GetActorScale3D();
        FVector NewChuteLocation = PlatformCenter + FVector(
            FMath::RandRange(-PlatformExtent.X + TerraShiftParams->ChuteRadius, PlatformExtent.X - TerraShiftParams->ChuteRadius),
            FMath::RandRange(-PlatformExtent.Y + TerraShiftParams->ChuteRadius, PlatformExtent.Y - TerraShiftParams->ChuteRadius),
            TerraShiftParams->ChuteHeight + 4 * TerraShiftParams->MaxColumnHeight // Position above the platform
        );

        Chute->SetActorLocation(NewChuteLocation);
    }

    return State();
}

AStaticMeshActor* ATerraShiftEnvironment::SpawnColumn(FVector Dimensions, FName Name)
{
    if (UWorld* World = GetWorld())
    {
        UStaticMesh* ColumnMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));
        AStaticMeshActor* ColumnActor = nullptr;
        if (ColumnMesh)
        {
            FActorSpawnParameters SpawnParams;
            ColumnActor = World->SpawnActor<AStaticMeshActor>(AStaticMeshActor::StaticClass(), FVector::ZeroVector, FRotator::ZeroRotator, SpawnParams);
            if (ColumnActor)
            {
                ColumnActor->GetStaticMeshComponent()->SetStaticMesh(ColumnMesh);
                ColumnActor->GetStaticMeshComponent()->SetWorldScale3D(Dimensions);

                /*UMaterial* Material = LoadObject<UMaterial>(nullptr, TEXT("/Game/Material/Column_Material.Column_Material"));
                if (Material)
                {
                    ColumnActor->GetStaticMeshComponent()->SetMaterial(0, Material);
                }*/

                ColumnActor->GetStaticMeshComponent()->SetCollisionEnabled(ECollisionEnabled::QueryOnly);
                ColumnActor->GetStaticMeshComponent()->SetMobility(EComponentMobility::Movable);
                ColumnActor->GetStaticMeshComponent()->BodyInstance.SetMassOverride(TerraShiftParams->ColumnMass, true);
                ColumnActor->GetStaticMeshComponent()->SetEnableGravity(false);
                ColumnActor->GetStaticMeshComponent()->SetSimulatePhysics(false);
            }
        }
        return ColumnActor;
    }
    return nullptr;
}

AStaticMeshActor* ATerraShiftEnvironment::SpawnPlatform(FVector Location, FVector Size)
{
    if (UWorld* World = GetWorld())
    {
        UStaticMesh* PlaneMesh = LoadObject<UStaticMesh>(nullptr, TEXT("StaticMesh'/Engine/BasicShapes/Plane.Plane'"));
        if (PlaneMesh)
        {
            FActorSpawnParameters SpawnParams;
            Platform = World->SpawnActor<AStaticMeshActor>(AStaticMeshActor::StaticClass(), Location, FRotator::ZeroRotator, SpawnParams);
            if (Platform)
            {
                Platform->GetStaticMeshComponent()->SetStaticMesh(PlaneMesh);
                Platform->GetStaticMeshComponent()->SetWorldScale3D(Size);
                Platform->SetMobility(EComponentMobility::Static);

                UMaterial* Material = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/Material/Platform_Material.Platform_Material'"));
                if (Material)
                {
                    Material->TwoSided = true;
                    Platform->GetStaticMeshComponent()->SetMaterial(0, Material);
                }
                Platform->GetStaticMeshComponent()->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
                Platform->GetStaticMeshComponent()->SetCollisionResponseToAllChannels(ECollisionResponse::ECR_Block);
            }
        }
        return Platform;
    }
    return nullptr;
}

AStaticMeshActor* ATerraShiftEnvironment::SpawnChute(FVector Location)
{
    if (UWorld* World = GetWorld())
    {
        UStaticMesh* CylinderMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cylinder.Cylinder"));
        if (CylinderMesh)
        {
            FActorSpawnParameters SpawnParams;
            Chute = World->SpawnActor<AStaticMeshActor>(AStaticMeshActor::StaticClass(), Location, FRotator(0, 0, 180), SpawnParams); // 180-degree rotation to face downwards
            if (Chute)
            {
                Chute->GetStaticMeshComponent()->SetStaticMesh(CylinderMesh);
                Chute->GetStaticMeshComponent()->SetWorldScale3D(FVector(TerraShiftParams->ChuteRadius, TerraShiftParams->ChuteRadius, TerraShiftParams->ChuteHeight / 2)); // Adjust height scaling
                Chute->SetMobility(EComponentMobility::Movable);
                Chute->GetStaticMeshComponent()->SetEnableGravity(false);
            }
        }
        return Chute;
    }
    return nullptr;
}

void ATerraShiftEnvironment::SetColumnHeight(int ColumnIndex, float NewHeight)
{
    AStaticMeshActor* Column = Columns[ColumnIndex];
    if (Column)
    {
        FVector CurrentLocation = Column->GetActorLocation();
        CurrentLocation.Z = FMath::Clamp(NewHeight * TerraShiftParams->MaxColumnHeight, 0.0f, TerraShiftParams->MaxColumnHeight);
        Column->SetActorLocation(CurrentLocation);
    }
}

void ATerraShiftEnvironment::SetColumnAcceleration(int ColumnIndex, float Acceleration) {
    AStaticMeshActor* Column = Columns[ColumnIndex];
    if (Column) {
        float MaxHeight = TerraShiftParams->MaxColumnHeight;
        float MinHeight = 0.0f;

        // Update the column's velocity based on the acceleration
        ColumnVelocities[ColumnIndex] += Acceleration;

        // Clamp the velocity if the column is at its limits
        FVector CurrentLocation = Column->GetActorLocation();
        float CurrentHeight = CurrentLocation.Z;
        if ((CurrentHeight >= MaxHeight && ColumnVelocities[ColumnIndex] > 0.0f) ||
            (CurrentHeight <= MinHeight && ColumnVelocities[ColumnIndex] < 0.0f))
        {
            ColumnVelocities[ColumnIndex] = 0.0f;
        }
    }
}

void ATerraShiftEnvironment::Tick(float DeltaTime) {
    Super::Tick(DeltaTime);

    // Update column positions based on their velocities
    for (int i = 0; i < Columns.Num(); ++i) {
        AStaticMeshActor* Column = Columns[i];
        if (Column && ColumnVelocities[i] != 0.0f) {
            FVector NewLocation = Column->GetActorLocation();
            NewLocation.Z += ColumnVelocities[i] * DeltaTime;
            NewLocation.Z = FMath::Clamp(NewLocation.Z, 0.0f, TerraShiftParams->MaxColumnHeight);
            Column->SetActorLocation(NewLocation);

            // If the column reaches its limit, set its velocity to zero
            if (NewLocation.Z == 0.0f || NewLocation.Z == TerraShiftParams->MaxColumnHeight) {
                ColumnVelocities[i] = 0.0f;
            }
        }
    }
}

AStaticMeshActor* ATerraShiftEnvironment::InitializeGridObject()
{
    if (UWorld* World = GetWorld())
    {
        // Spawn the object slightly above the chute's opening
        FVector Location = Chute->GetActorLocation() - FVector(0, 0, TerraShiftParams->ChuteHeight / 2 + TerraShiftParams->ObjectSize.Z / 2 + 0.1f);

        UStaticMesh* SphereMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Sphere.Sphere"));
        AStaticMeshActor* ObjectActor = nullptr;
        if (SphereMesh)
        {
            FActorSpawnParameters SpawnParams;
            ObjectActor = World->SpawnActor<AStaticMeshActor>(AStaticMeshActor::StaticClass(), Location, FRotator::ZeroRotator, SpawnParams);
            if (ObjectActor)
            {
                ObjectActor->Tags.Add(FName("Ball"));
                ObjectActor->GetStaticMeshComponent()->SetStaticMesh(SphereMesh);
                ObjectActor->GetStaticMeshComponent()->SetWorldScale3D(TerraShiftParams->ObjectSize);

                UMaterial* Material = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/Material/Manip_Object_Material.Manip_Object_Material'"));
                if (Material)
                {
                    ObjectActor->GetStaticMeshComponent()->SetMaterial(0, Material);
                }

                ObjectActor->GetStaticMeshComponent()->SetMobility(EComponentMobility::Movable);
                ObjectActor->GetStaticMeshComponent()->BodyInstance.SetMassOverride(TerraShiftParams->ObjectMass, true);
                ObjectActor->GetStaticMeshComponent()->SetEnableGravity(true);
                ObjectActor->GetStaticMeshComponent()->SetSimulatePhysics(true);
            }
            return ObjectActor;
        }
    }
    return nullptr;
}

int ATerraShiftEnvironment::SelectColumn(int AgentIndex, int Direction) const
{
    if (LastColumnIndexArray[AgentIndex] == INDEX_NONE)
    {
        // First call after InitEnv or ResetEnv, find the closest column to the current grid object
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
        // Determine the next column based on the direction from the last selected column
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

int ATerraShiftEnvironment::Get1DIndexFromPoint(const FIntPoint& Point, int gridSize) const
{
    return Point.X * gridSize + Point.Y;
}

float ATerraShiftEnvironment::GridDistance(const FIntPoint& Point1, const FIntPoint& Point2) const
{
    return FMath::Sqrt(FMath::Square(static_cast<float>(Point2.X) - static_cast<float>(Point1.X)) + FMath::Square(static_cast<float>(Point2.Y) - static_cast<float>(Point1.Y)));
}

bool ATerraShiftEnvironment::ObjectOffPlatform(int AgentIndex) const
{
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

bool ATerraShiftEnvironment::ObjectReachedWrongGoal(int AgentIndex) const
{
    FVector ObjectPosition = Objects[AgentIndex]->GetActorLocation();
    FVector ObjectExtent = Objects[AgentIndex]->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().BoxExtent * Objects[AgentIndex]->GetActorScale3D();

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
        // Corrected action indexing
        int Direction = FMath::RoundToInt(Action.Values[i * 2]);
        int HeightAction = FMath::RoundToInt(Action.Values[i * 2 + 1]);

        int SelectedColumnIndex = SelectColumn(i, Direction);
        LastColumnIndexArray[i] = SelectedColumnIndex; // Update LastColumnIndex for this agent

        switch (HeightAction)
        {
        case 0: // accelerate
            SetColumnAcceleration(SelectedColumnIndex, TerraShiftParams->ColumnAccelConstant);
            break;
        case 1: // decelerate
            SetColumnAcceleration(SelectedColumnIndex, -TerraShiftParams->ColumnAccelConstant);
            break;
        case 2: // noop
            break;
        }
    }
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
        if (ObjectOffPlatform(i))
        {
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
        // Get the goal position for this agent based on its assigned goal index
        FVector GoalPosition = GoalPositionArray[AgentGoalIndices[i]];

        // Calculate the distance to the goal
        float DistanceToGoal = FVector::Dist(ObjectPosition, GoalPosition);

        // Provide a reward based on the inverse of the distance 
        // (closer to the goal = higher reward)
        float RewardForAgent = 1.0f / (DistanceToGoal + 1e-5); // Add a small epsilon to avoid division by zero

        // You can adjust the reward scaling or add other factors as needed
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

void ATerraShiftEnvironment::SetColumnColor(int ColumnIndex, FLinearColor Color)
{
    AStaticMeshActor* Column = Columns[ColumnIndex];
    if (Column)
    {
        UMaterial* BaseMaterial = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/StarterContent/Materials/M_Basic_Floor.M_Basic_Floor'"));
        Column->GetStaticMeshComponent()->SetMaterial(0, BaseMaterial);

        // Create a dynamic material instance to set the color
        UMaterialInstanceDynamic* DynMaterial = UMaterialInstanceDynamic::Create(Column->GetStaticMeshComponent()->GetMaterial(0), Column);
        DynMaterial->SetVectorParameterValue("Color", Color);
        Column->GetStaticMeshComponent()->SetMaterial(0, DynMaterial);
    }
}