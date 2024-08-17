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
    LastColumnIndex = INDEX_NONE;

    // Set TerraShiftRoot at the specified location.
    TerraShiftRoot->SetWorldLocation(TerraShiftParams->Location);
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
    AStaticMeshActor* Column = nullptr;

    for (int i = 0; i < TerraShiftParams->GridSize; ++i)
    {
        for (int j = 0; j < TerraShiftParams->GridSize; ++j)
        {
            // Spawn each column.
            Column = SpawnColumn(
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
            GridCenterPoints[Get1DIndexFromPoint(FIntPoint(i, j), TerraShiftParams->GridSize)] = GridCenter;

            SetColumnHeight(Get1DIndexFromPoint(FIntPoint(i, j), TerraShiftParams->GridSize), 0.5f);
        }
    }

    // Set the single agent observation size
    EnvInfo.SingleAgentObsSize = 6; // Current column location (3) + gridObject location (3)

    // Set the state size and action space
    EnvInfo.StateSize = CurrentAgents * EnvInfo.SingleAgentObsSize;

    TArray<FDiscreteActionSpec> DiscreteActions;
    DiscreteActions.Add({ 8 }); // columnChangeDirection: 8 directions (N, NE, E, SE, S, SW, W, NW)
    DiscreteActions.Add({ 3 }); // columnHeight: 3 options (accelerate, decelerate, noop)

    if (EnvInfo.ActionSpace)
    {
        EnvInfo.ActionSpace->Init({}, DiscreteActions);
    }

    // Initialize agents (GridObjects)
    for (int i = 0; i < CurrentAgents; ++i)
    {
        Objects.Add(InitializeGridObject());
    }
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

                UMaterial* Material = LoadObject<UMaterial>(nullptr, TEXT("/Game/Material/Column_Material.Column_Material"));
                if (Material)
                {
                    ColumnActor->GetStaticMeshComponent()->SetMaterial(0, Material);
                }

                ColumnActor->GetStaticMeshComponent()->SetCollisionEnabled(ECollisionEnabled::QueryOnly);
                ColumnActor->GetStaticMeshComponent()->SetMobility(EComponentMobility::Movable);
                ColumnActor->GetStaticMeshComponent()->BodyInstance.SetMassOverride(TerraShiftParams->ColumnMass, true);
                ColumnActor->GetStaticMeshComponent()->SetEnableGravity(false); // Disable gravity to prevent columns from falling
                ColumnActor->GetStaticMeshComponent()->SetSimulatePhysics(false); // Disable physics simulation to prevent columns from being pushed around
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
        int32 RandomIndex = FMath::RandRange(0, GridCenterPoints.Num() - 1);
        FVector Location = GridCenterPoints[RandomIndex] + FVector(0, 0, 10.0f);
        UStaticMesh* SphereMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Sphere.Sphere"));
        AStaticMeshActor* ObjectActor = nullptr;
        if (SphereMesh)
        {
            FActorSpawnParameters SpawnParams;
            ObjectActor = World->SpawnActor<AStaticMeshActor>(AStaticMeshActor::StaticClass(), Location, FRotator::ZeroRotator, SpawnParams);
            if (ObjectActor)
            {
                ObjectActor->Tags.Add(FName("Ball")); // Tagging the object as 'Ball'
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

int ATerraShiftEnvironment::SelectColumn(int AgentIndex, int Direction) const {
    if (LastColumnIndex == INDEX_NONE)
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
        int Row = LastColumnIndex / GridSize;
        int Col = LastColumnIndex % GridSize;

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

    if (LastColumnIndex != INDEX_NONE)
    {
        FVector ColumnLocation = Columns[LastColumnIndex]->GetActorLocation();
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

FState ATerraShiftEnvironment::ResetEnv(int NumAgents)
{
    CurrentStep = 0;
    GoalPosition = FVector(FMath::RandRange(-1.0f, 1.0f), FMath::RandRange(-1.0f, 1.0f), 0.0f);

    for (int i = 0; i < Columns.Num(); ++i)
    {
        SetColumnHeight(i, 0.5f);
        ColumnVelocities[i] = 0.0f; // Reset column velocities on reset
    }

    for (AStaticMeshActor* Object : Objects)
    {
        int32 RandomIndex = FMath::RandRange(0, GridCenterPoints.Num() - 1);
        FVector NewLocation = GridCenterPoints[RandomIndex] + FVector(0, 0, 10.0f);
        Object->SetActorLocation(NewLocation);
    }

    return State();
}

void ATerraShiftEnvironment::Act(FAction Action)
{
    for (int i = 0; i < Objects.Num(); ++i)
    {
        int Direction = FMath::RoundToInt(Action.Values[i * 2]);
        int HeightAction = FMath::RoundToInt(Action.Values[i * 2 + 1]);

        int SelectedColumnIndex = SelectColumn(i, Direction);

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

        LastColumnIndex = SelectedColumnIndex;
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
    return false; // Implement any termination condition if required
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

    // TODO: Calculate rewards based on environment conditions
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