// Fill out your copyright notice in the Description page of Project Settings.


#include "TerraShiftEnvironment.h"


ATerraShiftEnvironment::ATerraShiftEnvironment()
{
    // Setup Env Info
    EnvInfo.EnvID = 3;
    EnvInfo.MaxAgents = MaxAgents;
    EnvInfo.SingleAgentObsSize = 3;
    EnvInfo.StateSize = EnvInfo.MaxAgents * EnvInfo.SingleAgentObsSize;
    EnvInfo.IsMultiAgent = true;

    const TArray<FContinuousActionSpec>& ContinuousActions = { {-1.0, 1.0} };
    const TArray<FDiscreteActionSpec>& DiscreteActions = { };

    // Create a default sub-object for ActionSpace
    EnvInfo.ActionSpace = CreateDefaultSubobject<UActionSpace>(TEXT("ActionSpace"));
    if (EnvInfo.ActionSpace)
    {
        EnvInfo.ActionSpace->Init(ContinuousActions, DiscreteActions);
    }

    // Initialize Scene components
    TerraShiftRoot = CreateDefaultSubobject<USceneComponent>(TEXT("TerraShiftRoot"));
    RootComponent = TerraShiftRoot;
}

void ATerraShiftEnvironment::InitEnv(FBaseInitParams* BaseParams)
{
    TerraShiftParams = static_cast<FTerraShiftEnvironmentInitParams*>(BaseParams);

    // Set TerraShiftRoot at the specified location.
    TerraShiftRoot->SetWorldLocation(TerraShiftParams->Location);
    Platform = SpawnPlatform(
        TerraShiftParams->Location,
        FVector(TerraShiftParams->GroundPlaneSize, TerraShiftParams->GroundPlaneSize, TerraShiftParams->GroundPlaneSize)
    );

    // Prepare for column spawning and prismatic joint attachment.
    Columns.SetNum(GridSize * GridSize);
    PrismaticJoints.SetNum(GridSize * GridSize);
    GridCenterPoints.SetNum(GridSize * GridSize);

    // Calculate the actual size of world objects
    FVector PlatformWorldSize = Platform->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().BoxExtent * 2.0f * TerraShiftParams->GroundPlaneSize;
    FVector PlatformCenter = Platform->GetActorLocation();

    float CellWidth = (TerraShiftParams->GroundPlaneSize / static_cast<float>(GridSize)) - 1e-2;
    float CellHeight = CellWidth;
    AStaticMeshActor* Column = nullptr;

    for (int i = 0; i < GridSize; ++i)
    {

        for (int j = 0; j < GridSize; ++j)
        {
            // Spawn each column.
            Column = SpawnColumn(
                FVector(CellWidth, CellWidth, CellHeight),
                *FString::Printf(TEXT("ColumnMesh_%d_%d"), i, j)
            );

            FVector ColumnWorldSize = Column->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().BoxExtent * 2.0f * CellHeight;
            FVector GridCenter = PlatformCenter + FVector(
                (i - GridSize / 2.0f + 0.5f) * (PlatformWorldSize.X / GridSize),
                (j - GridSize / 2.0f + 0.5f) * (PlatformWorldSize.Y / GridSize),
                ColumnWorldSize.Z / 2
            );
            Column->SetActorLocation(GridCenter);

            PrismaticJoints[Get1DIndexFromPoint(FIntPoint(i, j), GridSize)] = AttachPrismaticJoint(Column);
            Columns[Get1DIndexFromPoint(FIntPoint(i, j), GridSize)] = Column;
            GridCenterPoints[Get1DIndexFromPoint(FIntPoint(i, j), GridSize)] = GridCenter;

            SetColumnHeight(Get1DIndexFromPoint(FIntPoint(i, j), GridSize), 0.5);
        }
    }

    // Assuming ColumnHeight is adjusted for scale.
    FVector Bounds = Column->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().BoxExtent;
    ScaledHeight = Bounds.Z * Column->GetActorScale3D().Z;
    ScaledWidth = Bounds.X * Column->GetActorScale3D().X;

    Objects.Add(InitializeGridObject());
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

                ColumnActor->GetStaticMeshComponent()->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
                ColumnActor->GetStaticMeshComponent()->SetMobility(EComponentMobility::Movable);
                ColumnActor->GetStaticMeshComponent()->BodyInstance.SetMassOverride(TerraShiftParams->ColumnMass, true);
                ColumnActor->GetStaticMeshComponent()->SetEnableGravity(true);
                ColumnActor->GetStaticMeshComponent()->SetSimulatePhysics(true);
            }
        }
        return ColumnActor;
    }
    return nullptr;
}

UPhysicsConstraintComponent* ATerraShiftEnvironment::AttachPrismaticJoint(AStaticMeshActor* Column)
{
    UPhysicsConstraintComponent* PrismaticJoint = NewObject<UPhysicsConstraintComponent>(Column);
    if (PrismaticJoint)
    {
        // Setting up linear and angular limits
        PrismaticJoint->SetLinearXLimit(ELinearConstraintMotion::LCM_Locked, 0);
        PrismaticJoint->SetLinearYLimit(ELinearConstraintMotion::LCM_Locked, 0);
        PrismaticJoint->SetLinearZLimit(ELinearConstraintMotion::LCM_Limited, TerraShiftParams->MaxColumnHeight);

        PrismaticJoint->SetAngularSwing1Limit(EAngularConstraintMotion::ACM_Locked, 0);
        PrismaticJoint->SetAngularSwing2Limit(EAngularConstraintMotion::ACM_Locked, 0);
        PrismaticJoint->SetAngularTwistLimit(EAngularConstraintMotion::ACM_Locked, 0);

        // Setting the components to be constrained
        PrismaticJoint->SetConstrainedComponents(Column->GetStaticMeshComponent(), NAME_None, Platform->GetStaticMeshComponent(), NAME_None);

        // Initializing the component constraint
        PrismaticJoint->InitComponentConstraint();
        PrismaticJoint->RegisterComponent();

        // Activate the constraint
        PrismaticJoint->SetActive(true);

        // Setting up the linear drive
        PrismaticJoint->SetLinearVelocityDrive(false, false, true);
        if (TerraShiftParams->PositionalDrive) {
            PrismaticJoint->SetLinearPositionDrive(false, false, true);
            PrismaticJoint->SetLinearDriveParams(
                50000.0, // spring "position strength"
                10000.0, // damping "velocity strength"
                0.0 // no limit 
            );
        }
        else {
            PrismaticJoint->SetLinearDriveParams(
                10000.0, // spring "position strength"
                50000.0, // damping "velocity strength"
                0.0 // no limit 
            );
        }
    }

    return PrismaticJoint;
}

AStaticMeshActor* ATerraShiftEnvironment::SpawnPlatform(FVector Location, FVector Size)
{
    if (UWorld* World = GetWorld())
    {
        UStaticMesh* PlaneMesh = LoadObject<UStaticMesh>(nullptr, TEXT("StaticMesh'/Engine/BasicShapes/Plane.Plane'"));
        if (PlaneMesh)
        {
            FActorSpawnParameters SpawnParams;
            // SpawnParams.Name = TEXT("GroundPlane");
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
                Platform->GetStaticMeshComponent()->SetMobility(EComponentMobility::Static);
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
    FVector MinLoc = GridCenterPoints[ColumnIndex];
    UPhysicsConstraintComponent* ConstraintComponent = PrismaticJoints[ColumnIndex];

    if (Column && ConstraintComponent)
    {
        // FVector CurrentLocation = Column->GetActorLocation();
        /*if (CurrentLocation.Z < MinLoc.Z)
        {
            ConstraintComponent->SetLinearPositionTarget(FVector(0, 0, 0));
            Column->TeleportTo(MinLoc, Column->GetActorRotation(), false, true);
        }*/
        
        FVector Scale = Column->GetActorScale3D();
        float AdjustedHeight = (NewHeight * Scale.Z) * TerraShiftParams->MaxColumnHeight;

        ConstraintComponent->SetLinearPositionTarget(FVector(0, 0, AdjustedHeight));
        ConstraintComponent->SetLinearPositionDrive(false, false, true);

        FVector CurrentLocation = Column->GetActorLocation();
        float CurrentZ = CurrentLocation.Z;
        float TargetZ = Column->GetRootComponent()->GetComponentLocation().Z + AdjustedHeight; // Assuming base position + height.

        // Determine velocity direction.
        float Difference = TargetZ - CurrentZ;
        float VelocityDirection = FMath::Sign(Difference);
        float VelocityMagnitude = FMath::Abs(Difference) > KINDA_SMALL_NUMBER ? TerraShiftParams->ColumnVelocity : 0; // Apply velocity only if there's a significant difference.

        // Set linear velocity target based on direction.
        ConstraintComponent->SetLinearVelocityTarget(FVector(0, 0, VelocityMagnitude * VelocityDirection));
        ConstraintComponent->SetLinearVelocityDrive(false, false, true);
    }
}

void ATerraShiftEnvironment::SetColumnVelocity(int ColumnIndex, float Velocity)
{
    AStaticMeshActor* Column = Columns[ColumnIndex];
    FVector MinLoc = GridCenterPoints[ColumnIndex];
    UPhysicsConstraintComponent* ConstraintComponent = PrismaticJoints[ColumnIndex];

    if (Column && ConstraintComponent)
    {
        FVector Scale = Column->GetActorScale3D();
        float CurrentHeight = Column->GetActorLocation().Z;
        float MaxHeight = Scale.Z * TerraShiftParams->MaxColumnHeight;
        float MinHeight = MinLoc.Z;

        // Going below platform
        if ((CurrentHeight < MinHeight) && (Velocity < 0.0))
        {
            Column->TeleportTo(MinLoc, Column->GetActorRotation(), false, true);
            ConstraintComponent->SetLinearVelocityDrive(false, false, true);
            ConstraintComponent->SetLinearVelocityTarget(FVector(0, 0, 0));
        }
        else {
            ConstraintComponent->SetLinearVelocityTarget(FVector(0, 0, Velocity * TerraShiftParams->ColumnVelocity));
            ConstraintComponent->SetLinearVelocityDrive(false, false, true);
        }
        //// Going above max
        //else if ((CurrentHeight > MaxHeight) && (Velocity > 0.0)) {
        //    FVector MaxLoc(MinLoc.X, MinLoc.Y, MaxHeight);
        //    // Column->TeleportTo(MaxLoc, Column->GetActorRotation(), false, true);
        //    ConstraintComponent->SetLinearVelocityDrive(false, false, true);
        //    ConstraintComponent->SetLinearVelocityTarget(FVector(0, 0, 0));
        //}
    }
}

AStaticMeshActor* ATerraShiftEnvironment::InitializeGridObject()
{
    if (UWorld* World = GetWorld())
    {
        // Pick a random location from GridCenterPoints
        int32 RandomIndex = FMath::RandRange(0, GridCenterPoints.Num() - 1);
        FVector Location = GridCenterPoints[RandomIndex] + FVector(0, 0, 10.0);
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

void ATerraShiftEnvironment::MoveAgent(int AgentIndex, float Value)
{
    if (TerraShiftParams->PositionalDrive) {
        SetColumnHeight(
            AgentIndex,
            map(Value, -1.0, 1.0, 0.0, 1.0)
        );
    }
    else {
        SetColumnVelocity(
            AgentIndex,
            Value
        );
    }
}

TArray<float> ATerraShiftEnvironment::AgentGetState(int AgentIndex) {
    TArray<float> State;
    FVector PlatformWorldSize = Platform->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().BoxExtent * 2.0f * TerraShiftParams->GroundPlaneSize;
    FVector PlatformCenter = Platform->GetActorLocation();

    // Adjust camera height calculation using diagonal and FOV
    float diagonal = FMath::Sqrt(FMath::Square(PlatformWorldSize.X) + FMath::Square(PlatformWorldSize.Y));
    float FOVRadians = FMath::DegreesToRadians(TerraShiftParams->FOV);
    float h = diagonal / (2.0f * FMath::Tan(FOVRadians / 2.0f));
    FVector CameraLocation = PlatformCenter + FVector(0, 0, h);

    // Update for accurate cell size calculation
    float CellSize = PlatformWorldSize.X / GridSize; // Assuming square cells

    // Calculate starting point for traces within each agent's corresponding grid cell
    FVector StartPoint = PlatformCenter - FVector(PlatformWorldSize.X / 2, PlatformWorldSize.Y / 2, 0) + FVector(CellSize / 2, CellSize / 2, 0);
    FVector CellCenter = StartPoint + FVector((AgentIndex % GridSize) * CellSize, (AgentIndex / GridSize) * CellSize, 0);

    // Tracing logic
    for (int TraceIndex = 0; TraceIndex < TerraShiftParams->TracesPerAgent; ++TraceIndex) {
        // Calculate even distribution of traces within the cell
        FVector TraceEnd = CellCenter - CameraLocation;
        TraceEnd.Normalize();
        TraceEnd = CameraLocation + TraceEnd * (h + 100); // 100 is an arbitrary distance to ensure reaching the platform

        FHitResult Hit;
        FCollisionQueryParams Params;
        // Params.AddIgnoredActor(Column);
        bool bHit = GetWorld()->LineTraceSingleByObjectType(Hit, CameraLocation, TraceEnd, FCollisionObjectQueryParams(ECollisionChannel::ECC_PhysicsBody), Params);

        // Debug lines
        DrawDebugLine(GetWorld(), CameraLocation, TraceEnd, bHit ? FColor::Red : FColor::Green, false, -1.0f, (uint8)'\000', 1.0f);

        // Append hit result to state
        if (bHit) {
            State.Add((CameraLocation - Hit.ImpactPoint).Size());
        }
        else {
            State.Add((CameraLocation - TraceEnd).Size());
        }
    }

    return State;
}

int ATerraShiftEnvironment::Get1DIndexFromPoint(const FIntPoint& point, int gridSize)
{
    return point.X * gridSize + point.Y;
}

float ATerraShiftEnvironment::GridDistance(const FIntPoint& Point1, const FIntPoint& Point2)
{
    return FMath::Sqrt(FMath::Square(static_cast<float>(Point2.X) - static_cast<float>(Point1.X)) + FMath::Square(static_cast<float>(Point2.Y) - static_cast<float>(Point1.Y)));
}

FState ATerraShiftEnvironment::ResetEnv(int NumAgents)
{
    /*
        This function reinitializes the TerraShift grid to its default neutral positions
    */
    CurrentStep = 0;



    // Always make sure after modifying actors
    return State();
}

void ATerraShiftEnvironment::Act(FAction Action)
{
    if ((CurrentStep % 256) == 0) { // TODO:: Remove this once AI is hooked up
        if (CurrentPressure == 0.0) {
            CurrentPressure = 1.0;
        }
        else if (CurrentPressure == 1.0) {
            CurrentPressure = -1.0;
        }
        else if (CurrentPressure == -1.0) {
            CurrentPressure = 0.0;
        }
    }
    for (int i = 0; i < MaxAgents; ++i)
    {
        MoveAgent(i, 
            FMath::RandRange(-1.0, 1.0)
            // CurrentPressure
        );
    }
}

void ATerraShiftEnvironment::PostStep()
{
    CurrentStep += 1;
}

FState ATerraShiftEnvironment::State()
{
    /*
        This function gets each agent;s indivual observations, concatinating them into one
        large state array.
    */
    FState CurrentState;

    for (int i = 0; i < MaxAgents; ++i) {
        CurrentState.Values += AgentGetState(i);
    }

    return CurrentState;
}

bool ATerraShiftEnvironment::Done()
{
    /*
        This function check for terminating condition with environment.
    */

    // TODO: Check if any of the objects are out of bounds.

    return false;
}

bool ATerraShiftEnvironment::Trunc()
{
    /*
        This function checks if we have reached the maximum number of steps in the environment
    */
    if (CurrentStep > MaxSteps) {
        CurrentStep = 0;
        return true;
    }

    return false;
}

float ATerraShiftEnvironment::Reward()
{
    /*
        This function determines the collective rewards for the environment
        +1 for Object reach goal
        -1 for Object falling out of bounds
        -0.0001 for step penalty
    */
    float totalrewards = 0.0;

    // TODO

    return totalrewards;
}

void ATerraShiftEnvironment::PostTransition() {

}

void ATerraShiftEnvironment::setCurrentAgents(int NumAgents) {
    CurrentAgents = NumAgents;
}

float ATerraShiftEnvironment::map(float x, float in_min, float in_max, float out_min, float out_max) {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

