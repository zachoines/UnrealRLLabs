// Fill out your copyright notice in the Description page of Project Settings.


#include "TerraShiftEnvironment.h"


ATerraShiftEnvironment::ATerraShiftEnvironment()
{
    // Setup Env Info
    EnvInfo.EnvID = 3;
    EnvInfo.MaxAgents = MaxAgents;
    EnvInfo.SingleAgentObsSize = 2;
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
    float CellHeight = CellWidth; // TerraShiftParams->ColumnHeight;
    for (int i = 0; i < GridSize; ++i)
    {

        for (int j = 0; j < GridSize; ++j)
        { 
            // Spawn each column.
            AStaticMeshActor* Column = SpawnColumn(
                FVector(CellWidth, CellWidth, CellHeight) ,
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
        }
    }

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
        PrismaticJoint->SetLinearXLimit(ELinearConstraintMotion::LCM_Locked, 0);
        PrismaticJoint->SetLinearYLimit(ELinearConstraintMotion::LCM_Locked, 0);
        PrismaticJoint->SetLinearZLimit(ELinearConstraintMotion::LCM_Limited, 10);

        PrismaticJoint->SetAngularSwing1Limit(EAngularConstraintMotion::ACM_Locked, 0);
        PrismaticJoint->SetAngularSwing2Limit(EAngularConstraintMotion::ACM_Locked, 0);
        PrismaticJoint->SetAngularTwistLimit(EAngularConstraintMotion::ACM_Locked, 0);

        PrismaticJoint->SetConstrainedComponents(Column->GetStaticMeshComponent(), NAME_None, Platform->GetStaticMeshComponent(), NAME_None);
        PrismaticJoint->InitComponentConstraint();
        PrismaticJoint->RegisterComponent();
        PrismaticJoint->SetActive(true);
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
        FVector CurrentLocation = Column->GetActorLocation();
        if (CurrentLocation.Z <= MinLoc.Z)
        {
            ConstraintComponent->SetLinearPositionTarget(FVector(0, 0, 0));
            Column->TeleportTo(MinLoc, Column->GetActorRotation(), false, true);
        }
        else
        {
            // Otherwise, apply the desired position
            ConstraintComponent->SetLinearPositionTarget(FVector(0, 0, 10 * NewHeight));
            ConstraintComponent->SetLinearPositionDrive(false, false, true);
        }
    }
}

void ATerraShiftEnvironment::SetColumnVelocity(int ColumnIndex, float Velocity)
{
    AStaticMeshActor* Column = Columns[ColumnIndex];
    FVector MinLoc = GridCenterPoints[ColumnIndex];
    UPhysicsConstraintComponent* ConstraintComponent = PrismaticJoints[ColumnIndex];

    if (Column && ConstraintComponent)
    {
        FVector CurrentLocation = Column->GetActorLocation();
        if (CurrentLocation.Z <= MinLoc.Z)
        {
            ConstraintComponent->SetLinearVelocityTarget(FVector(0, 0, 0));
            Column->TeleportTo(MinLoc, Column->GetActorRotation(), false, true);
        }
        else
        {
            // Otherwise, apply the desired velocity
            ConstraintComponent->SetLinearVelocityTarget(FVector(0, 0, 100 * Velocity));
            ConstraintComponent->SetLinearVelocityDrive(false, false, true);
        }
    }
}

void ATerraShiftEnvironment::ApplyForceToColumn(int ColumnIndex, float ForceMagnitude)
{   
    AStaticMeshActor* Column = Columns[ColumnIndex];
    if (Column)
    {
        UStaticMeshComponent* MeshComponent = Column->GetStaticMeshComponent();
        if (MeshComponent)
        {
            FVector ForceDirection = FVector(0, 0, 1); // Assuming the Z-axis is the desired direction
            MeshComponent->AddForce(ForceDirection * ForceMagnitude);
        }
    }
}

AStaticMeshActor* ATerraShiftEnvironment::InitializeGridObject() // const FLinearColor& Color)
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
                ObjectActor->GetStaticMeshComponent()->SetStaticMesh(SphereMesh);
                ObjectActor->GetStaticMeshComponent()->SetWorldScale3D(TerraShiftParams->ObjectSize);

                UMaterial* Material = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/Material/Manip_Object_Material.Manip_Object_Material'"));
                if (Material)
                {
                    ObjectActor->GetStaticMeshComponent()->SetMaterial(0, Material);
                }

                ObjectActor->GetStaticMeshComponent()->SetMobility(EComponentMobility::Movable);
                ObjectActor->GetStaticMeshComponent()->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
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
    /*
        This functon takes in an 1d agent index (column within TerraShift grid)
        and sets the agents velocity
    */
    SetColumnVelocity(AgentIndex, Value);
    /*SetColumnHeight(AgentIndex, 
        map(Value, -1.0, 1.0, 0.0, 1.0)
    );*/
}

TArray<float> ATerraShiftEnvironment::AgentGetState(int AgentIndex)
{
    /*
        This function gets a single agents state (column).
        It is comprized of its current position and load.
    */
    TArray<float> State;
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
        if (CurrentPressure == 1.0) {
            CurrentPressure = 0.0;
        }
        else if (CurrentPressure == 0.0) {
            CurrentPressure = -1.0;
        }
        else if (CurrentPressure == -1.0) {
            CurrentPressure = 1.0;
        }
    }
    for (int i = 0; i < MaxAgents; ++i)
    {
        MoveAgent(i, CurrentPressure);
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

    for (int i = 0; i < CurrentAgents; ++i) {
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

