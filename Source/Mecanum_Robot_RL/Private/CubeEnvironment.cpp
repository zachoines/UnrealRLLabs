#include "CubeEnvironment.h"
#include "MaterialShared.h"

// Sets default values
ACubeEnvironment::ACubeEnvironment()
{
    // Set default values
    GoalLocation = FVector::ZeroVector;

}

void ACubeEnvironment::InitEnv(FBaseInitParams* BaseParams)
{
    // Initialize the ActionSpace
    TArray<FActionRange> Ranges = { FActionRange{-1.0f, 1.0f}, FActionRange{-1.0f, 1.0f} };
    ActionSpace->InitContinuous(Ranges);

    FCubeEnvironmentInitParams* Params = static_cast<FCubeEnvironmentInitParams*>(BaseParams);
    if (!Params)
    {
        // Handle the error, perhaps by logging or returning early
        UE_LOG(LogTemp, Warning, TEXT("Failed to cast to FCubeEnvironmentInitParams"));
        return;
    }

    // Use the parameters from the struct
    this->CubeParams = Params;

    // Initialize Ground Plane
    FVector GroundPlaneSpawnLocation = Params->Location;
    GroundPlane = GetWorld()->SpawnActor<AStaticMeshActor>(GroundPlaneSpawnLocation, FRotator::ZeroRotator);
    UStaticMesh* PlaneMesh = LoadObject<UStaticMesh>(nullptr, TEXT("StaticMesh'/Engine/BasicShapes/Plane.Plane'"));

    if (GroundPlane)
    {
        // Setup Plane Properties
        GroundPlane->GetStaticMeshComponent()->SetStaticMesh(PlaneMesh);
        GroundPlane->GetStaticMeshComponent()->SetWorldScale3D(Params->GroundPlaneSize);
        GroundPlane->SetMobility(EComponentMobility::Movable);

        // Setup Material
        UMaterial* WoodMaterial = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/StarterContent/Materials/M_Wood_Oak.M_Wood_Oak'"));
        WoodMaterial->TwoSided = true;
        // WoodMaterial->PostEditChange();
        GroundPlane->GetStaticMeshComponent()->SetMaterial(0, WoodMaterial);

        // Setup Physics Properties
        GroundPlane->GetStaticMeshComponent()->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        GroundPlane->GetStaticMeshComponent()->SetCollisionResponseToAllChannels(ECollisionResponse::ECR_Block);
    }

    // Initialize Controlled Cube
    UStaticMesh* CubeMesh = LoadObject<UStaticMesh>(nullptr, TEXT("StaticMesh'/Engine/BasicShapes/Cube.Cube'"));
    ControlledCube = GetWorld()->SpawnActor<AStaticMeshActor>(FVector::Zero(), FRotator::ZeroRotator);

    if (ControlledCube)
    {
        // Setup Cube properties
        ControlledCube->GetStaticMeshComponent()->SetStaticMesh(CubeMesh);
        ControlledCube->GetStaticMeshComponent()->SetWorldScale3D(Params->ControlledCubeSize);
        ControlledCube->SetMobility(EComponentMobility::Movable);
        ControlledCube->SetActorLocation(GenerateRandomLocationCube());

        // Setup Material
        UMaterial* MetalMaterial = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/StarterContent/Materials/M_Metal_Steel.M_Metal_Steel'"));
        ControlledCube->GetStaticMeshComponent()->SetMaterial(0, MetalMaterial);

        // Setup Physics Properties
        ControlledCube->GetStaticMeshComponent()->SetSimulatePhysics(false);
        ControlledCube->GetStaticMeshComponent()->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        ControlledCube->GetStaticMeshComponent()->SetCollisionResponseToAllChannels(ECollisionResponse::ECR_Block);
    }

    // Initialize Goal Object (a glowing sphere)
    UStaticMesh* SphereMesh = LoadObject<UStaticMesh>(nullptr, TEXT("StaticMesh'/Engine/BasicShapes/Sphere.Sphere'"));
    GoalObject = GetWorld()->SpawnActor<AStaticMeshActor>(GoalLocation, FRotator::ZeroRotator);

    if (GoalObject)
    {
        // Setup Sphere properties
        GoalObject->GetStaticMeshComponent()->SetStaticMesh(SphereMesh);
        GoalObject->GetStaticMeshComponent()->SetWorldScale3D(Params->ControlledCubeSize); // Same scale as the cube
        GoalObject->SetMobility(EComponentMobility::Movable);

        // Apply a basic glowing material
        UMaterial* GoldMaterial = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/StarterContent/Materials/M_Metal_Gold.M_Metal_Gold'"));
        GoalObject->GetStaticMeshComponent()->SetMaterial(0, GoldMaterial);
    }
}

TArray<float> ACubeEnvironment::ResetEnv()
{
    // Generate a random location for the cube and the goal in world space
    FVector CubeWorldLocation = GenerateRandomLocationCube();
    GoalLocation = GenerateRandomLocationCube();

    // Set the cube's location
    ControlledCube->SetActorLocation(CubeWorldLocation);

    // Set the goal's location
    GoalObject->SetActorLocation(GoalLocation);

    // Convert the cube and goal locations to the local space of the GroundPlane for the state
    FTransform InverseGroundPlaneTransform = GroundPlane->GetActorTransform().Inverse();
    FVector CubeLocationRelativeToGround = InverseGroundPlaneTransform.TransformPosition(CubeWorldLocation);
    FVector GoalLocationRelativeToGround = InverseGroundPlaneTransform.TransformPosition(GoalLocation);

    // Return the initial state (location of the robot and goal relative to the GroundPlane)
    TArray<float> State;
    State.Add(CubeLocationRelativeToGround.X);
    State.Add(CubeLocationRelativeToGround.Y);
    State.Add(CubeLocationRelativeToGround.Z);
    State.Add(GoalLocationRelativeToGround.X);
    State.Add(GoalLocationRelativeToGround.Y);
    State.Add(GoalLocationRelativeToGround.Z);
    return State;
}

TTuple<bool, float, TArray<float>> ACubeEnvironment::Step(TArray<float> Action)
{
    // Get the transformation matrix of the GroundPlane
    FTransform GroundPlaneTransform = GroundPlane->GetActorTransform();
    FTransform InverseGroundPlaneTransform = GroundPlaneTransform.Inverse();

    // Convert the cube's world location to the local space of the GroundPlane
    FVector CubeLocationRelativeToGround = InverseGroundPlaneTransform.TransformPosition(ControlledCube->GetActorLocation());

    // Get the delta time for the current tick
    float DeltaTime = GetWorld()->GetDeltaSeconds();

    // Move the cube based on the action
    FVector ForwardVector = ControlledCube->GetActorForwardVector();
    FVector NewLocationRelativeToGround = CubeLocationRelativeToGround + ForwardVector * Action[0] * DeltaTime; // Linear velocity

    // Apply angular velocity
    if (Action.Num() > 1)
    {
        FRotator NewRotation = ControlledCube->GetActorRotation();
        NewRotation.Yaw += Action[1] * DeltaTime; // Angular velocity
        // ControlledCube->SetActorRotation(NewRotation);
    }

    // Convert the new location back to world space and set the cube's location
    FVector NewWorldLocation = GroundPlaneTransform.TransformPosition(NewLocationRelativeToGround);
    ControlledCube->SetActorLocation(NewWorldLocation);

    // Calculate the reward
    float reward = Reward();

    // Check if the cube has reached the goal
    FVector GoalLocationRelativeToGround = InverseGroundPlaneTransform.TransformPosition(GoalLocation);
    bool Done = (NewLocationRelativeToGround - GoalLocationRelativeToGround).IsNearlyZero();

    // Return the done flag, the reward, and the new state (location of the robot and goal relative to the GroundPlane)
    TArray<float> State;
    State.Add(NewLocationRelativeToGround.X);
    State.Add(NewLocationRelativeToGround.Y);
    State.Add(NewLocationRelativeToGround.Z);
    State.Add(GoalLocationRelativeToGround.X);
    State.Add(GoalLocationRelativeToGround.Y);
    State.Add(GoalLocationRelativeToGround.Z);
    return TTuple<bool, float, TArray<float>>(Done, reward, State);
}

FVector ACubeEnvironment::GenerateRandomLocationOnPlane()
{
    // Get the actual size of the GroundPlane in world space
    UStaticMesh* GroundPlaneMesh = GroundPlane->GetStaticMeshComponent()->GetStaticMesh();
    FVector GroundPlaneActualSize = GroundPlaneMesh->GetBounds().BoxExtent;

    // Calculate the bounds for the random location based on the ground plane's actual size and the cube's size
    FVector LocalRandomLocation(FMath::RandRange(-GroundPlaneActualSize.X, GroundPlaneActualSize.X), FMath::RandRange(-GroundPlaneActualSize.Y, GroundPlaneActualSize.Y), GroundPlaneActualSize.Z);

    // Convert the local random location to world space
    FTransform GroundPlaneTransform = GroundPlane->GetActorTransform();
    return GroundPlaneTransform.TransformPosition(LocalRandomLocation);
}

FVector ACubeEnvironment::GenerateRandomLocationCube()
{
    FVector RandomPlaneLocation = GenerateRandomLocationOnPlane();

    // Adjust the spawn location of the cube so its bottom face is flush with ground
    FVector ActualCubeSize = ControlledCube->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().BoxExtent;
    float CubeSpawnHeight = ActualCubeSize.Z * CubeParams->ControlledCubeSize.Z;
    
    return FVector(RandomPlaneLocation.X, RandomPlaneLocation.Y, RandomPlaneLocation.Z + CubeSpawnHeight);
}

float ACubeEnvironment::Reward()
{
    if ((ControlledCube->GetActorLocation() - GoalLocation).IsNearlyZero())
    {
        return 10.0f;
    }
    return -1.0f;
}
