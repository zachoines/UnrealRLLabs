#include "CubeEnvironment.h"


// Sets default values
ACubeEnvironment::ACubeEnvironment()
{
    // Setup Env Info
    currentUpdate = 0;
    EnvInfo.StateSize = 6;
    EnvInfo.IsMultiAgent = false;

    const TArray<FContinuousActionSpec>& ContinuousActions = {
        {-1.0, 1.0},
        {-1.0, 1.0}
    };
    const TArray<FDiscreteActionSpec>& DiscreteActions = {
        // Your discrete actions initialization
    };

    // Create a default sub-object for ActionSpace
    EnvInfo.ActionSpace = CreateDefaultSubobject<UActionSpace>(TEXT("ActionSpace"));
    if (EnvInfo.ActionSpace)
    {
        EnvInfo.ActionSpace->Init(ContinuousActions, DiscreteActions);
    }
    EnvInfo.EnvID = 1;
}

void ACubeEnvironment::InitEnv(FBaseInitParams* BaseParams)
{
    this->CubeParams = static_cast<FCubeEnvironmentInitParams*>(BaseParams);;
    
   
    // Initialize Ground Plane
    FVector GroundPlaneSpawnLocation = CubeParams->Location;
    GroundPlane = GetWorld()->SpawnActor<AStaticMeshActor>(GroundPlaneSpawnLocation, FRotator::ZeroRotator);
    UStaticMesh* PlaneMesh = LoadObject<UStaticMesh>(nullptr, TEXT("StaticMesh'/Engine/BasicShapes/Plane.Plane'"));
    if (GroundPlane)
    {
        GroundPlane->GetStaticMeshComponent()->SetStaticMesh(PlaneMesh);
        GroundPlane->GetStaticMeshComponent()->SetWorldScale3D(CubeParams->GroundPlaneSize);
        GroundPlane->SetMobility(EComponentMobility::Movable);

        UMaterial* WoodMaterial = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/StarterContent/Materials/M_Wood_Oak.M_Wood_Oak'"));
        WoodMaterial->TwoSided = true;
        GroundPlane->GetStaticMeshComponent()->SetMaterial(0, WoodMaterial);
        GroundPlane->GetStaticMeshComponent()->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        GroundPlane->GetStaticMeshComponent()->SetCollisionResponseToAllChannels(ECollisionResponse::ECR_Block);

        GroundPlaneTransform = GroundPlane->GetActorTransform();
        InverseGroundPlaneTransform = GroundPlaneTransform.Inverse();
        GroundPlaneSize = GroundPlane->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().BoxExtent * CubeParams->GroundPlaneSize;
        GroundPlaneCenter = GroundPlane->GetActorLocation();
    }

    // Initialize Controlled Cube
    UStaticMesh* CubeMesh = LoadObject<UStaticMesh>(nullptr, TEXT("StaticMesh'/Engine/BasicShapes/Cube.Cube'"));
    ControlledCube = GetWorld()->SpawnActor<AStaticMeshActor>(FVector::Zero(), FRotator::ZeroRotator);
    if (ControlledCube)
    {
        ControlledCube->GetStaticMeshComponent()->SetStaticMesh(CubeMesh);
        ControlledCube->GetStaticMeshComponent()->SetWorldScale3D(CubeParams->ControlledCubeSize);
        ControlledCube->SetMobility(EComponentMobility::Movable);
        ControlledCube->SetActorLocation(GenerateRandomLocationCube());

        UMaterial* MetalMaterial = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/StarterContent/Materials/M_Metal_Steel.M_Metal_Steel'"));
        ControlledCube->GetStaticMeshComponent()->SetMaterial(0, MetalMaterial);

        // Physics properties
        ControlledCube->GetStaticMeshComponent()->SetSimulatePhysics(true);
        ControlledCube->GetStaticMeshComponent()->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        ControlledCube->GetStaticMeshComponent()->SetCollisionResponseToAllChannels(ECollisionResponse::ECR_Block);
        ControlledCube->GetStaticMeshComponent()->SetMassOverrideInKg(NAME_None, 1.0f, true);
        ControlledCube->GetStaticMeshComponent()->BodyInstance.LinearDamping = 0.5f;
        ControlledCube->GetStaticMeshComponent()->BodyInstance.AngularDamping = 0.5f;

        CubeSize = ControlledCube->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().BoxExtent * CubeParams->ControlledCubeSize;
    }

    // Initialize Goal Object (gold sphere)
    UStaticMesh* SphereMesh = LoadObject<UStaticMesh>(nullptr, TEXT("StaticMesh'/Engine/BasicShapes/Sphere.Sphere'"));
    GoalObject = GetWorld()->SpawnActor<AStaticMeshActor>(FVector::ZeroVector, FRotator::ZeroRotator);
    if (GoalObject)
    {
        GoalObject->GetStaticMeshComponent()->SetStaticMesh(SphereMesh);
        GoalObject->GetStaticMeshComponent()->SetWorldScale3D(CubeParams->ControlledCubeSize);
        GoalObject->SetMobility(EComponentMobility::Movable);

        UMaterial* GoldMaterial = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/StarterContent/Materials/M_Metal_Gold.M_Metal_Gold'"));
        GoalObject->GetStaticMeshComponent()->SetMaterial(0, GoldMaterial);

        GoalRadius = GoalObject->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().SphereRadius * CubeParams->ControlledCubeSize.Z;
    }
}

FState ACubeEnvironment::ResetEnv(int NumAgents)
{
    currentUpdate = 0;

    // Generate and set random locations for the cube and goal in world space
    CubeWorldLocation = GenerateRandomLocationCube();
    GoalWorldLocation = GenerateRandomLocationCube();
    ControlledCube->SetActorLocation(CubeWorldLocation);
    GoalObject->SetActorLocation(GoalWorldLocation);

    // Reset the physics properties of the ControlledCube
    if (ControlledCube->GetStaticMeshComponent()->IsSimulatingPhysics())
    {
        ControlledCube->GetStaticMeshComponent()->SetPhysicsLinearVelocity(FVector::ZeroVector);
        ControlledCube->GetStaticMeshComponent()->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
        ControlledCube->GetStaticMeshComponent()->SetAllPhysicsPosition(CubeWorldLocation);
        ControlledCube->GetStaticMeshComponent()->SetAllPhysicsRotation(FRotator::ZeroRotator);
    }

    // Always make sure after modifying actors
    Update();
    return State();
}

void ACubeEnvironment::Act(FAction Action)
{
    // Get the delta time for the current tick
    float DeltaTime = GetWorld()->GetDeltaSeconds();

    // Set the linear velocity based on the action
    FVector CubeForwardInWorldSpace = ControlledCube->GetActorForwardVector();
    FVector LinearVelocity = CubeForwardInWorldSpace * Action.Values[0] * MaxLinearSpeed;
    ControlledCube->GetStaticMeshComponent()->SetPhysicsLinearVelocity(LinearVelocity);

    // Set the angular velocity based on the action
    FVector AngularVelocity(0, 0, Action.Values[1] * MaxAngularSpeed);
    ControlledCube->GetStaticMeshComponent()->SetPhysicsAngularVelocityInDegrees(AngularVelocity);
}

void ACubeEnvironment::Update() 
{
    currentUpdate += 1;
    CubeWorldRotation = ControlledCube->GetActorRotation();
    CubeWorldLocation = ControlledCube->GetActorLocation();
    GoalWorldLocation = GoalObject->GetActorLocation();
    CubeLocationRelativeToGround = InverseGroundPlaneTransform.TransformPosition(CubeWorldLocation);
    GoalLocationRelativeToGround = InverseGroundPlaneTransform.TransformPosition(GoalWorldLocation);
    CubeDistToGoal = FVector::Dist(CubeLocationRelativeToGround, GoalLocationRelativeToGround);
    CubeOffGroundPlane = IsCubeOffGroundPlane();
}

FState ACubeEnvironment::State()
{
    FState CurrentState;

    // 1. Distance from Cube to Goal (normalized between -1.0 and 1.0)
    float MaxPossibleDistance = FVector::Dist(FVector(-GroundPlaneSize.X, -GroundPlaneSize.Y, 0), FVector(GroundPlaneSize.X, GroundPlaneSize.Y, 0));
    CurrentState.Values.Add(2.0f * (FVector::Dist(CubeLocationRelativeToGround, GoalLocationRelativeToGround) / MaxPossibleDistance) - 1.0f);

    // 2. Angle from cube's forward vector to goal (normalized between -1.0 and 1.0)
    FVector CubeForwardInLocalSpace = InverseGroundPlaneTransform.TransformVector(ControlledCube->GetActorForwardVector()).GetSafeNormal();
    FVector CubeToGoalVector = (GoalLocationRelativeToGround - CubeLocationRelativeToGround).GetSafeNormal();
    float AngleToGoal = FMath::Acos(FVector::DotProduct(CubeForwardInLocalSpace, CubeToGoalVector));
    CurrentState.Values.Add(FVector::CrossProduct(CubeForwardInLocalSpace, CubeToGoalVector).Z < 0 ? -AngleToGoal / PI : AngleToGoal / PI);

    // 3. Normalized X and Y position of Cube
    CurrentState.Values.Add(2.0f * (CubeLocationRelativeToGround.X / (2.0f * GroundPlaneSize.X)) - 1.0f);
    CurrentState.Values.Add(2.0f * (CubeLocationRelativeToGround.Y / (2.0f * GroundPlaneSize.Y)) - 1.0f);

    // 4. Normalized linear velocity of the cube
    float MaxLinearVelocity = 1000.0f; // Set this based on your expected maximum linear velocity
    float LinearVelocity = ControlledCube->GetStaticMeshComponent()->GetPhysicsLinearVelocity().Size();
    CurrentState.Values.Add(2.0f * (LinearVelocity / MaxLinearVelocity) - 1.0f);

    // 5. Normalized angular velocity of the cube
    float MaxAngularVelocity = 360.0f; // Set this based on your expected maximum angular velocity in degrees per second
    float AngularVelocity = ControlledCube->GetStaticMeshComponent()->GetPhysicsAngularVelocityInDegrees().Size();
    CurrentState.Values.Add(2.0f * (AngularVelocity / MaxAngularVelocity) - 1.0f);

    return CurrentState;
}

bool ACubeEnvironment::Done() 
{
    return (CubeDistToGoal <= GoalRadius) || CubeOffGroundPlane;
}

bool ACubeEnvironment::Trunc()
{
    if (currentUpdate >= maxStepsPerEpisode) {
        return true;
    }

    return false;
}

float ACubeEnvironment::Reward()
{
    float reward = 0.0;

    // Add step penalty
    reward += -.01;

    // Add distance penalty
    reward += -(CubeDistToGoal / FVector::Dist(FVector(-GroundPlaneSize.X, -GroundPlaneSize.Y, 0), FVector(GroundPlaneSize.X, GroundPlaneSize.Y, 0)));;
    
    // Add reaching goal bonus
    reward += CubeDistToGoal <= GoalRadius ? 1.0 : 0.0;

    // Add out of bounds penalty
    reward += CubeOffGroundPlane ? -1.0f : 0.0;
    
    return reward;
}

FVector ACubeEnvironment::GenerateRandomLocationOnPlane()
{
    // Get the actual size of the GroundPlane in world space
    UStaticMesh* GroundPlaneMesh = GroundPlane->GetStaticMeshComponent()->GetStaticMesh();
    FVector GroundPlaneActualSize = GroundPlaneMesh->GetBounds().BoxExtent;

    // Calculate the bounds for the random location based on the ground plane's actual size and the cube's size
    FVector LocalRandomLocation(FMath::RandRange(-GroundPlaneActualSize.X, GroundPlaneActualSize.X), FMath::RandRange(-GroundPlaneActualSize.Y, GroundPlaneActualSize.Y), GroundPlaneActualSize.Z);

    // Convert the local random location to world space
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

bool ACubeEnvironment::IsCubeOffGroundPlane()
{
    bool isXLeftEdgeWithinBounds = FMath::IsWithin(CubeWorldLocation.X - CubeSize.X, GroundPlaneCenter.X - GroundPlaneSize.X, GroundPlaneCenter.X + GroundPlaneSize.X);
    bool isXRightEdgeWithinBounds = FMath::IsWithin(CubeWorldLocation.X + CubeSize.X, GroundPlaneCenter.X - GroundPlaneSize.X, GroundPlaneCenter.X + GroundPlaneSize.X);

    bool isYBottomEdgeWithinBounds = FMath::IsWithin(CubeWorldLocation.Y - CubeSize.Y, GroundPlaneCenter.Y - GroundPlaneSize.Y, GroundPlaneCenter.Y + GroundPlaneSize.Y);
    bool isYTopEdgeWithinBounds = FMath::IsWithin(CubeWorldLocation.Y + CubeSize.Y, GroundPlaneCenter.Y - GroundPlaneSize.Y, GroundPlaneCenter.Y + GroundPlaneSize.Y);

    return !(isXLeftEdgeWithinBounds && isXRightEdgeWithinBounds && isYBottomEdgeWithinBounds && isYTopEdgeWithinBounds);
}

