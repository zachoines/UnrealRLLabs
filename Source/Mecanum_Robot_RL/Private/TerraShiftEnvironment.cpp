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
}

void ATerraShiftEnvironment::InitEnv(FBaseInitParams* BaseParams)
{
    // Store Params
    TerraShiftParams = static_cast<FTerraShiftEnvironmentInitParams*>(BaseParams);

    // Spawn Platform
    FVector PlatformSize = FVector(GridSize * 100.0f, GridSize * 100.0f, 10.0f); // Example dimensions
    AStaticMeshActor* Platform = SpawnPlatform(TerraShiftParams->Location, PlatformSize);

    // Initialize Grid Center Points and spawn columns with prismatic joints
    GridCenterPoints.SetNum(GridSize);
    for (int i = 0; i < GridSize; ++i)
    {
        GridCenterPoints[i].SetNum(GridSize);
        for (int j = 0; j < GridSize; ++j)
        {
            FVector GridCenter = TerraShiftParams->Location + FVector((i - GridSize / 2.0f + 0.5f) * 100.0f, (j - GridSize / 2.0f + 0.5f) * 100.0f, 0.0f); // Adjust for Z if necessary
            GridCenterPoints[i][j] = GridCenter;

            // Spawn and setup each column at GridCenter
            AStaticMeshActor* Column = SpawnColumn(GridCenter, TerraShiftParams->ColumnSize);
            AttachPrismaticJoint(Column, Platform);
        }
    }
}

void ATerraShiftEnvironment::MoveAgent(int AgentIndex, float position)
{
    /*
        This functon takes in an 1d agent index (column within TerraShift grid)
        and sets the agents position relative to the global "MovementConstraint" (default 0.1).
    */
}

void ATerraShiftEnvironment::SpawnGridObject(FIntPoint SpawnLocation, FIntPoint GaolLocation)
{
    /*
        This function spawns a new object for the TerraShift grid to manipulate.
        It will spawn a sphere will have uniform physics properties (mass; grivity; friction; size)
    */
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

FVector ATerraShiftEnvironment::GetWorldLocationFromGridIndex(FIntPoint GridIndex)
{
    if (GridIndex.X >= 0 && GridIndex.X < GridCenterPoints.Num() && GridIndex.Y >= 0 && GridIndex.Y < GridCenterPoints[0].Num())
    {
        FVector WorldLocation = GridCenterPoints[GridIndex.X][GridIndex.Y];
        // WorldLocation.Z += ... // TODO:: Adjust to be flush with column head

        return WorldLocation;
    }
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
    /*
        This function passes the calculated action from the caller to the respective agent
        (column) in TerraShift grid
    */
    for (int i = 0; i < Action.Values.Num(); ++i)
    {
  
        MoveAgent(i, Action.Values[0]);
    }
}

FIntPoint ATerraShiftEnvironment::GenerateRandomLocation()
{
    /*
        Helper function to generate random location within bounds of TerraShift grid
    */
    FIntPoint RandomPoint;
    do
    {
        RandomPoint.X = FMath::RandRange(0, GridCenterPoints.Num() - 1);
        RandomPoint.Y = FMath::RandRange(0, GridCenterPoints[0].Num() - 1);
    } while (UsedLocations.Contains(RandomPoint));
    return RandomPoint;
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

AStaticMeshActor* ATerraShiftEnvironment::InitializeObject(const FLinearColor& Color)
{
    UStaticMesh* SphereMesh = LoadObject<UStaticMesh>(nullptr, TEXT("StaticMesh'/Engine/BasicShapes/Sphere.Sphere'"));
    AStaticMeshActor* NewObject = GetWorld()->SpawnActor<AStaticMeshActor>(FVector::ZeroVector, FRotator::ZeroRotator);

    if (NewObject)
    {
        NewObject->SetMobility(EComponentMobility::Movable);
        NewObject->GetStaticMeshComponent()->SetStaticMesh(SphereMesh);
        NewObject->GetStaticMeshComponent()->SetWorldScale3D(TerraShiftParams->ObjectSize);
        UMaterial* ColoredMaterial = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/StarterContent/Materials/M_Basic_Floor.M_Basic_Floor'"));
        NewObject->GetStaticMeshComponent()->SetMaterial(0, ColoredMaterial);

        // Create a dynamic material instance to set the color
        UMaterialInstanceDynamic* DynMaterial = UMaterialInstanceDynamic::Create(NewObject->GetStaticMeshComponent()->GetMaterial(0), NewObject);
        DynMaterial->SetVectorParameterValue("Color", Color);
        NewObject->GetStaticMeshComponent()->SetMaterial(0, DynMaterial);

        // TODO::Setup physics properties such as gravity; collisions; friction; mass; etc...
    }

    return NewObject;
}

AStaticMeshActor* ATerraShiftEnvironment::SpawnColumn(FVector Location, FVector Dimensions)
{
    if (UWorld* World = GetWorld())
    {
        static ConstructorHelpers::FObjectFinder<UStaticMesh> ColumnMesh(TEXT("StaticMesh'/Engine/BasicShapes/Cube.Cube'"));
        AStaticMeshActor* ColumnActor = World->SpawnActor<AStaticMeshActor>(Location, FRotator::ZeroRotator);
        if (ColumnActor && ColumnMesh.Succeeded())
        {
            ColumnActor->GetStaticMeshComponent()->SetStaticMesh(ColumnMesh.Object);

            FVector MeshBoundsExtent = ColumnActor->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().BoxExtent;
            FVector Scale = Dimensions / (MeshBoundsExtent * 2);
            ColumnActor->GetStaticMeshComponent()->SetWorldScale3D(Scale);

            UMaterial* Material = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/StarterContent/Materials/M_Metal_Burnished_Steel.M_Metal_Burnished_Steel'"));
            if (Material)
            {
                ColumnActor->GetStaticMeshComponent()->SetMaterial(0, Material);
            }

            ColumnActor->GetStaticMeshComponent()->SetSimulatePhysics(true);
            ColumnActor->GetStaticMeshComponent()->SetMobility(EComponentMobility::Movable);

            return ColumnActor;
        }
    }
    return nullptr;
}

void ATerraShiftEnvironment::AttachPrismaticJoint(AStaticMeshActor* Column, AStaticMeshActor* Platform)
{
    if (!Column || !Platform) return;

    // Create a new physics constraint component
    UPhysicsConstraintComponent* PrismaticJoint = NewObject<UPhysicsConstraintComponent>(Column);
    if (!PrismaticJoint) return;

    // Attach the constraint component to the column
    PrismaticJoint->AttachToComponent(Column->GetRootComponent(), FAttachmentTransformRules::KeepWorldTransform);

    // Now, configure the constraint directly.
    PrismaticJoint->SetLinearXLimit(ELinearConstraintMotion::LCM_Locked, 0);
    PrismaticJoint->SetLinearYLimit(ELinearConstraintMotion::LCM_Locked, 0);
    PrismaticJoint->SetLinearZLimit(ELinearConstraintMotion::LCM_Free, 0);

    PrismaticJoint->SetAngularSwing1Limit(EAngularConstraintMotion::ACM_Locked, 0);
    PrismaticJoint->SetAngularSwing2Limit(EAngularConstraintMotion::ACM_Locked, 0);
    PrismaticJoint->SetAngularTwistLimit(EAngularConstraintMotion::ACM_Locked, 0);

    // Initialize the constrained components (Column and Platform)
    PrismaticJoint->ConstraintActor1 = Column;
    PrismaticJoint->ConstraintActor2 = Platform; // Use nullptr if you want the world as the second component

    // Optionally, you might want to define the constraint's position and orientation
    PrismaticJoint->SetRelativeLocationAndRotation(FVector::ZeroVector, FRotator::ZeroRotator);

    // Apply the configuration and activate the joint
    PrismaticJoint->InitComponentConstraint();

    // Register the constraint component with the engine
    PrismaticJoint->RegisterComponent();
}

void ATerraShiftEnvironment::SpawnPlatformWithColumns(FVector CenterPoint, int32 GridSize)
{
    float PlatformWidth = 100.0f;
    FVector ColumnDimensions = FVector(10.0f, 10.0f, 50.0f);

    FVector PlatformSize = FVector(GridSize * PlatformWidth, GridSize * PlatformWidth, 10.0f);
    AStaticMeshActor* Platform = SpawnPlatform(CenterPoint, PlatformSize);

    FVector StartPoint = CenterPoint - FVector(GridSize / 2.0f * PlatformWidth, GridSize / 2.0f * PlatformWidth, 0.0f);

    for (int32 i = 0; i < GridSize; ++i)
    {
        for (int32 j = 0; j < GridSize; ++j)
        {
            FVector ColumnLocation = StartPoint + FVector(i * PlatformWidth + PlatformWidth / 2.0f, j * PlatformWidth + PlatformWidth / 2.0f, ColumnDimensions.Z / 2.0f);
            AStaticMeshActor* Column = SpawnColumn(ColumnLocation, ColumnDimensions);
            AttachPrismaticJoint(Column, Platform);
        }
    }
}

AStaticMeshActor* ATerraShiftEnvironment::SpawnPlatform(FVector Location, FVector Size)
{
    UStaticMesh* PlaneMesh = LoadObject<UStaticMesh>(nullptr, TEXT("StaticMesh'/Engine/BasicShapes/Plane.Plane'"));
    AStaticMeshActor* Platform = GetWorld()->SpawnActor<AStaticMeshActor>(Location, FRotator::ZeroRotator);

    if (Platform)
    {
        Platform->GetStaticMeshComponent()->SetStaticMesh(PlaneMesh);
        Platform->GetStaticMeshComponent()->SetWorldScale3D(Size);
        Platform->SetMobility(EComponentMobility::Movable);

        UMaterial* Material = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/StarterContent/Materials/M_Wood_Oak.M_Wood_Oak'"));
        if (Material)
        {
            Material->TwoSided = true;
            Platform->GetStaticMeshComponent()->SetMaterial(0, Material);
        }
        Platform->GetStaticMeshComponent()->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        Platform->GetStaticMeshComponent()->SetCollisionResponseToAllChannels(ECollisionResponse::ECR_Block);
    }

    return Platform;
}

