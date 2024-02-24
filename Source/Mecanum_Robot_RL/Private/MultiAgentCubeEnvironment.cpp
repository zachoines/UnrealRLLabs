// Fill out your copyright notice in the Description page of Project Settings.


#include "MultiAgentCubeEnvironment.h"

AMultiAgentCubeEnvironment::AMultiAgentCubeEnvironment()
{
    // Setup Env Info
    EnvInfo.EnvID = 2;
    EnvInfo.MaxAgents = MaxAgents;
    EnvInfo.SingleAgentObsSize = ((2 * AgentVisibility + 1) * (2 * AgentVisibility + 1)) + 2;
    EnvInfo.StateSize = EnvInfo.MaxAgents * EnvInfo.SingleAgentObsSize;
    EnvInfo.IsMultiAgent = true;

    const TArray<FContinuousActionSpec>& ContinuousActions = {};
    const TArray<FDiscreteActionSpec>& DiscreteActions = {
        { 5 }
        /*
            up,
            down,
            left,
            right,
            no-op
        */
    };

    // Create a default sub-object for ActionSpace
    EnvInfo.ActionSpace = CreateDefaultSubobject<UActionSpace>(TEXT("ActionSpace"));
    if (EnvInfo.ActionSpace)
    {
        EnvInfo.ActionSpace->Init(ContinuousActions, DiscreteActions);
    }
}

void AMultiAgentCubeEnvironment::InitEnv(FBaseInitParams* BaseParams)
{
    // Store Params
    MultiAgentCubeParams = static_cast<FMultiAgentCubeEnvironmentInitParams*>(BaseParams);

    // Initialize Ground Plane
    CurrentAgents = EnvInfo.MaxAgents;
    CubeSize = MultiAgentCubeParams->ControlledCubeSize;
    FVector PlaneSize = CubeSize * GridSize;
    MultiAgentCubeParams->GroundPlaneSize = PlaneSize;

    FVector GroundPlaneSpawnLocation = MultiAgentCubeParams->Location;
    GroundPlane = GetWorld()->SpawnActor<AStaticMeshActor>(GroundPlaneSpawnLocation, FRotator::ZeroRotator);
    UStaticMesh* PlaneMesh = LoadObject<UStaticMesh>(nullptr, TEXT("StaticMesh'/Engine/BasicShapes/Plane.Plane'"));
    if (GroundPlane)
    {
        GroundPlane->GetStaticMeshComponent()->SetStaticMesh(PlaneMesh);
        GroundPlane->GetStaticMeshComponent()->SetWorldScale3D(MultiAgentCubeParams->GroundPlaneSize);
        GroundPlane->SetMobility(EComponentMobility::Movable);

        UMaterial* WoodMaterial = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/StarterContent/Materials/M_Wood_Oak.M_Wood_Oak'"));
        WoodMaterial->TwoSided = true;
        GroundPlane->GetStaticMeshComponent()->SetMaterial(0, WoodMaterial);
        GroundPlane->GetStaticMeshComponent()->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        GroundPlane->GetStaticMeshComponent()->SetCollisionResponseToAllChannels(ECollisionResponse::ECR_Block);

        GroundPlaneTransform = GroundPlane->GetActorTransform();
        InverseGroundPlaneTransform = GroundPlaneTransform.Inverse();
        GroundPlaneSize = GroundPlane->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().BoxExtent * MultiAgentCubeParams->GroundPlaneSize;
        GroundPlaneCenter = GroundPlane->GetActorLocation();
    }

    // Calculate the actual world size of the ground plane
    FVector GroundPlaneWorldSize = GroundPlane->GetStaticMeshComponent()->GetStaticMesh()->GetBounds().BoxExtent * 2.0f * MultiAgentCubeParams->GroundPlaneSize;

    // Now use GroundPlaneWorldSize to calculate the grid points
    GridCenterPoints.SetNum(GridSize);
    for (int i = 0; i < GridSize; ++i)
    {
        GridCenterPoints[i].SetNum(GridSize);
        for (int j = 0; j < GridSize; ++j)
        {
            FVector GridCenter = GroundPlaneCenter + FVector((i - GridSize / 2.0f + 0.5f) * (GroundPlaneWorldSize.X / GridSize), (j - GridSize / 2.0f + 0.5f) * (GroundPlaneWorldSize.Y / GridSize), CubeSize.Z / 2);
            GridCenterPoints[i][j] = GridCenter;
        }
    }
}

void AMultiAgentCubeEnvironment::AssignRandomGridLocations()
{
    UsedLocations.Empty();
    AgentGoalPositions.Empty();
    ActorToLocationMap.Empty();

    for (int i = 0; i < CurrentAgents; ++i)
    {
        // Initialize cubes
        FIntPoint CubeLocationIndex = GenerateRandomLocation();
        FVector CubeWorldLocation = GetWorldLocationFromGridIndex(CubeLocationIndex);
        ControlledCubes[i]->SetActorLocation(CubeWorldLocation);
        ActorToLocationMap.Add(ControlledCubes[i], CubeLocationIndex);
        UsedLocations.FindOrAdd(CubeLocationIndex).Add(ControlledCubes[i]);

        // Initialize goals
        FIntPoint GoalLocationIndex = GenerateRandomLocation();
        FVector GoalWorldLocation = GetWorldLocationFromGridIndex(GoalLocationIndex);
        GoalObjects[i]->SetActorLocation(GoalWorldLocation);
        ActorToLocationMap.Add(GoalObjects[i], GoalLocationIndex);
        UsedLocations.FindOrAdd(GoalLocationIndex).Add(GoalObjects[i]);

        AgentGoalPositions.Add(i, TPair<FIntPoint, FIntPoint>(CubeLocationIndex, GoalLocationIndex));
    }
}

void AMultiAgentCubeEnvironment::MoveAgent(int AgentIndex, FIntPoint Location)
{
    if (!AgentGoalPositions.Contains(AgentIndex)) {
        UE_LOG(LogTemp, Log, TEXT("Undefined Agent Index: %"), AgentIndex);
    }
    else 
    {
        FVector NewWorldLocation = GetWorldLocationFromGridIndex(Location);
        if (NewWorldLocation != FVector::ZeroVector) {
            ControlledCubes[AgentIndex]->SetActorLocation(NewWorldLocation); // out of bounds movement
        }

        // Update AgentGoalPositions
        FIntPoint OldLocation = AgentGoalPositions[AgentIndex].Key;
        AgentGoalPositions[AgentIndex].Key = Location;

        // Update UsedLocations: remove the agent from its old location and add to the new one
        if (UsedLocations.Contains(OldLocation))
        {
            UsedLocations[OldLocation].Remove(ControlledCubes[AgentIndex]);
            if (UsedLocations[OldLocation].Num() == 0)
            {
                UsedLocations.Remove(OldLocation); // Remove the entry if no more actors are at this location
            }
        }
        UsedLocations.FindOrAdd(Location).Add(ControlledCubes[AgentIndex]);

        // Update ActorToLocationMap
        ActorToLocationMap[ControlledCubes[AgentIndex]] = Location;
    }
}

void AMultiAgentCubeEnvironment::MoveGoal(int AgentIndex, FIntPoint Location)
{
    if (!AgentGoalPositions.Contains(AgentIndex)) {
        UE_LOG(LogTemp, Log, TEXT("Undefined Agent Index: %"), AgentIndex);
    }
    else
    {
        FVector NewWorldLocation = GetWorldLocationFromGridIndex(Location);
        if (NewWorldLocation != FVector::ZeroVector) {  // out of bounds movement
            GoalObjects[AgentIndex]->SetActorLocation(NewWorldLocation);
        }

        // Update AgentGoalPositions
        FIntPoint OldLocation = AgentGoalPositions[AgentIndex].Value;
        AgentGoalPositions[AgentIndex].Value = Location;

        // Update UsedLocations: remove the goal from its old location and add to the new one
        if (UsedLocations.Contains(OldLocation))
        {
            UsedLocations[OldLocation].Remove(GoalObjects[AgentIndex]);
            if (UsedLocations[OldLocation].Num() == 0)
            {
                UsedLocations.Remove(OldLocation);
            }
        }
        UsedLocations.FindOrAdd(Location).Add(GoalObjects[AgentIndex]);

        // Update ActorToLocationMap
        ActorToLocationMap[GoalObjects[AgentIndex]] = Location;
    }
}

bool AMultiAgentCubeEnvironment::AgentCollidedWithAgent(int AgentIndex)
{
    if (!AgentGoalPositions.Contains(AgentIndex)) {
        UE_LOG(LogTemp, Log, TEXT("Undefined Agent Index: %"), AgentIndex);
        return false;
    }

    FIntPoint AgentLocation = AgentGoalPositions[AgentIndex].Key;
    TArray<AStaticMeshActor*>* ActorsAtLocation = UsedLocations.Find(AgentLocation);

    if (!ActorsAtLocation || ActorsAtLocation->Num() <= 1)
    {
        return false; // No collision if only one actor is at this location
    }

    for (AStaticMeshActor* Actor : *ActorsAtLocation)
    {
        if (Actor != ControlledCubes[AgentIndex] && Actor != GoalObjects[AgentIndex]) // Exclude the agent and its gaol
        {
            if (this->ControlledCubes.Contains(Actor)) {
                return true;
            }
        }
    }

    return false; // No collision found
}

bool AMultiAgentCubeEnvironment::AgentWrongGoal(int AgentIndex)
{
    if (!AgentGoalPositions.Contains(AgentIndex)) {
        UE_LOG(LogTemp, Log, TEXT("Undefined Agent Index: %"), AgentIndex);
        return false;
    }

    FIntPoint AgentLocation = AgentGoalPositions[AgentIndex].Key;
    TArray<AStaticMeshActor*>* ActorsAtLocation = UsedLocations.Find(AgentLocation);

    if (!ActorsAtLocation || ActorsAtLocation->Num() <= 1)
    {
        return false; // No collision if only one actor is at this location
    }

    for (AStaticMeshActor* Actor : *ActorsAtLocation)
    {
        if (Actor != ControlledCubes[AgentIndex] && Actor != GoalObjects[AgentIndex]) // Exclude the agent and its gaol
        {
            if (GoalObjects.Contains(Actor)) {
                return true;
            }
        }
    }

    return false; // No collision found
}

TArray<float> AMultiAgentCubeEnvironment::AgentGetState(int AgentIndex)
{
    TArray<float> State;
    if (!AgentGoalPositions.Contains(AgentIndex)) {
        UE_LOG(LogTemp, Log, TEXT("Undefined Agent Index: %d"), AgentIndex);
        return State;
    }

    const int MaxAgentID = MaxAgents;
    const int MaxGoalID = MaxAgents;

    // Initialize the grid
    TArray<float> Grid;
    Grid.Init(0, GridSize * GridSize);

    // Encode each agent and goal
    for (auto& Elem : AgentGoalPositions) {
        int AgentID = Elem.Key;
        int GoalID = AgentID;
        FIntPoint AgentLocation = Elem.Value.Key;
        FIntPoint GoalLocation = Elem.Value.Value;
        float AgentValue = (static_cast<float>(AgentID + 1)) / static_cast<float>(CurrentAgents + 1);
        float GoalValue = -(static_cast<float>(GoalID + 1)) / static_cast<float>(CurrentAgents + 1);

        if (AgentLocation == GoalLocation) {
            Grid[Get1DIndexFromPoint(AgentLocation, GridSize)] = 1.0 + AgentValue;
        }
        else {
            if (AgentIndex == AgentID) {
                Grid[Get1DIndexFromPoint(AgentLocation, GridSize)] = AgentValue;
            }
            else {
                Grid[Get1DIndexFromPoint(AgentLocation, GridSize)] = AgentValue;
                Grid[Get1DIndexFromPoint(GoalLocation, GridSize)] = GoalValue;
            }
        }
    }

    // Calculate the total size of the visibility grid
    int VisibilityGridSize = (2 * AgentVisibility + 1) * (2 * AgentVisibility + 1);

    // Initialize unknown spaces as -1 (outside gride bounds)
    State.Init(-1, VisibilityGridSize + 2);

    FIntPoint AgentPosition = AgentGoalPositions[AgentIndex].Key;

    // Store points only within the agent's visibility
    for (int x = -AgentVisibility; x <= AgentVisibility; ++x) {
        for (int y = -AgentVisibility; y <= AgentVisibility; ++y) {
            int GlobalX = AgentPosition.X + x;
            int GlobalY = AgentPosition.Y + y;

            if (GlobalX >= 0 && GlobalX < GridSize && GlobalY >= 0 && GlobalY < GridSize) {
                int GridIndex = Get1DIndexFromPoint(FIntPoint(GlobalX, GlobalY), GridSize);
                int StateIndex = (x + AgentVisibility) * (2 * AgentVisibility + 1) + (y + AgentVisibility);
                State[StateIndex] = Grid[GridIndex];
            }
        }
    }

    // Add the agent's global position at the end of the State array
    State[VisibilityGridSize] = AgentPosition.X / static_cast<float>(GridSize);
    State[VisibilityGridSize + 1] = AgentPosition.Y / static_cast<float>(GridSize);

    return State;
}

int AMultiAgentCubeEnvironment::Get1DIndexFromPoint(const FIntPoint& point, int gridSize) 
{
    return point.X * gridSize + point.Y;
}

float AMultiAgentCubeEnvironment::GridDistance(const FIntPoint& Point1, const FIntPoint& Point2)
{
    return FMath::Sqrt(FMath::Square(static_cast<float>(Point2.X) - static_cast<float>(Point1.X)) + FMath::Square(static_cast<float>(Point2.Y) - static_cast<float>(Point1.Y)));
}

bool AMultiAgentCubeEnvironment::AgentGoalReached(int AgentIndex)
{
    if (!AgentGoalPositions.Contains(AgentIndex)) {
        UE_LOG(LogTemp, Log, TEXT("Undefined Agent Index: %"), AgentIndex);
        return false;
    }

    FIntPoint AgentLocation = AgentGoalPositions[AgentIndex].Key;
    FIntPoint GoalLocation = AgentGoalPositions[AgentIndex].Value;
    return AgentLocation == GoalLocation;
}

bool AMultiAgentCubeEnvironment::AgentOutOfBounds(int AgentIndex)
{
    FIntPoint Location = AgentGoalPositions[AgentIndex].Key;

    if (Location.X >= 0 && Location.X < GridCenterPoints.Num() &&
        Location.Y >= 0 && Location.Y < GridCenterPoints[0].Num())
    {
        return false;
    }
    
    return true;
}

FVector AMultiAgentCubeEnvironment::GetWorldLocationFromGridIndex(FIntPoint GridIndex)
{
    if (GridIndex.X >= 0 && GridIndex.X < GridCenterPoints.Num() && GridIndex.Y >= 0 && GridIndex.Y < GridCenterPoints[0].Num())
    {
        FVector WorldLocation = GridCenterPoints[GridIndex.X][GridIndex.Y];
        WorldLocation.Z += CubeSize.Z; // Adjust to be flush on ground plane
        if (WorldLocation == FVector::ZeroVector) {
            WorldLocation += FVector(1.0e-6, 1.0e-6, 1.0e-6); // Prevent overlap with zero-vector sentinal value
        }
        return WorldLocation;
    }

    return FVector::ZeroVector; // Out of bounds
}

FState AMultiAgentCubeEnvironment::ResetEnv(int NumAgents)
{
    CurrentStep = 0;

    // Handle reduction of agents
    while (ControlledCubes.Num() > NumAgents)
    {
        AStaticMeshActor* Cube = ControlledCubes.Pop();
        if (Cube)
        {
            Cube->Destroy();
        }

        AStaticMeshActor* Goal = GoalObjects.Pop();
        if (Goal)
        {
            Goal->Destroy();
        }
    }

    // Handle increase of agents
    for (int Count = ControlledCubes.Num(); Count < NumAgents; Count++)
    {
        AStaticMeshActor* NewCube = InitializeCube(Colors[Count]);
        ControlledCubes.Add(NewCube);

        AStaticMeshActor* NewGoal = InitializeGoalObject(Colors[Count]);
        GoalObjects.Add(NewGoal);
    }

    // Update the current number of agents
    CurrentAgents = NumAgents;

    // Now that all agents are correctly set up, assign them new locations
    AssignRandomGridLocations();

    // Always make sure after modifying actors
    return State();
}

void AMultiAgentCubeEnvironment::Act(FAction Action)
{
    if (Action.Values.Num() != ControlledCubes.Num())
    {
        UE_LOG(LogTemp, Log, TEXT("Incorrect Actions Shape: %"), Action.Values.Num());
        return;
    }

    for (int i = 0; i < Action.Values.Num(); ++i)
    {
        FIntPoint CurrentLocation = AgentGoalPositions[i].Key;
        FIntPoint NewLocation = CurrentLocation;

        switch (static_cast<int>(Action.Values[i]))
        {
        case 0: // Up
            NewLocation.Y += 1;
            break;
        case 1: // Down
            NewLocation.Y -= 1;
            break;
        case 2: // Left
            NewLocation.X -= 1;
            break;
        case 3: // Right
            NewLocation.X += 1;
            break;
        case 4: // no-op
            continue;
            break;
        default:
            // No operation
            UE_LOG(LogTemp, Log, TEXT("Undefined Action Index: %"), Action.Values[i]);
            break;
        }

        MoveAgent(i, NewLocation);
    }
}

void AMultiAgentCubeEnvironment::AgentGoalReset(int AgentIndex)
{
    if (!AgentGoalPositions.Contains(AgentIndex)) {
        UE_LOG(LogTemp, Log, TEXT("Undefined Agent Index: %"), AgentIndex);
    }
    GoalReset(AgentIndex);
    AgentReset(AgentIndex);
}

void AMultiAgentCubeEnvironment::GoalReset(int AgentIndex)
{
    FIntPoint NewGoalLocation = GenerateRandomLocation();
    MoveGoal(AgentIndex, NewGoalLocation);
}

void AMultiAgentCubeEnvironment::AgentReset(int AgentIndex)
{
    FIntPoint NewAgentLocation = GenerateRandomLocation();
    MoveAgent(AgentIndex, NewAgentLocation);
}

FIntPoint AMultiAgentCubeEnvironment::GenerateRandomLocation()
{
    FIntPoint RandomPoint;
    do
    {
        RandomPoint.X = FMath::RandRange(0, GridCenterPoints.Num() - 1);
        RandomPoint.Y = FMath::RandRange(0, GridCenterPoints[0].Num() - 1);
    } while (UsedLocations.Contains(RandomPoint));
    return RandomPoint;
}

void AMultiAgentCubeEnvironment::PostStep()
{
    CurrentStep += 1;
}

FState AMultiAgentCubeEnvironment::State()
{
    FState CurrentState;

    for (int i = 0; i < CurrentAgents; ++i) {
        CurrentState.Values += AgentGetState(i);
    }

    return CurrentState;
}

bool AMultiAgentCubeEnvironment::Done()
{
    bool allGoals = true;
    for (int i = 0; i < CurrentAgents; ++i) {
        if (AgentOutOfBounds(i) || AgentCollidedWithAgent(i) || AgentWrongGoal(i)) {
            return true;
        }
    }

    return false;
}

bool AMultiAgentCubeEnvironment::Trunc()
{
    if (CurrentStep > MaxSteps) {
        CurrentStep = 0;
        return true;
    } 

    return false;
}

float AMultiAgentCubeEnvironment::Reward()
{
    float totalrewards = 0.0;

    for (int i = 0; i < CurrentAgents; ++i){
        float rewards = 0.0;
        float normalized_distance = std::clamp(GridDistance(AgentGoalPositions[i].Key, AgentGoalPositions[i].Value) / (sqrtf(2.0) * static_cast<float>(GridSize)), 0.0f, 1.0f);
        rewards += (AgentOutOfBounds(i) || AgentCollidedWithAgent(i) || AgentWrongGoal(i)) ? -1.0 : 0.0;
        rewards += AgentGoalReached(i) ? 1.0 : map(-normalized_distance, -1.0, 0.0, -1.0, 1.0);
        totalrewards += rewards;
        // totalrewards += (AgentOutOfBounds(i) || AgentCollidedWithAgent(i) || AgentWrongGoal(i)) ? -1.0 : AgentGoalReached(i) ? 1.0 : -0.0001;
    }
 
    return totalrewards / static_cast<float>(CurrentAgents);
}

void AMultiAgentCubeEnvironment::PostTransition() {

}

void AMultiAgentCubeEnvironment::setCurrentAgents(int NumAgents) {
    CurrentAgents = NumAgents;
}

float AMultiAgentCubeEnvironment::map(float x, float in_min, float in_max, float out_min, float out_max) {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

AStaticMeshActor* AMultiAgentCubeEnvironment::InitializeCube()
{
    UStaticMesh* CubeMesh = LoadObject<UStaticMesh>(nullptr, TEXT("StaticMesh'/Engine/BasicShapes/Cube.Cube'"));
    AStaticMeshActor* NewCube = GetWorld()->SpawnActor<AStaticMeshActor>(FVector::ZeroVector, FRotator::ZeroRotator);

    if (NewCube)
    {
        NewCube->SetMobility(EComponentMobility::Movable);
        NewCube->GetStaticMeshComponent()->SetStaticMesh(CubeMesh);
        NewCube->GetStaticMeshComponent()->SetWorldScale3D(MultiAgentCubeParams->ControlledCubeSize);

        UMaterial* MetalMaterial = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/StarterContent/Materials/M_Metal_Steel.M_Metal_Steel'"));
        NewCube->GetStaticMeshComponent()->SetMaterial(0, MetalMaterial);
    }

    return NewCube;
}

AStaticMeshActor* AMultiAgentCubeEnvironment::InitializeGoalObject()
{
    UStaticMesh* SphereMesh = LoadObject<UStaticMesh>(nullptr, TEXT("StaticMesh'/Engine/BasicShapes/Sphere.Sphere'"));
    AStaticMeshActor* NewGoalObject = GetWorld()->SpawnActor<AStaticMeshActor>(FVector::ZeroVector, FRotator::ZeroRotator);

    if (NewGoalObject)
    {
        NewGoalObject->SetMobility(EComponentMobility::Movable);
        NewGoalObject->GetStaticMeshComponent()->SetStaticMesh(SphereMesh);
        NewGoalObject->GetStaticMeshComponent()->SetWorldScale3D(MultiAgentCubeParams->ControlledCubeSize);

        UMaterial* GoldMaterial = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/StarterContent/Materials/M_Metal_Gold.M_Metal_Gold'"));
        NewGoalObject->GetStaticMeshComponent()->SetMaterial(0, GoldMaterial); // M_Basic_Floor
    }

    return NewGoalObject;
}

AStaticMeshActor* AMultiAgentCubeEnvironment::InitializeCube(const FLinearColor& Color)
{
    UStaticMesh* CubeMesh = LoadObject<UStaticMesh>(nullptr, TEXT("StaticMesh'/Engine/BasicShapes/Cube.Cube'"));
    AStaticMeshActor* NewCube = GetWorld()->SpawnActor<AStaticMeshActor>(FVector::ZeroVector, FRotator::ZeroRotator);

    if (NewCube)
    {
        NewCube->SetMobility(EComponentMobility::Movable);
        NewCube->GetStaticMeshComponent()->SetStaticMesh(CubeMesh);
        NewCube->GetStaticMeshComponent()->SetWorldScale3D(MultiAgentCubeParams->ControlledCubeSize);
        UMaterial* ColoredMaterial = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/StarterContent/Materials/M_Basic_Floor.M_Basic_Floor'"));
        NewCube->GetStaticMeshComponent()->SetMaterial(0, ColoredMaterial);

        // Create a dynamic material instance to set the color
        UMaterialInstanceDynamic* DynMaterial = UMaterialInstanceDynamic::Create(NewCube->GetStaticMeshComponent()->GetMaterial(0), NewCube);
        DynMaterial->SetVectorParameterValue("Color", Color);
        NewCube->GetStaticMeshComponent()->SetMaterial(0, DynMaterial);
    }

    return NewCube;
}

AStaticMeshActor* AMultiAgentCubeEnvironment::InitializeGoalObject(const FLinearColor& Color)
{
    UStaticMesh* SphereMesh = LoadObject<UStaticMesh>(nullptr, TEXT("StaticMesh'/Engine/BasicShapes/Sphere.Sphere'"));
    AStaticMeshActor* NewGoalObject = GetWorld()->SpawnActor<AStaticMeshActor>(FVector::ZeroVector, FRotator::ZeroRotator);

    if (NewGoalObject)
    {
        NewGoalObject->SetMobility(EComponentMobility::Movable);
        NewGoalObject->GetStaticMeshComponent()->SetStaticMesh(SphereMesh);
        NewGoalObject->GetStaticMeshComponent()->SetWorldScale3D(MultiAgentCubeParams->ControlledCubeSize);
        UMaterial* ColoredMaterial = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/StarterContent/Materials/M_Basic_Floor.M_Basic_Floor'"));
        NewGoalObject->GetStaticMeshComponent()->SetMaterial(0, ColoredMaterial);

        // Create a dynamic material instance to set the color
        UMaterialInstanceDynamic* DynMaterial = UMaterialInstanceDynamic::Create(NewGoalObject->GetStaticMeshComponent()->GetMaterial(0), NewGoalObject);
        DynMaterial->SetVectorParameterValue("Color", Color);
        NewGoalObject->GetStaticMeshComponent()->SetMaterial(0, DynMaterial);
    }

    return NewGoalObject;
}