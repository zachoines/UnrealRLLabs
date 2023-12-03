// Fill out your copyright notice in the Description page of Project Settings.


#include "MultiAgentCubeEnvironment.h"

AMultiAgentCubeEnvironment::AMultiAgentCubeEnvironment()
{
    // Setup Env Info
    EnvInfo.EnvID = 2;
    EnvInfo.MaxAgents = 5;
    EnvInfo.SingleAgentObsSize = (EnvInfo.MaxAgents) * (EnvInfo.MaxAgents);
    EnvInfo.StateSize = EnvInfo.MaxAgents * EnvInfo.SingleAgentObsSize;
    EnvInfo.IsMultiAgent = true;

    const TArray<FContinuousActionSpec>& ContinuousActions = {
        // Your continuous actions initialization
    };
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
    int GridSize = EnvInfo.MaxAgents;
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
        NewGoalObject->GetStaticMeshComponent()->SetMaterial(0, GoldMaterial);
    }

    return NewGoalObject;
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

        // Initialize goals
        FIntPoint GoalLocationIndex = GenerateRandomLocation();
        FVector GoalWorldLocation = GetWorldLocationFromGridIndex(GoalLocationIndex);
        GoalObjects[i]->SetActorLocation(GoalWorldLocation);
        ActorToLocationMap.Add(GoalObjects[i], GoalLocationIndex);

        // Store in other maps
        UsedLocations.FindOrAdd(CubeLocationIndex).Add(ControlledCubes[i]);
        UsedLocations.FindOrAdd(GoalLocationIndex).Add(GoalObjects[i]);
        AgentGoalPositions.Add(i, TPair<FIntPoint, FIntPoint>(CubeLocationIndex, GoalLocationIndex));
    }
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

bool AMultiAgentCubeEnvironment::AgentHasCollided(int AgentIndex)
{
    if (!AgentGoalPositions.Contains(AgentIndex)) {
        UE_LOG(LogTemp, Log, TEXT("Undefined Agent Index: %"), AgentIndex);
        return false;
    }

    FIntPoint AgentLocation = AgentGoalPositions[AgentIndex].Key;
    TArray<AStaticMeshActor*>* ActorsAtLocation = UsedLocations.Find(AgentLocation);

    if (!ActorsAtLocation || ActorsAtLocation->Num() <= 1)
    {
        return false; // No collision if no or only one actor is at this location
    }

    for (AStaticMeshActor* Actor : *ActorsAtLocation)
    {
        if (Actor != ControlledCubes[AgentIndex]) // Exclude the agent itself
        {
            FIntPoint ActorLocation = ActorToLocationMap.Contains(Actor) ? ActorToLocationMap[Actor] : FIntPoint(-1, -1);

            // Check if the Actor is another agent or another agent's goal
            if (ActorLocation != FIntPoint(-1, -1) && ActorLocation != AgentGoalPositions[AgentIndex].Value)
            {
                return true; // Collision detected with another agent or another agent's goal
            }
        }
    }

    return false; // No collision found
}

TArray<float> AMultiAgentCubeEnvironment::AgentGetState(int AgentIndex, bool UseVisibilityRange)
{
    TArray<float> State;
    if (!AgentGoalPositions.Contains(AgentIndex)) {
        UE_LOG(LogTemp, Log, TEXT("Undefined Agent Index: %d"), AgentIndex);
        return State;
    }

    FIntPoint AgentLocation = AgentGoalPositions[AgentIndex].Key;
    FIntPoint GoalLocation = AgentGoalPositions[AgentIndex].Value;

    int GridSize = GridCenterPoints.Num();
    for (int i = 0; i < GridSize; ++i)
    {
        for (int j = 0; j < GridSize; ++j)
        {
            if (!UseVisibilityRange || (FMath::Abs(i - AgentLocation.X) <= VisibilityRange && FMath::Abs(j - AgentLocation.Y) <= VisibilityRange))
            {
                FIntPoint CurrentLocation(i, j);
                if (CurrentLocation == AgentLocation)
                {
                    State.Add(0.0f); // Itself
                }
                else if (CurrentLocation == GoalLocation)
                {
                    State.Add(1.0f / 6.0f); // Its goal
                }
                else if (UsedLocations.Contains(CurrentLocation))
                {
                    // Check if it's another agent or goal
                    bool isOtherAgent = false;
                    bool isOtherGoal = false;
                    for (const auto& Actor : UsedLocations[CurrentLocation])
                    {
                        if (Actor != ControlledCubes[AgentIndex] && Actor != GoalObjects[AgentIndex])
                        {
                            isOtherAgent = isOtherAgent || ControlledCubes.Contains(Actor);
                            isOtherGoal = isOtherGoal || GoalObjects.Contains(Actor);
                        }
                    }

                    if (isOtherAgent)
                    {
                        State.Add(2.0f / 6.0f);
                    }
                    else if (isOtherGoal)
                    {
                        State.Add(3.0f / 6.0f);
                    }
                    else
                    {
                        State.Add(4.0f / 6.0f); // Generic obstacle
                    }
                }
                else
                {
                    State.Add(5.0f / 6.0f); // Free space
                }
            }
            else
            {
                State.Add(1.0f); // Beyond visibility range
            }
        }
    }

    return State;
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
    else 
    {
        return true;
    }
}

FVector AMultiAgentCubeEnvironment::GetWorldLocationFromGridIndex(FIntPoint GridIndex)
{
    if (GridIndex.X >= 0 && GridIndex.X < GridCenterPoints.Num() && GridIndex.Y >= 0 && GridIndex.Y < GridCenterPoints[0].Num())
    {
        FVector WorldLocation = GridCenterPoints[GridIndex.X][GridIndex.Y];
        WorldLocation.Z += CubeSize.Z; // Adjust to be flush on ground plane
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
    while (ControlledCubes.Num() < NumAgents)
    {
        AStaticMeshActor* NewCube = InitializeCube();
        ControlledCubes.Add(NewCube);

        AStaticMeshActor* NewGoal = InitializeGoalObject();
        GoalObjects.Add(NewGoal);
    }

    // Update the current number of agents
    CurrentAgents = NumAgents;

    // Now that all agents are correctly set up, assign them new locations
    AssignRandomGridLocations();

    // Always make sure after modifying actors
    Update();
    return State();
}


void AMultiAgentCubeEnvironment::Act(FAction Action)
{
    if (Action.Values.Num() != ControlledCubes.Num())
    {
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
            continue;
        }

        MoveAgent(i, NewLocation);
    }
}

void AMultiAgentCubeEnvironment::AgentGoalReset(int AgentIndex)
{
    if (AgentIndex < 0 || AgentIndex >= ControlledCubes.Num())
    {
        return; // Check for valid AgentIndex
    }

    // Remove the current agent and goal from UsedLocations and ActorToLocationMap
    FIntPoint CurrentAgentLocation = AgentGoalPositions[AgentIndex].Key;
    FIntPoint CurrentGoalLocation = AgentGoalPositions[AgentIndex].Value;
    UsedLocations[CurrentAgentLocation].Remove(ControlledCubes[AgentIndex]);
    UsedLocations[CurrentGoalLocation].Remove(GoalObjects[AgentIndex]);
    ActorToLocationMap.Remove(ControlledCubes[AgentIndex]);
    ActorToLocationMap.Remove(GoalObjects[AgentIndex]);

    // Reset Agent Location
    FIntPoint NewAgentLocation = GenerateRandomLocation();
    FVector NewAgentWorldLocation = GetWorldLocationFromGridIndex(NewAgentLocation);
    ControlledCubes[AgentIndex]->SetActorLocation(NewAgentWorldLocation);
    UsedLocations.FindOrAdd(NewAgentLocation).Add(ControlledCubes[AgentIndex]);
    ActorToLocationMap.Add(ControlledCubes[AgentIndex], NewAgentLocation);

    // Reset Goal Location
    FIntPoint NewGoalLocation;
    do {
        NewGoalLocation = GenerateRandomLocation();
    } while (NewGoalLocation == NewAgentLocation); // Ensure goal is not placed at the agent's location

    FVector NewGoalWorldLocation = GetWorldLocationFromGridIndex(NewGoalLocation);
    GoalObjects[AgentIndex]->SetActorLocation(NewGoalWorldLocation);
    UsedLocations.FindOrAdd(NewGoalLocation).Add(GoalObjects[AgentIndex]);
    ActorToLocationMap.Add(GoalObjects[AgentIndex], NewGoalLocation);

    // Update AgentGoalPositions
    AgentGoalPositions[AgentIndex] = TPair<FIntPoint, FIntPoint>(NewAgentLocation, NewGoalLocation);
}


void AMultiAgentCubeEnvironment::Update()
{
    CurrentStep += 1;
}

FState AMultiAgentCubeEnvironment::State()
{
    FState CurrentState;
    int index = 0;
    for (int i = 0; i < CurrentAgents; ++i)
    {
        CurrentState.Values += AgentGetState(i);
    }

    return CurrentState;
}

bool AMultiAgentCubeEnvironment::Done()
{
    for (int i = 0; i < CurrentAgents; ++i)
    {
        if (AgentOutOfBounds(i)) {
            return true;
        }

        if (AgentHasCollided(i)) {
            return true;
        }
    }

    return false;
}

bool AMultiAgentCubeEnvironment::Trunc()
{
    if (CurrentStep >= maxStepsPerEpisode) {
        return true;
    }

    return false;
}

float AMultiAgentCubeEnvironment::Reward()
{
    int rewards = 0;
    for (int i = 0; i < CurrentAgents; ++i)
    {
        // rewards += -.1;

        if (AgentOutOfBounds(i)) {
            rewards += -1.0;
        } else if (AgentHasCollided(i)) {
            rewards += -1.0;
        } else if (AgentGoalReached(i)) {
            rewards += 2.0;
            AgentGoalReset(i);
        }
    }

    return rewards;
}

void AMultiAgentCubeEnvironment::setCurrentAgents(int NumAgents)
{
    CurrentAgents = NumAgents;
}