#include "TerraShift/GoalManager.h"
#include "Engine/World.h"
#include "Kismet/KismetMathLibrary.h"

AGoalManager::AGoalManager()
{
    PrimaryActorTick.bCanEverTick = false;
    DefaultRadius = 1.f;
}

void AGoalManager::BeginPlay()
{
    Super::BeginPlay();
}

void AGoalManager::InitializeFromConfig(UEnvironmentConfig* GMConfig)
{
    if (!GMConfig || !GMConfig->IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("GoalManager::InitializeFromConfig => null or invalid config!"));
        return;
    }

    // Example: read a "DefaultRadius" param from the JSON.
    if (GMConfig->HasPath(TEXT("DefaultRadius")))
    {
        DefaultRadius = GMConfig->Get(TEXT("DefaultRadius"))->AsNumber();
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("GoalManager config missing 'DefaultRadius' => fallback to %f"), DefaultRadius);
    }
}

void AGoalManager::ResetGoals(const TArray<AActor*>& InGoalActors,
    const TArray<FVector>& InOffsets)
{
    // Check that arrays match
    if (InGoalActors.Num() != InOffsets.Num())
    {
        UE_LOG(LogTemp, Error, TEXT("GoalManager::ResetGoals => array size mismatch! Actors=%d, Offsets=%d"),
            InGoalActors.Num(), InOffsets.Num());
        return;
    }

    // Clear out existing
    Goals.Empty();

    // Rebuild
    for (int32 i = 0; i < InGoalActors.Num(); i++)
    {
        FGoalData data;
        data.GoalActor = InGoalActors[i];
        data.Offset = InOffsets[i];

        Goals.Add(data);
    }

    UE_LOG(LogTemp, Log, TEXT("GoalManager => ResetGoals => total goals: %d"), Goals.Num());
}

FVector AGoalManager::GetGoalLocation(int32 GoalIndex) const
{
    if (!Goals.IsValidIndex(GoalIndex))
    {
        UE_LOG(LogTemp, Warning, TEXT("GoalManager::GetGoalLocation => invalid index: %d"), GoalIndex);
        return FVector::ZeroVector;
    }
    const FGoalData& g = Goals[GoalIndex];
    if (!g.GoalActor)
    {
        UE_LOG(LogTemp, Warning, TEXT("GoalManager::GetGoalLocation => null GoalActor at index %d"), GoalIndex);
        return FVector::ZeroVector;
    }

    FVector actorPos = g.GoalActor->GetActorLocation();
    return actorPos + g.Offset;
}

bool AGoalManager::IsInRadiusOf(int32 GoalIndex, const FVector& Location, float Radius) const
{
    if (!Goals.IsValidIndex(GoalIndex))
    {
        UE_LOG(LogTemp, Warning, TEXT("GoalManager::IsInRadiusOf => invalid index: %d"), GoalIndex);
        return false;
    }
    const FGoalData& g = Goals[GoalIndex];
    if (!g.GoalActor)
    {
        UE_LOG(LogTemp, Warning, TEXT("GoalManager::IsInRadiusOf => null GoalActor at index %d"), GoalIndex);
        return false;
    }

    FVector goalWorldLoc = g.GoalActor->GetActorLocation() + g.Offset;
    float dist = FVector::Dist(goalWorldLoc, Location);
    return (dist <= Radius);
}
