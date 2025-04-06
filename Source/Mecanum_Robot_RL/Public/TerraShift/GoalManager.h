#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "EnvironmentConfig.h"
#include "GoalManager.generated.h"

/**
 * Simple struct for one goal definition:
 *  - Actor reference
 *  - Offset (relative to actor’s center)
 *  - Radius for "in-range" checks
 */
USTRUCT(BlueprintType)
struct FGoalData
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    AActor* GoalActor;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FVector Offset;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Radius;

    FGoalData()
        : GoalActor(nullptr)
        , Offset(FVector::ZeroVector)
        , Radius(100.f)
    {}
};

/**
 * AGoalManager:
 *  - Tracks a list of goal entries (Actor + offset + radius).
 *  - Exposes an API to get each goal's world location and check radius membership.
 *  - Can initialize from config (for default radius, etc.),
 *    but does NOT store color or spawn random actors.
 *    (That logic can be in the environment or StateManager.)
 */
UCLASS()
class UNREALRLLABS_API AGoalManager : public AActor
{
    GENERATED_BODY()

public:
    AGoalManager();

    /**
     * Initialize from a "GoalManager" sub-config in JSON (logging or storing
     * any relevant defaults, e.g. a default radius or other toggles).
     */
    UFUNCTION(BlueprintCallable, Category = "GoalManager")
    void InitializeFromConfig(UEnvironmentConfig* GMConfig);

    /**
     * Reset goals to a new set of (Actor, Offset, Radius).
     * This overwrites the internal array with the new data.
     *
     * @param InGoalActors  => array of AActors that define each goal's center
     * @param InOffsets     => array of offsets (same length as InGoalActors)
     * @param InRadia       => array of radii (same length as InGoalActors)
     */
    UFUNCTION(BlueprintCallable, Category = "GoalManager")
    void ResetGoals(const TArray<AActor*>& InGoalActors,
        const TArray<FVector>& InOffsets);

    /**
     * Returns the world location for the given goal index = ActorLocation + Offset.
     * Returns FVector::ZeroVector if index invalid or GoalActor is null.
     */
    UFUNCTION(BlueprintCallable, Category = "GoalManager")
    FVector GetGoalLocation(int32 GoalIndex) const;

    /**
     * Returns true if the given Location is within the radius
     * of the specified goal (computed from Actor’s location + offset).
     */
    UFUNCTION(BlueprintCallable, Category = "GoalManager")
    bool IsInRadiusOf(int32 GoalIndex, const FVector& Location, float Radius) const;

protected:
    virtual void BeginPlay() override;

private:
    /** Possibly a default radius or other config we read from JSON. */
    float DefaultRadius;

    /** The array of goal data. */
    UPROPERTY()
    TArray<FGoalData> Goals;
};
