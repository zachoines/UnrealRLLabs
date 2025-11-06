#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "EnvironmentConfig.h"
#include "GoalManager.generated.h"

/** Goal definition storing an actor, offset, and radius. */
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

/** Actor that tracks goal definitions and exposes world-space queries. */
UCLASS()
class UNREALRLLABS_API AGoalManager : public AActor
{
    GENERATED_BODY()

public:
    AGoalManager();

    /** Initializes the manager from the GoalManager section of the JSON config. */
    UFUNCTION(BlueprintCallable, Category = "GoalManager")
    void InitializeFromConfig(UEnvironmentConfig* GMConfig);

    /** Replaces the managed goals with the provided actors and offsets. */
    UFUNCTION(BlueprintCallable, Category = "GoalManager")
    void ResetGoals(const TArray<AActor*>& InGoalActors,
        const TArray<FVector>& InOffsets);

    /** Returns the world location of a goal (actor location plus offset). */
    UFUNCTION(BlueprintCallable, Category = "GoalManager")
    FVector GetGoalLocation(int32 GoalIndex) const;

    /** Returns true when Location lies within the specified goal radius. */
    UFUNCTION(BlueprintCallable, Category = "GoalManager")
    bool IsInRadiusOf(int32 GoalIndex, const FVector& Location, float Radius) const;

protected:
    virtual void BeginPlay() override;

private:
    float DefaultRadius;

    UPROPERTY()
    TArray<FGoalData> Goals;
};
