#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "TerraShift/GridObject.h"
#include "Engine/World.h"
#include "TimerManager.h"
#include "Kismet/KismetMathLibrary.h"
#include "Engine/Engine.h"
#include "UObject/NameTypes.h"
#include "GridObjectManager.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnGridObjectSpawned, int32, Index, AGridObject*, NewGridObject);

UCLASS()
class UNREALRLLABS_API AGridObjectManager : public AActor {
    GENERATED_BODY()

public:
    AGridObjectManager();

    /** Sets the platform actor reference. */
    void SetPlatformActor(AActor* InPlatform);

    /** Spawns grid objects at the provided locations with an optional spawn delay. */
    void SpawnGridObjects(const TArray<FVector>& Locations, FVector InObjectSize, float InObjectMass, float SpawnDelay);

    /** Resets all managed grid objects to their inactive state. */
    void ResetGridObjects();

    /** Returns the world location of the grid object at Index. */
    FVector GetGridObjectWorldLocation(int32 Index) const;

    /** Returns the set of column indices that overlap any active grid object. */
    TSet<int32> GetActiveColumnsInProximity(int32 GridSize, const TArray<FVector>& ColumnCenters, const FVector& PlatformCenter, float PlatformSize, float CellSize) const;

    /** Deactivates the grid object at Index. */
    void DisableGridObject(int32 Index);

    /** Returns the grid object at Index, if any. */
    AGridObject* GetGridObject(int32 Index) const;

    /** Spawns or reuses a grid object at Index and places it at the requested location. */
    void SpawnGridObjectAtIndex(int32 Index, FVector InWorldLocation, FVector InObjectSize, float InObjectMass);
    
    UPROPERTY(BlueprintAssignable, Category = "Events")
    FOnGridObjectSpawned OnGridObjectSpawned;

    /** Returns the folder path used for spawned actors. */
    FName GetActorFolderPath() const;

private:
    UPROPERTY()
    TArray<AGridObject*> GridObjects;

    FVector ObjectSize;

    float ObjectMass;

    AActor* PlatformActor;

    /** Helper that spawns and configures a new grid object. */
    AGridObject* CreateNewGridObjectAtIndex(int32 Index, FVector InObjectSize, float InObjectMass);
};
