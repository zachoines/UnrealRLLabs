#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "TerraShift/GridObject.h"
#include "Engine/World.h"
#include "TimerManager.h"
#include "Kismet/KismetMathLibrary.h"
#include "GridObjectManager.generated.h"

// Declare a delegate for when a GridObject is spawned
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnGridObjectSpawned, int32, Index, AGridObject*, NewGridObject);

UCLASS()
class UNREALRLLABS_API AGridObjectManager : public AActor {
    GENERATED_BODY()

public:
    AGridObjectManager();

    // Spawn grid objects at specified locations with a spawn delay
    void SpawnGridObjects(const TArray<FVector>& Locations, FVector ObjectSize, float SpawnDelay);

    // Reset all grid objects to their inactive states
    void ResetGridObjects();

    // Get the world location of a specified grid object
    FVector GetGridObjectWorldLocation(int32 Index) const;

    // Get active grid objects
    TArray<AGridObject*> GetActiveGridObjects() const;

    // Get active columns in proximity to grid objects
    TSet<int32> GetActiveColumnsInProximity(int32 GridSize, const TArray<FVector>& ColumnCenters, const FVector& PlatformCenter, float PlatformSize, float CellSize) const;

    // Deactivate a GridObject at a specific index
    void DeactivateGridObject(int32 Index);

    // Delete a GridObject at a specific index
    void DeleteGridObject(int32 Index);

    // Respawn a GridObject at a specific index and location after a delay
    void RespawnGridObjectAtLocation(int32 Index, FVector InLocation, float SpawnDelay);

    // Get a GridObject by index
    AGridObject* GetGridObject(int32 Index) const;

    // Event triggered when a GridObject is spawned
    UPROPERTY(BlueprintAssignable, Category = "Events")
    FOnGridObjectSpawned OnGridObjectSpawned;

    // Set the size of GridObjects to be spawned
    void SetObjectSize(FVector InObjectSize);

private:
    // Array of grid objects managed by this class
    UPROPERTY()
    TArray<AGridObject*> GridObjects;

    // Size of GridObjects to be spawned
    FVector ObjectSize;

    // Spawn a single grid object at the specified index and location
    void SpawnGridObjectAtIndex(int32 Index, FVector InLocation);

    // Helper function to find the closest column index to the given location
    int32 FindClosestColumnIndex(const FVector& Location, const TArray<FVector>& ColumnCenters) const;

    // Helper function to convert 1D index to 2D grid coordinates
    FIntPoint Get2DIndexFrom1D(int32 Index, int32 GridSize) const;
};
