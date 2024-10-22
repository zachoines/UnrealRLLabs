#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "TerraShift/GridObject.h"
#include "Engine/World.h"
#include "TimerManager.h"
#include "Kismet/KismetMathLibrary.h"
#include "GridObjectManager.generated.h"

UCLASS()
class UNREALRLLABS_API AGridObjectManager : public AActor {
    GENERATED_BODY()

public:
    AGridObjectManager();

    // Spawn grid objects at specified locations
    void SpawnGridObjects(const TArray<FVector>& Locations, FVector ObjectSize, float SpawnDelay);

    // Reset all grid objects to their inactive states
    void ResetGridObjects();

    // Get the world location of a specified grid object
    FVector GetGridObjectWorldLocation(int32 Index) const;

    // Get the current grid object locations as FVector2D (grid space)
    TArray<FVector2D> GetGridObjectLocations() const;

    // Get the current grid object world locations
    TSet<int32> GetActiveColumnsInProximity(int32 GridSize, const TArray<FVector>& ColumnCenters, const FVector& PlatformCenter, float PlatformSize, float CellSize) const;

private:
    // Array of grid objects managed by this class
    UPROPERTY()
    TArray<AGridObject*> GridObjects;

    // Spawn a single grid object at the specified location
    void SpawnGridObjectAtLocation(FVector InLocation, FVector ObjectSize, float SpawnDelay);

    // Helper function to find the closest column index to the given location
    int32 FindClosestColumnIndex(const FVector& Location, const TArray<FVector>& ColumnCenters) const;

    // Helper function to convert 1D index to 2D grid coordinates
    FIntPoint Get2DIndexFrom1D(int32 Index, int32 GridSize) const;
};
