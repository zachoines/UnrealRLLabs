#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "TerraShift/GridObject.h"
#include "Engine/World.h"
#include "TimerManager.h"
#include "Kismet/KismetMathLibrary.h"
#include "Engine/Engine.h"
#include "UObject/NameTypes.h" // For FName
#include "GridObjectManager.generated.h"

// Declare a delegate for when a GridObject is spawned
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnGridObjectSpawned, int32, Index, AGridObject*, NewGridObject);

UCLASS()
class UNREALRLLABS_API AGridObjectManager : public AActor {
    GENERATED_BODY()

public:
    AGridObjectManager();

    // Set the platform actor reference
    void SetPlatformActor(AActor* InPlatform);

    // Spawn grid objects at specified locations with a spawn delay
    void SpawnGridObjects(const TArray<FVector>& Locations, FVector InObjectSize, float InObjectMass, float SpawnDelay);

    // Reset all grid objects to their inactive states
    void ResetGridObjects();

    // Get the world location of a specified grid object
    FVector GetGridObjectWorldLocation(int32 Index) const;

    // Get active columns in proximity to grid objects
    TSet<int32> GetActiveColumnsInProximity(int32 GridSize, const TArray<FVector>& ColumnCenters, const FVector& PlatformCenter, float PlatformSize, float CellSize) const;

    // Deactivate a GridObject at a specific index
    void DisableGridObject(int32 Index);

    // Get a GridObject by index
    AGridObject* GetGridObject(int32 Index) const;

    // Spawns or reuses a GridObject at the specified index and location
    void SpawnGridObjectAtIndex(int32 Index, FVector InWorldLocation, FVector InObjectSize, float InObjectMass);
    
    // Event triggered when a GridObject is spawned
    UPROPERTY(BlueprintAssignable, Category = "Events")
    FOnGridObjectSpawned OnGridObjectSpawned;

    // Get the actor's folder path
    FName GetActorFolderPath() const;

private:

    // Array of grid objects managed by this class
    UPROPERTY()
    TArray<AGridObject*> GridObjects;

    // Size of GridObjects to be spawned
    FVector ObjectSize;

    // Mass of GridObjects to be spawned
    float ObjectMass;

    // The platform actor reference
    AActor* PlatformActor;

    // Helper function to create a new GridObject
    AGridObject* CreateNewGridObjectAtIndex(int32 Index, FVector InObjectSize, float InObjectMass);
};