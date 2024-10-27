#include "TerraShift/GridObjectManager.h"
#include "TimerManager.h"
#include "Engine/World.h"
#include "TerraShift/GridObject.h"
#include "Engine/Engine.h" // For logging

// Constructor
AGridObjectManager::AGridObjectManager() {
    PrimaryActorTick.bCanEverTick = false;
    RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("GridObjectManagerRoot"));

    // Initialize ObjectSize to a default value
    ObjectSize = FVector(0.1f, 0.1f, 0.1f);
}

// Set the size of GridObjects to be spawned
void AGridObjectManager::SetObjectSize(FVector InObjectSize) {
    ObjectSize = InObjectSize;
}

// Spawns GridObjects at the given locations with the specified size and delay
void AGridObjectManager::SpawnGridObjects(const TArray<FVector>& Locations, FVector InObjectSize, float SpawnDelay) {
    // Set the object size
    ObjectSize = InObjectSize;

    for (int i = 0; i < Locations.Num(); i++) {
        float TotalSpawnDelay = SpawnDelay * static_cast<float>(i);

        // Schedule the spawn with a delay
        if (UWorld* World = GetWorld()) {
            FTimerHandle TimerHandle;
            World->GetTimerManager().SetTimer(
                TimerHandle,
                [this, i, Location = Locations[i]]() {
                    SpawnGridObjectAtIndex(i, Location);
                },
                TotalSpawnDelay,
                false
            );
        }
    }
}

// Spawns a GridObject at a specific index and location
void AGridObjectManager::SpawnGridObjectAtIndex(int32 Index, FVector InLocation) {
    if (UWorld* World = GetWorld()) {
        AGridObject* NewGridObject = World->SpawnActor<AGridObject>(AGridObject::StaticClass(), FVector::ZeroVector, FRotator::ZeroRotator);
        if (NewGridObject) {
            NewGridObject->InitializeGridObject(ObjectSize);
            NewGridObject->AttachToComponent(RootComponent, FAttachmentTransformRules::KeepRelativeTransform);

            // Expand if adding new
            if (GridObjects.Num() <= Index) {
                GridObjects.SetNum(Index + 1);
            }
            GridObjects[Index] = NewGridObject;

            // Adjust location to ensure the GridObject is above the Grid
            FVector Location = InLocation;
            FBoxSphereBounds GridObjectBounds = NewGridObject->MeshComponent->CalcLocalBounds();
            FVector LocalOffsets = GridObjectBounds.BoxExtent * NewGridObject->MeshComponent->GetRelativeScale3D();
            Location.Z += LocalOffsets.Z * 2; // So GridObjects dont spawn in each other

            NewGridObject->SetActorRelativeLocation(Location);
            NewGridObject->SetGridObjectActive(true);

            // Notify that a GridObject has been spawned
            OnGridObjectSpawned.Broadcast(Index, NewGridObject);
        }
    }
}

// Resets all GridObjects
void AGridObjectManager::ResetGridObjects() {
    for (AGridObject* GridObject : GridObjects) {
        if (GridObject) {
            GridObject->SetGridObjectActive(false);
            GridObject->SetActorRelativeLocation(FVector::ZeroVector);
        }
    }
}

// Gets the world location of a GridObject at a given index
FVector AGridObjectManager::GetGridObjectWorldLocation(int32 Index) const {
    if (GridObjects.IsValidIndex(Index) && GridObjects[Index]) {
        return GridObjects[Index]->GetActorLocation();
    }
    return FVector::ZeroVector;
}

// Retrieves active grid objects
TArray<AGridObject*> AGridObjectManager::GetActiveGridObjects() const {
    TArray<AGridObject*> ActiveObjects;
    for (AGridObject* GridObject : GridObjects) {
        if (GridObject && GridObject->IsActive()) {
            ActiveObjects.Add(GridObject);
        }
    }
    return ActiveObjects;
}

// Retrieves active columns in proximity to grid objects
TSet<int32> AGridObjectManager::GetActiveColumnsInProximity(int32 GridSize, const TArray<FVector>& ColumnCenters, const FVector& PlatformCenter, float PlatformSize, float CellSize) const {
    TSet<int32> ActiveColumnIndices;

    float HalfPlatformSize = PlatformSize / 2.0f;
    float GridOriginX = PlatformCenter.X - HalfPlatformSize;
    float GridOriginY = PlatformCenter.Y - HalfPlatformSize;

    // Iterate over each GridObject
    for (AGridObject* GridObject : GridObjects) {
        if (!GridObject || !GridObject->IsActive()) {
            continue; // Skip inactive or null objects
        }

        // Get the GridObject's location in world space
        FVector ObjectLocation = GridObject->MeshComponent->GetComponentLocation();

        // Get the GridObject's bounds in world space
        FBoxSphereBounds ObjectBoundsWorld = GridObject->MeshComponent->CalcBounds(
            GridObject->MeshComponent->GetComponentTransform()
        );

        // Calculate the effective radius in world units
        float EffectiveRadius = ObjectBoundsWorld.SphereRadius;

        // Calculate the min and max world coordinates
        float MinX = ObjectLocation.X - EffectiveRadius;
        float MaxX = ObjectLocation.X + EffectiveRadius;
        float MinY = ObjectLocation.Y - EffectiveRadius;
        float MaxY = ObjectLocation.Y + EffectiveRadius;

        // Map world coordinates to grid indices
        int32 MinXIndex = FMath::FloorToInt((MinX - GridOriginX) / CellSize);
        int32 MaxXIndex = FMath::FloorToInt((MaxX - GridOriginX) / CellSize);
        int32 MinYIndex = FMath::FloorToInt((MinY - GridOriginY) / CellSize);
        int32 MaxYIndex = FMath::FloorToInt((MaxY - GridOriginY) / CellSize);

        // Clamp indices to grid bounds
        MinXIndex = FMath::Clamp(MinXIndex, 0, GridSize - 1);
        MaxXIndex = FMath::Clamp(MaxXIndex, 0, GridSize - 1);
        MinYIndex = FMath::Clamp(MinYIndex, 0, GridSize - 1);
        MaxYIndex = FMath::Clamp(MaxYIndex, 0, GridSize - 1);

        // Iterate over the indices
        for (int32 X = MinXIndex; X <= MaxXIndex; ++X) {
            for (int32 Y = MinYIndex; Y <= MaxYIndex; ++Y) {
                int32 ColumnIndex = X * GridSize + Y;
                FVector ColumnCenter = ColumnCenters[ColumnIndex];

                // Check if the column center is within the effective radius
                if (FVector::Dist2D(ColumnCenter, ObjectLocation) <= EffectiveRadius) {
                    ActiveColumnIndices.Add(ColumnIndex);
                }
            }
        }
    }

    return ActiveColumnIndices;
}

// Helper function to find the closest column index to the given location
int32 AGridObjectManager::FindClosestColumnIndex(const FVector& Location, const TArray<FVector>& ColumnCenters) const {
    int32 ClosestIndex = -1;
    float MinDistance = FLT_MAX;

    for (int32 i = 0; i < ColumnCenters.Num(); ++i) {
        float Distance = FVector::Dist2D(Location, ColumnCenters[i]);
        if (Distance < MinDistance) {
            MinDistance = Distance;
            ClosestIndex = i;
        }
    }

    return ClosestIndex;
}

// Helper function to convert 1D index to 2D grid coordinates
FIntPoint AGridObjectManager::Get2DIndexFrom1D(int32 Index, int32 GridSize) const {
    return FIntPoint(Index / GridSize, Index % GridSize);
}

// Deactivates a GridObject at a specific index
void AGridObjectManager::DeactivateGridObject(int32 Index) {
    if (GridObjects.IsValidIndex(Index)) {
        AGridObject* GridObject = GridObjects[Index];
        if (GridObject) {
            GridObject->SetGridObjectActive(false);
        }
    }
}

// Deletes a GridObject at a specific index
void AGridObjectManager::DeleteGridObject(int32 Index) {
    if (GridObjects.IsValidIndex(Index)) {
        AGridObject* GridObject = GridObjects[Index];
        if (GridObject) {
            GridObject->Destroy();
            GridObjects[Index] = nullptr;
        }
    }
}

// Respawns a GridObject at a specific index and location after a delay
void AGridObjectManager::RespawnGridObjectAtLocation(int32 Index, FVector InLocation, float SpawnDelay) {
    if (UWorld* World = GetWorld()) {
        FTimerHandle TimerHandle;
        World->GetTimerManager().SetTimer(
            TimerHandle,
            [this, Index, InLocation]() {
                SpawnGridObjectAtIndex(Index, InLocation);
            },
            SpawnDelay,
            false
        );
    }
}

// Get a GridObject by index
AGridObject* AGridObjectManager::GetGridObject(int32 Index) const {
    if (GridObjects.IsValidIndex(Index)) {
        return GridObjects[Index];
    }
    return nullptr;
}
