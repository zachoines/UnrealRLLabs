#include "TerraShift/GridObjectManager.h"
#include "TerraShift/GridObject.h"
#include "Engine/World.h"
#include "TimerManager.h"
#include "UObject/NameTypes.h" // For FName
#include "String/LexFromString.h"
#include "Components/PrimitiveComponent.h"

// Constructor
AGridObjectManager::AGridObjectManager() {
    PrimaryActorTick.bCanEverTick = false;

    // Initialize ObjectSize to a default value
    ObjectSize = FVector(0.1f, 0.1f, 0.1f);
    ObjectMass = 0.1f;
    PlatformActor = nullptr;
}

// Set the platform actor reference
void AGridObjectManager::SetPlatformActor(AActor* InPlatform) {
    PlatformActor = InPlatform;
}

// Get the actor's folder path
FName AGridObjectManager::GetActorFolderPath() const {
    return GetFolderPath();
}

// Spawns GridObjects at the given locations with the specified size and delay
void AGridObjectManager::SpawnGridObjects(const TArray<FVector>& Locations, FVector InObjectSize, float InObjectMass, float SpawnDelay) {
    // Set the object size
    ObjectSize = InObjectSize;
    ObjectMass = InObjectMass;

    for (int i = 0; i < Locations.Num(); i++) {
        float TotalSpawnDelay = SpawnDelay * static_cast<float>(i);

        if (TotalSpawnDelay == 0) {
            SpawnGridObjectAtIndex(i, Locations[i], InObjectSize, InObjectMass); // Spawn the first right away
        }
        else {
            // Schedule the spawn with a delay
            if (UWorld* World = GetWorld()) {
                FTimerHandle TimerHandle;
                World->GetTimerManager().SetTimer(
                    TimerHandle,
                    [this, i, Location = Locations[i], InObjectSize, InObjectMass]() {
                        SpawnGridObjectAtIndex(i, Location, InObjectSize, InObjectMass);
                    },
                    TotalSpawnDelay,
                    false
                );
            }
        }
    }
}

// Spawns or reuses a GridObject at a specific index and location
void AGridObjectManager::SpawnGridObjectAtIndex(int32 Index, FVector InWorldLocation, FVector InObjectSize, float InObjectMass) 
{
    AGridObject* GridObject = nullptr;

    if (GridObjects.IsValidIndex(Index) && GridObjects[Index]) {
        GridObject = GridObjects[Index];
    }
    else {
        GridObject = CreateNewGridObjectAtIndex(Index, InObjectSize, InObjectMass);
    }

    if (GridObject) {
        // Tag with environment id if available from the platform actor
        auto GetEnvIdFromActorTags = [](const AActor* Actor) -> int32
        {
            if (!Actor) return -1;
            for (const FName& Tag : Actor->Tags)
            {
                const FString S = Tag.ToString();
                if (S.StartsWith(TEXT("EnvId=")))
                {
                    int32 V = -1; LexFromString(V, *S.RightChop(6)); return V;
                }
            }
            return -1;
        };
        const int32 EnvId = GetEnvIdFromActorTags(PlatformActor);
        if (EnvId >= 0 && GridObject->MeshComponent)
        {
            const FName Tag(*FString::Printf(TEXT("EnvId=%d"), EnvId));
            if (!GridObject->MeshComponent->ComponentTags.Contains(Tag))
            {
                GridObject->MeshComponent->ComponentTags.Add(Tag);
            }
        }
        // Ensure visual scale matches requested size every spawn (for reuse cases)
        if (GridObject->MeshComponent)
        {
            GridObject->MeshComponent->SetWorldScale3D(InObjectSize);
            GridObject->MeshComponent->UpdateBounds();
        }
        // Adjust the GridObject's location to ensure it's above the grid
        FVector Location = InWorldLocation;

        // Reset and activate the GridObject
        GridObject->ResetGridObject();

        // Ensure bounds are up-to-date after any scale changes
        if (GridObject->MeshComponent)
        {
            GridObject->MeshComponent->UpdateBounds();
        }

        // Place directly above the provided point (expected to be column top)
        // by one radius plus a small padding to avoid initial interpenetration.
        float sphereRadius = GridObject->MeshComponent ? GridObject->MeshComponent->Bounds.BoxExtent.Z : 0.f;
        Location = InWorldLocation;
        Location.Z += sphereRadius + SpawnPaddingZ;

        // Set the GridObject's world location
        GridObject->SetActorLocation(Location);

        // Set the folder path of the GridObject
        FName GridObjectManagerFolderPath = GetActorFolderPath();
        FName GridObjectFolderPath(*(GridObjectManagerFolderPath.ToString() + "/GridObjects"));
        GridObject->SetFolderPath(GridObjectFolderPath);
    }
}

// Helper function to create a new GridObject
AGridObject* AGridObjectManager::CreateNewGridObjectAtIndex(int32 Index, FVector InObjectSize, float InObjectMass) {
    if (UWorld* World = GetWorld()) {
        AGridObject* NewGridObject = World->SpawnActor<AGridObject>(AGridObject::StaticClass(), FVector::ZeroVector, FRotator::ZeroRotator);
        if (NewGridObject) {
            NewGridObject->InitializeGridObject(InObjectSize, InObjectMass);
            // Tag with environment id if available
            auto GetEnvIdFromActorTags = [](const AActor* Actor) -> int32
            {
                if (!Actor) return -1;
                for (const FName& Tag : Actor->Tags)
                {
                    const FString S = Tag.ToString();
                    if (S.StartsWith(TEXT("EnvId=")))
                    {
                        int32 V = -1; LexFromString(V, *S.RightChop(6)); return V;
                    }
                }
                return -1;
            };
            const int32 EnvId = GetEnvIdFromActorTags(PlatformActor);
            if (EnvId >= 0 && NewGridObject->MeshComponent)
            {
                const FName Tag(*FString::Printf(TEXT("EnvId=%d"), EnvId));
                if (!NewGridObject->MeshComponent->ComponentTags.Contains(Tag))
                {
                    NewGridObject->MeshComponent->ComponentTags.Add(Tag);
                }
            }

            // Set the folder path of the GridObject
            FName GridObjectManagerFolderPath = GetActorFolderPath();
            FName GridObjectFolderPath(*(GridObjectManagerFolderPath.ToString() + "/GridObjects"));
            NewGridObject->SetFolderPath(GridObjectFolderPath);
            NewGridObject->MeshComponent->UpdateBounds();

            // Add the GridObject to the array
            if (GridObjects.Num() <= Index) {
                GridObjects.SetNum(Index + 1);
            }
            GridObjects[Index] = NewGridObject;

            return NewGridObject;
        }
    }
    return nullptr;
}

// Resets all GridObjects by deactivating them
void AGridObjectManager::ResetGridObjects() {
    for (AGridObject* GridObject : GridObjects) {
        if (GridObject) {
            GridObject->SetGridObjectActive(false);
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
        FVector ObjectLocation = GridObject->GetObjectLocation();

        // Get the GridObject's bounds in world space
        FBoxSphereBounds ObjectBoundsWorld = GridObject->MeshComponent->CalcBounds(
            GridObject->MeshComponent->GetComponentTransform()
        );

        // Calculate the effective radius in world units
        float EffectiveRadius = ObjectBoundsWorld.SphereRadius * 2;

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

// Deactivates a GridObject at a specific index
void AGridObjectManager::DisableGridObject(int32 Index) {
    if (GridObjects.IsValidIndex(Index)) {
        AGridObject* GridObject = GridObjects[Index];
        if (GridObject) {
            GridObject->SetGridObjectActive(false);
        }
    }
}

// Get a GridObject by index
AGridObject* AGridObjectManager::GetGridObject(int32 Index) const {
    if (GridObjects.IsValidIndex(Index)) {
        return GridObjects[Index];
    }
    return nullptr;
}
