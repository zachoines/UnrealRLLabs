// NOTICE: This file includes modifications generated with the assistance of generative AI.
// Original code structure and logic by the project author.

#include "TerraShift/GridObjectManager.h"
#include "TerraShift/GridObject.h"
#include "Engine/World.h"
#include "TimerManager.h"
#include "UObject/NameTypes.h"

AGridObjectManager::AGridObjectManager()
{
    PrimaryActorTick.bCanEverTick = false;

    ObjectSize = FVector(0.1f, 0.1f, 0.1f);
    ObjectMass = 0.1f;
    PlatformActor = nullptr;
}

void AGridObjectManager::SetPlatformActor(AActor* InPlatform)
{
    PlatformActor = InPlatform;
}

FName AGridObjectManager::GetActorFolderPath() const
{
    return GetFolderPath();
}

void AGridObjectManager::SpawnGridObjects(const TArray<FVector>& Locations, FVector InObjectSize, float InObjectMass, float SpawnDelay)
{
    ObjectSize = InObjectSize;
    ObjectMass = InObjectMass;

    for (int i = 0; i < Locations.Num(); i++) {
        float TotalSpawnDelay = SpawnDelay * static_cast<float>(i);

        if (TotalSpawnDelay == 0) {
            SpawnGridObjectAtIndex(i, Locations[i], InObjectSize, InObjectMass);
        }
        else {
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
        FVector Location = InWorldLocation;

        GridObject->ResetGridObject();

        float sphereRadius = GridObject->MeshComponent->Bounds.BoxExtent.Z * 2;

        Location.Z += sphereRadius;

        GridObject->SetActorLocation(Location);

        FName GridObjectManagerFolderPath = GetActorFolderPath();
        FName GridObjectFolderPath(*(GridObjectManagerFolderPath.ToString() + "/GridObjects"));
        GridObject->SetFolderPath(GridObjectFolderPath);
    }
}

AGridObject* AGridObjectManager::CreateNewGridObjectAtIndex(int32 Index, FVector InObjectSize, float InObjectMass)
{
    if (UWorld* World = GetWorld()) {
        AGridObject* NewGridObject = World->SpawnActor<AGridObject>(AGridObject::StaticClass(), FVector::ZeroVector, FRotator::ZeroRotator);
        if (NewGridObject) {
            NewGridObject->InitializeGridObject(InObjectSize, InObjectMass);

            FName GridObjectManagerFolderPath = GetActorFolderPath();
            FName GridObjectFolderPath(*(GridObjectManagerFolderPath.ToString() + "/GridObjects"));
            NewGridObject->SetFolderPath(GridObjectFolderPath);
            NewGridObject->MeshComponent->UpdateBounds();

            if (GridObjects.Num() <= Index) {
                GridObjects.SetNum(Index + 1);
            }
            GridObjects[Index] = NewGridObject;

            return NewGridObject;
        }
    }
    return nullptr;
}

void AGridObjectManager::ResetGridObjects()
{
    for (AGridObject* GridObject : GridObjects) {
        if (GridObject) {
            GridObject->SetGridObjectActive(false);
        }
    }
}

FVector AGridObjectManager::GetGridObjectWorldLocation(int32 Index) const
{
    if (GridObjects.IsValidIndex(Index) && GridObjects[Index]) {
        return GridObjects[Index]->GetActorLocation();
    }
    return FVector::ZeroVector;
}

TSet<int32> AGridObjectManager::GetActiveColumnsInProximity(int32 GridSize, const TArray<FVector>& ColumnCenters, const FVector& PlatformCenter, float PlatformSize, float CellSize) const
{
    TSet<int32> ActiveColumnIndices;

    float HalfPlatformSize = PlatformSize / 2.0f;
    float GridOriginX = PlatformCenter.X - HalfPlatformSize;
    float GridOriginY = PlatformCenter.Y - HalfPlatformSize;

    for (AGridObject* GridObject : GridObjects) {
        if (!GridObject || !GridObject->IsActive()) {
            continue;
        }

        FVector ObjectLocation = GridObject->GetObjectLocation();

        FBoxSphereBounds ObjectBoundsWorld = GridObject->MeshComponent->CalcBounds(
            GridObject->MeshComponent->GetComponentTransform()
        );

        float EffectiveRadius = ObjectBoundsWorld.SphereRadius * 2;

        float MinX = ObjectLocation.X - EffectiveRadius;
        float MaxX = ObjectLocation.X + EffectiveRadius;
        float MinY = ObjectLocation.Y - EffectiveRadius;
        float MaxY = ObjectLocation.Y + EffectiveRadius;

        int32 MinXIndex = FMath::FloorToInt((MinX - GridOriginX) / CellSize);
        int32 MaxXIndex = FMath::FloorToInt((MaxX - GridOriginX) / CellSize);
        int32 MinYIndex = FMath::FloorToInt((MinY - GridOriginY) / CellSize);
        int32 MaxYIndex = FMath::FloorToInt((MaxY - GridOriginY) / CellSize);

        MinXIndex = FMath::Clamp(MinXIndex, 0, GridSize - 1);
        MaxXIndex = FMath::Clamp(MaxXIndex, 0, GridSize - 1);
        MinYIndex = FMath::Clamp(MinYIndex, 0, GridSize - 1);
        MaxYIndex = FMath::Clamp(MaxYIndex, 0, GridSize - 1);

        for (int32 X = MinXIndex; X <= MaxXIndex; ++X) {
            for (int32 Y = MinYIndex; Y <= MaxYIndex; ++Y) {
                int32 ColumnIndex = X * GridSize + Y;
                FVector ColumnCenter = ColumnCenters[ColumnIndex];

                if (FVector::Dist2D(ColumnCenter, ObjectLocation) <= EffectiveRadius) {
                    ActiveColumnIndices.Add(ColumnIndex);
                }
            }
        }
    }

    return ActiveColumnIndices;
}

void AGridObjectManager::DisableGridObject(int32 Index)
{
    if (GridObjects.IsValidIndex(Index)) {
        AGridObject* GridObject = GridObjects[Index];
        if (GridObject) {
            GridObject->SetGridObjectActive(false);
        }
    }
}

AGridObject* AGridObjectManager::GetGridObject(int32 Index) const
{
    if (GridObjects.IsValidIndex(Index)) {
        return GridObjects[Index];
    }
    return nullptr;
}
