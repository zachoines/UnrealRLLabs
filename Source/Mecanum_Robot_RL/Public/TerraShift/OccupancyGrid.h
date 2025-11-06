#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "Matrix2D.h"
#include "Containers/Map.h"
#include "Containers/Set.h"
#include "Engine/DataTable.h"

#include "OccupancyGrid.generated.h"

/** Metadata describing the cells occupied by a single object. */
USTRUCT()
struct FOccupancyNode
{
    GENERATED_BODY()

    UPROPERTY()
    TSet<int32> OccupiedCells;

    UPROPERTY()
    int32 ObjectId;

    float Radius;

    FOccupancyNode()
        : ObjectId(-1)
        , Radius(1.f)
    {
    }
};

/** Map of object ids to their occupancy metadata for a single layer. */
USTRUCT()
struct FOccupancyLayer
{
    GENERATED_BODY()

    UPROPERTY()
    TMap<int32, FOccupancyNode> Objects;
};

/** UObject wrapper that tracks per-layer cell occupancy for TerraShift actors. */
UCLASS(BlueprintType)
class UNREALRLLABS_API UOccupancyGrid : public UObject
{
    GENERATED_BODY()

public:
    UFUNCTION(BlueprintCallable)
    void InitGrid(int32 InGridSize, float InPlatformSize, FVector InPlatformCenter);

    UFUNCTION(BlueprintCallable)
    void ResetGrid();

    /** Adds an object to a layer and reserves cells based on its radius. */
    UFUNCTION(BlueprintCallable)
    int32 AddObjectToGrid(
        int32 ObjectId,
        FName LayerName,
        float Radius,
        const TArray<FName>& OverlapLayers
    );

    /** Removes an object from a layer and frees its occupied cells. */
    UFUNCTION(BlueprintCallable)
    void RemoveObject(int32 ObjectId, FName LayerName);

    /** Updates a tracked object, recomputing its occupied cells. */
    UFUNCTION(BlueprintCallable)
    void UpdateObject(int32 ObjectId, FName LayerName, float NewRadius, const TArray<FName>& OverlapLayers);

    /** Returns a matrix view for the provided layers (binary or object-id encoded). */
    UFUNCTION(BlueprintCallable)
    FMatrix2D GetOccupancyMatrix(const TArray<FName>& Layers, bool bUseBinary) const;

    /** Converts a grid cell index to world space. */
    UFUNCTION(BlueprintCallable)
    FVector GridToWorld(int32 GridIndex) const;

    /** Converts a world-space location to the nearest grid cell index. */
    UFUNCTION(BlueprintCallable)
    int32 WorldToGrid(const FVector& WorldLocation) const;

    /** Checks whether a placement would overlap disallowed layers. */
    bool WouldOverlap(int32 ProposedCellIndex, float RadiusCells, const TArray<FName>& OverlapLayers) const;

    UFUNCTION(BlueprintCallable)
    TSet<int32> GetCellOccupants(FName LayerName, int32 CellIndex) const;

    bool GetOccupancyNode(FName LayerName, int32 OccupantId, FOccupancyNode& OutNode) const;

    void UpdateObjectPosition(
        int32 ObjectId,
        FName LayerName,
        int32 DesiredCellIndex,
        float RadiusCells,
        const TArray<FName>& OverlapLayers
    );

private:
    /** Finds a free cell index that respects the provided overlap policy. */
    int32 FindFreeCellForRadius(float RadiusCells, const TArray<FName>& OverlapLayers);

    /** Computes the set of cells covered by a disc placed at the given index. */
    TSet<int32> ComputeOccupiedCells(int32 CellIndex, float RadiusCells) const;

    UPROPERTY()
    TMap<FName, FOccupancyLayer> Layers;

    UPROPERTY()
    int32 GridSize;

    UPROPERTY()
    float PlatformSize;
    UPROPERTY()
    FVector PlatformCenter;

    UPROPERTY()
    float CellSize;
};
