#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "Matrix2D.h"          // your custom 2D matrix, or you can store TArrays
#include "Containers/Map.h"
#include "Containers/Set.h"
#include "Engine/DataTable.h"  // optional, if you want structured data?

#include "OccupancyGrid.generated.h"

USTRUCT()
struct FOccupancyNode
{
    GENERATED_BODY()

    // For simplicity, store the set of "cells" this node occupies (2D grid indices)
    UPROPERTY()
    TSet<int32> OccupiedCells;

    // Optionally store other info about the object:
    UPROPERTY()
    int32 ObjectId;

    // Possibly store radius, bounding box, or other metadata
    float Radius;

    FOccupancyNode()
        : ObjectId(-1)
        , Radius(1.f)
    {
    }
};

/**
 * A single layer of occupancy data:
 *   - TMap<ObjectId, FOccupancyNode> so we can track which cells each object occupies
 */
USTRUCT()
struct FOccupancyLayer
{
    GENERATED_BODY()

    // For each object => store an occupancy node
    UPROPERTY()
    TMap<int32, FOccupancyNode> Objects;
};

/**
 * UOccupancyGrid:
 *  - Manages multiple "layers" of occupancy (like "Goals", "GridObjects", etc.)
 *  - Tracks which cells each object occupies
 *  - Allows overlap or disallows overlap between layers
 */
UCLASS(BlueprintType)
class UNREALRLLABS_API UOccupancyGrid : public UObject
{
    GENERATED_BODY()

public:
    // -------------------------------------------
    // Initialization
    // -------------------------------------------
    UFUNCTION(BlueprintCallable)
    void InitGrid(int32 InGridSize, float InPlatformSize, FVector InPlatformCenter);

    UFUNCTION(BlueprintCallable)
    void ResetGrid();

    // -------------------------------------------
    //  Adding & Removing
    // -------------------------------------------
    /**
     * Add an object with a given "radius" or bounding shape.
     * We'll find a free location (or we accept a location param if you want),
     * store its OccupiedCells, etc.
     *
     * @param ObjectId => A unique ID for the object
     * @param LayerName => e.g. "Goals", "GridObjects"
     * @param OverlapLayers => which layers we can overlap. If empty => no overlap allowed.
     * @return 2D "cell index" or -1 if fails
     */
    UFUNCTION(BlueprintCallable)
    int32 AddObjectToGrid(
        int32 ObjectId,
        FName LayerName,
        float Radius,
        const TArray<FName>& OverlapLayers
        // Possibly pass "desired location" if you want
    );

    /**
     * Removes an object from a specific layer => frees up its OccupiedCells
     */
    UFUNCTION(BlueprintCallable)
    void RemoveObject(int32 ObjectId, FName LayerName);

    /**
     * Re-positions or updates an existing object =>
     * changes its OccupiedCells accordingly
     */
    UFUNCTION(BlueprintCallable)
    void UpdateObject(int32 ObjectId, FName LayerName, float NewRadius, const TArray<FName>& OverlapLayers);

    // -------------------------------------------
    //  Queries
    // -------------------------------------------
    /**
     * Renders the occupancy from the given set of layers into an NxN matrix:
     *  - 0 => free
     *  - 1 => occupied
     * Or possibly store object Id in the cell
     */
    UFUNCTION(BlueprintCallable)
    FMatrix2D GetOccupancyMatrix(const TArray<FName>& Layers, bool bUseBinary) const;

    /**
     * Helper: convert grid coords => world location (top of column or same plane),
     * or possibly uses your own "CellSize" logic.
     */
    UFUNCTION(BlueprintCallable)
    FVector GridToWorld(int32 GridIndex) const;

    /**
     * Helper: convert world location => nearest grid cell index
     */
    UFUNCTION(BlueprintCallable)
    int32 WorldToGrid(const FVector& WorldLocation) const;

    /**
     * Check if a circle at CellIndex with radius "RadiusCells" would overlap the existing layers
     * that do NOT appear in OverlapLayers
     */
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
    /**
     * Internal function that tries to find a free cell index
     * that doesn't conflict with any "no-overlap" layers
     * or existing objects in the same layer if you want unique space.
     */
    int32 FindFreeCellForRadius(float RadiusCells, const TArray<FName>& OverlapLayers);

    // Possibly a function that calculates which cells a circle of radius "RadiusCells"
    // will occupy if placed at "CellIndex"
    TSet<int32> ComputeOccupiedCells(int32 CellIndex, float RadiusCells) const;

    // -------------------------------------------
    //  Data
    // -------------------------------------------

    // For each layer => the TMap of objectId => node info
    UPROPERTY()
    TMap<FName, FOccupancyLayer> Layers;

    // dimension
    UPROPERTY()
    int32 GridSize;

    UPROPERTY()
    float PlatformSize;
    UPROPERTY()
    FVector PlatformCenter;

    UPROPERTY()
    float CellSize;
};
