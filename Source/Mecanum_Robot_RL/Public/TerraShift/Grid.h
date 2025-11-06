#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Engine/World.h"
#include "Engine/Engine.h"
#include "TerraShift/Column.h"
#include "TerraShift/Matrix2D.h"
#include "Grid.generated.h"

UCLASS()
class UNREALRLLABS_API AGrid : public AActor {
    GENERATED_BODY()

public:
    AGrid();

    /** Initializes the grid with the specified size, platform radius, and world location. */
    void InitializeGrid(int32 InGridSize, float InPlatformSize, FVector Location);

    /** Updates column heights from a height map. */
    void UpdateColumnHeights(const FMatrix2D& HeightMap);

    /** Enables or disables physics on each column index according to EnablePhysics. */
    void TogglePhysicsForColumns(const TArray<int32>& ColumnIndices, const TArray<bool>& EnablePhysics);

    /** Resets the grid to its initial state. */
    void ResetGrid();

    /** Returns the world location of a column by flattened index. */
    FVector GetColumnWorldLocation(int32 ColumnIndex) const;

    /** Returns world-space centers for all columns. */
    TArray<FVector> GetColumnCenters() const;

    /** Returns local-space offsets for a column coordinate. */
    FVector GetColumnOffsets(int32 X, int32 Y) const;

    /** Computes the world position for a column at (X, Y) and height. */
    FVector CalculateColumnLocation(int32 X, int32 Y, float Height) const;

    /** Sets the local-space movement bounds for all columns. */
    void SetColumnMovementBounds(float Min, float Max);

    /** Computes corrective offsets applied to edge columns. */
    FVector2D CalculateEdgeCorrectiveOffsets(int32 X, int32 Y) const;

    /** Returns the total number of columns. */
    int32 GetTotalColumns() const;

    /** Returns the current height of a column by index. */
    float GetColumnHeight(int32 ColumnIndex) const;

    /** Sets the color of a column. */
    void SetColumnColor(int32 ColumnIndex, const FLinearColor& Color);

    /** Returns the minimum allowed column height. */
    float GetMinHeight() const;

    /** Returns the maximum allowed column height. */
    float GetMaxHeight() const;

    UPROPERTY()
    TArray<AColumn*> Columns;

private:
    float MinHeight;

    float MaxHeight;

    int32 GridSize;

    float CellSize;

    float PlatformSize;

    /** Converts a flattened index into an (X, Y) grid coordinate. */
    FIntPoint Get2DIndexFrom1D(int32 Index) const;

    /** Sets the physics state of a column. */
    void SetColumnPhysics(int32 ColumnIndex, bool bEnablePhysics);

    /** Maps a value from one range to another. */
    float Map(float x, float in_min, float in_max, float out_min, float out_max);
};
