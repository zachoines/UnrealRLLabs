#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Engine/World.h"
#include "TerraShift/Column.h"
#include "TerraShift/Matrix2D.h"
#include "Grid.generated.h"

UCLASS()
class UNREALRLLABS_API AGrid : public AActor {
    GENERATED_BODY()

public:
    AGrid();

    // Initialize the grid with specified parameters
    void InitializeGrid(int32 InGridSize, float InPlatformSize, FVector Location);

    // Update the heights of the columns based on a 2D matrix of heights
    void UpdateColumnHeights(const Matrix2D& HeightMap);

    // Enable or disable physics for specified columns based on proximity data
    void TogglePhysicsForColumns(const TArray<int32>& ColumnIndices, const TArray<bool>& EnablePhysics);
    
    // Reset the grid to its initial state
    void ResetGrid();

    // Get the world location of the column at a specific index
    FVector GetColumnWorldLocation(int32 ColumnIndex) const;

    // Get the center positions of all columns
    TArray<FVector> GetColumnCenters() const;

    // Gets column localspace offsets
    FVector GetColumnOffsets(int32 X, int32 Y) const;

    // Helper function to calculate the world position of a column based on grid indices
    FVector CalculateColumnLocation(int32 X, int32 Y, float Height) const;

    // Funciton to set the localspace min/max Z movement of columns 
    void SetColumnMovementBounds(float Min, float Max);

    FVector2D CalculateEdgeCorrectiveOffsets(int32 X, int32 Y) const;


private:

    // Min localspace movement of columns
    float MinHeight;

    float MaxHeight;

    // Number of cells along one side of the grid
    int32 GridSize;

    // Size of each grid cell
    float CellSize;

    // Platform size
    float PlatformSize;

    // Array of column actors
    UPROPERTY()
    TArray<AColumn*> Columns;

    // Helper function to convert a 1D index to a 2D grid point
    FIntPoint Get2DIndexFrom1D(int32 Index) const;

    // Helper function to set the physics state of a column
    void SetColumnPhysics(int32 ColumnIndex, bool bEnablePhysics);

    // Map a value from one range to another
    float Map(float x, float in_min, float in_max, float out_min, float out_max);
};