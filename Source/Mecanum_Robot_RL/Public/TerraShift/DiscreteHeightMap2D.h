#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "TerraShift/Matrix2D.h"
#include "Math/UnrealMathUtility.h"
#include "DiscreteHeightMap2D.generated.h"

UENUM(BlueprintType)
enum class EAgentDirection : uint8
{
    Up    UMETA(DisplayName = "Up"),
    Down  UMETA(DisplayName = "Down"),
    Left  UMETA(DisplayName = "Left"),
    Right UMETA(DisplayName = "Right"),
    None  UMETA(DisplayName = "None"),
};

UENUM(BlueprintType)
enum class EAgentMatrixUpdate : uint8
{
    Inc  UMETA(DisplayName = "Inc"),
    Dec  UMETA(DisplayName = "Dec"),
    None UMETA(DisplayName = "None"),
};


UENUM(BlueprintType)
enum class EAgentReflection : uint8
{
    Reflect  UMETA(DisplayName = "Reflect"),
    None UMETA(DisplayName = "None"),
};


USTRUCT(BlueprintType)
struct UNREALRLLABS_API FAgentHeightDelta
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    EAgentDirection Direction;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    EAgentMatrixUpdate MatrixUpdate;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    EAgentReflection Reflect;
};

/**
 * A simple discrete heightmap class that manages grid-based heights for each cell,
 * and tracks agent positions in the grid. Agents can move in discrete directions
 * (with wrap-around). They can also apply partial updates to the current cell.
 */
UCLASS(BlueprintType)
class UNREALRLLABS_API UDiscreteHeightMap2D : public UObject
{
    GENERATED_BODY()

public:

    UDiscreteHeightMap2D();

    /** Initialize the matrix dimension, agent count, etc. */
    UFUNCTION(BlueprintCallable, Category = "HeightMap")
    void Initialize(int32 InGridSizeX,
        int32 InGridSizeY,
        int32 InNumAgents,
        FVector2D InMatrixDeltaRange,
        float InMaxAbsMatrixHeight);

    /** Reset with a possibly new number of agents. */
    UFUNCTION(BlueprintCallable, Category = "HeightMap")
    void Reset(int32 NewNumAgents);

    /** Update the height map given an array of deltas, one per agent. */
    UFUNCTION(BlueprintCallable, Category = "HeightMap")
    void Update(const TArray<FAgentHeightDelta>& Deltas);

    /** Read-only access to the entire FMatrix2D. */
    UFUNCTION(BlueprintCallable, Category = "HeightMap")
    const FMatrix2D& GetHeights() const;

    /** Retrieve a small array of agent-specific data (x, y, height, etc.). */
    UFUNCTION(BlueprintCallable, Category = "HeightMap")
    TArray<float> GetAgentState(int32 AgentIndex) const;

private:

    // 2D matrix storing the height at each cell
    UPROPERTY()
    FMatrix2D HeightMap;

    UPROPERTY()
    int32 GridSizeX;

    UPROPERTY()
    int32 GridSizeY;

    UPROPERTY()
    int32 NumAgents;

    UPROPERTY()
    FVector2D MatrixDeltaRange;

    UPROPERTY()
    float MaxAbsMatrixHeight;

    /** Where each agent is in the grid. */
    UPROPERTY()
    TArray<FIntPoint> AgentPositions;

    /** Helper to place agents randomly or at default positions. */
    void PlaceAgents();

    /** Wrap an X coordinate into the valid [0..GridSizeX-1] range. */
    int32 WrapX(int32 X) const;

    /** Wrap a Y coordinate into the valid [0..GridSizeY-1] range. */
    int32 WrapY(int32 Y) const;
};
