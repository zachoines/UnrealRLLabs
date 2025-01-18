#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "TerraShift/Matrix2D.h"
#include "Math/UnrealMathUtility.h"
#include "DiscreteFourier2D.generated.h"

/**
 * Simple enum for agent's discrete direction choice.
 */
UENUM(BlueprintType)
enum class EAgentDirection : uint8
{
    Up,
    Down,
    Left,
    Right,
    None
};

/**
 * Simple enum for partial matrix update choice at current (row,col).
 */
UENUM(BlueprintType)
enum class EAgentMatrixUpdate : uint8
{
    Inc,
    Dec,
    Zero,
    None
};

/**
 * Stores each agent's local 2D Fourier state:
 *   - A (2K x 2K) coefficient matrix
 *   - (row, col) location in the A matrix
 */
USTRUCT(BlueprintType)
struct FAgentFourierState
{
    GENERATED_BODY()

    // (2K x 2K) matrix for agent-specific Fourier coefficients
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FMatrix2D AgentA;

    // Current location in the A matrix
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    int32 Row;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    int32 Col;
};

/**
 * Discrete "delta" for each agent in one step:
 *   - A direction to move (row,col) in A
 *   - A partial update to the cell at (row,col)
 */
USTRUCT(BlueprintType)
struct FAgentFourierDelta
{
    GENERATED_BODY()

    // Where to move (up/down/left/right)
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    EAgentDirection Direction;

    // partial update to the A[row,col]
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    EAgentMatrixUpdate MatrixUpdate;
};

/**
 * A class that manages a grid-based 2D Fourier representation with multiple agents:
 *   - Each agent has a (2K x 2K) matrix + (row, col) location.
 *   - At each Update, we:
 *       1) Move each agent's (row,col) with wrap-around if needed
 *       2) Partially update the cell at (row,col) clamped by MatrixDeltaRange
 *       3) Recompute the final NxN heightmap by summing agent contributions
 *         G_total = Σ ( Sx * AgentA * Sy^T )
 *         (We keep a standard basis Sx, Sy with no shift.)
 */
UCLASS(Blueprintable)
class UNREALRLLABS_API UDiscreteFourier2D : public UObject
{
    GENERATED_BODY()

public:
    UDiscreteFourier2D();

    /**
     * Initialize the 2D Fourier system.
     * @param InGridSizeX       - The width (X dimension) of final NxN grid
     * @param InGridSizeY       - The height (Y dimension) of final NxN grid
     * @param InNumAgents       - Number of discrete Fourier agents
     * @param InK               - Number of fundamental frequency steps => each agent's matrix is (2K x 2K)
     * @param InMatrixDeltaRange - Clamping range for partial matrix updates: [Low..High]
     */
    UFUNCTION(BlueprintCallable)
    void Initialize(int32 InGridSizeX, int32 InGridSizeY, int32 InNumAgents, int32 InK, FVector2D InMatrixDeltaRange);

    /**
     * Reset with a new number of agents.
     * This re-allocates agent states (with random row,col) and clears the heightmaps.
     */
    UFUNCTION(BlueprintCallable)
    void Reset(int32 NewNumAgents);

    /**
     * Apply an array of FAgentFourierDelta (one per agent),
     * then re-sum the NxN heightmap from each agent.
     *  Steps:
     *   - Move agent's (row,col)
     *   - partial update (row,col), clamped by MatrixDeltaRange
     * @param Deltas - The array of discrete updates for each agent
     * @return       - The new NxN heights
     */
    UFUNCTION(BlueprintCallable)
    const FMatrix2D& Update(const TArray<FAgentFourierDelta>& Deltas);

    /**
     * Return the latest NxN heightmap.
     */
    UFUNCTION(BlueprintCallable)
    const FMatrix2D& GetHeights() const;

    /**
     * Return the difference in heights from the last step: (CurrentHeights - PreviousHeights).
     */
    UFUNCTION(BlueprintCallable)
    const FMatrix2D& GetDeltaHeights() const;

    /**
     * Returns the current number of agents in the system.
     */
    UFUNCTION(BlueprintCallable)
    int32 GetNumAgents() const { return Agents.Num(); }

    /**
     * Return a copy of a given agent's (2K x 2K) coefficient matrix.
     */
    UFUNCTION(BlueprintCallable)
    FMatrix2D GetAgentMatrix(int32 AgentIndex) const;

    /**
     * Flatten an agent's internal state into a float array:
     * [ row, col, ...all elements of A matrix in row-major ...]
     */
    UFUNCTION(BlueprintCallable)
    TArray<float> GetAgentFourierState(int32 AgentIndex) const;

private:
    // Holds each agent's Fourier data
    UPROPERTY()
    TArray<FAgentFourierState> Agents;

    // Grid size for final NxN heightmap
    UPROPERTY()
    int32 GridSizeX;

    UPROPERTY()
    int32 GridSizeY;

    // Number of fundamental modes => each agent's A is (2K x 2K).
    UPROPERTY()
    int32 K;

    // The final NxN heightmap
    UPROPERTY()
    FMatrix2D Heights;

    // The difference from the last step
    UPROPERTY()
    FMatrix2D DeltaHeights;

    // Cached copy of the previous NxN (for DeltaHeights)
    FMatrix2D PreviousHeights;

    // A standard 2D basis (Sx, Sy) with shape: (GridSizeY x 2K), (GridSizeX x 2K).
    // Built once in Initialize.
    UPROPERTY()
    FMatrix2D BasisSx;

    UPROPERTY()
    FMatrix2D BasisSy;

    /**
     * Range for partial matrix increments. We clamp the final cell value to [Low..High].
     * e.g. [-0.05..+0.05].
     */
    UPROPERTY()
    FVector2D MatrixDeltaRange;

private:
    /**
     * Compute a single agent's NxN contribution:
     *    G_i = BasisSx * AgentA * BasisSy^T
     */
    FMatrix2D ComputeAgentHeight(int32 AgentIndex) const;

    /**
     * Build the constant basis Sx, Sy for the entire grid.
     * They do not shift per agent, so just once is enough.
     */
    void BuildBasis();
};
