#pragma once

#include "CoreMinimal.h"
#include "TerraShift/Matrix2D.h"
#include "Math/UnrealMathUtility.h"
#include "MorletWavelets2D.generated.h"

/**
 * Enum for indexing agent parameters, if needed for debugging or dynamic access.
 */
UENUM()
enum class EAgentParameterIndex : uint8
{
    VelocityX,
    VelocityY,
    Amplitude,
    WaveOrientation,
    Wavenumber,
    Frequency,     // a.k.a. PhaseVelocity
    Phase,
    Sigma,
    Time,
    Count
};

/**
 * Struct to hold per-step wave parameter deltas in [-1..1].
 */
USTRUCT(BlueprintType)
struct FAgentDeltaParameters
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FVector2f Velocity;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Amplitude;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float WaveOrientation;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Wavenumber;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float PhaseVelocity;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Phase;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Sigma;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Time; // Typically absolute time or external accum
};

/**
 * Absolute wave parameter state for each agent within the simulator.
 */
USTRUCT(BlueprintType)
struct FAgentWaveState
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FVector2f Velocity;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Amplitude;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float WaveOrientation;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Wavenumber;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float PhaseVelocity;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Phase;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Sigma;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Time;
};

/**
 * Holds min/max for each wave parameter, used for clamping agent states.
 */
USTRUCT(BlueprintType)
struct FWaveParameterRanges
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FVector2D AmplitudeRange = FVector2D(0.0f, 10.0f);

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FVector2D WaveOrientationRange = FVector2D(0.0f, 2.0f * PI);

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FVector2D WavenumberRange = FVector2D(0.0f, 1.5f);

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FVector2D PhaseVelocityRange = FVector2D(0.0f, 5.0f);

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FVector2D PhaseRange = FVector2D(0.0f, 2.0f * PI);

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FVector2D SigmaRange = FVector2D(0.01f, 15.0f);

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FVector2D VelocityRange = FVector2D(-2.0f, 2.0f);
};

/**
 * Holds user-specified param ranges plus a delta scale factor
 * that controls how strongly a delta of ±1 modifies each parameter.
 */
USTRUCT(BlueprintType)
struct FAgentWaveBounds
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FWaveParameterRanges Ranges;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float DeltaScale = 0.1f;
};

/**
 * MorletWavelets2D:
 * Generates wave heights using per-agent Morlet wave parameters.
 * Each agent's parameters are kept in absolute form, but the environment
 * can pass "deltas" in [-1..1] each step to modify those absolute parameters.
 */
UCLASS(Blueprintable)
class UNREALRLLABS_API UMorletWavelets2D : public UObject
{
    GENERATED_BODY()

public:
    UMorletWavelets2D();

    /**
     * Initializes the wave simulator grid and sets param bounds/delta scale.
     */
    UFUNCTION(BlueprintCallable)
    void Initialize(
        int32 InGridSizeX,
        int32 InGridSizeY,
        const FWaveParameterRanges& InRanges,
        float InDeltaScale = 0.1f
    );

    /**
     * Resets agent states:
     *  - Positions to center
     *  - Wave params to midpoints of their ranges
     *  - Clears the height map
     */
    UFUNCTION(BlueprintCallable)
    void Reset(int32 NumAgents);

    /**
     * Interprets the given deltas in [-1..1], updates each agent's absolute wave
     * parameters, and generates the new height map.
     */
    UFUNCTION(BlueprintCallable)
    const FMatrix2D& Update(const TArray<FAgentDeltaParameters>& DeltaParameters);

    /** Returns the latest height map. */
    UFUNCTION(BlueprintCallable)
    const FMatrix2D& GetHeights() const;

    /** Returns the delta heights from the last update. */
    UFUNCTION(BlueprintCallable)
    const FMatrix2D& GetDeltaHeights() const;

    /** Gets the position of an agent in continuous grid coords. */
    UFUNCTION(BlueprintCallable)
    FVector2f GetAgentPosition(int32 AgentIndex) const;

    /** Basic getters if needed. */
    UFUNCTION(BlueprintCallable)
    int32 GetNumAgents() const { return AgentWaveStates.Num(); }

    /** Get the wave state (absolute parameters) for a specific agent. */
    UFUNCTION(BlueprintCallable)
    FAgentWaveState GetAgentWaveState(int32 AgentIndex);

private:
    /** Internal wave parameter states for each agent */
    UPROPERTY()
    TArray<FAgentWaveState> AgentWaveStates;

    /** Current agent positions in continuous coords */
    UPROPERTY()
    TArray<FVector2f> AgentPositions;

    /** The height maps */
    UPROPERTY()
    FMatrix2D Heights;

    UPROPERTY()
    FMatrix2D DeltaHeights;

    /** Grid dimension, e.g. 50x50 */
    UPROPERTY()
    int32 GridSizeX;

    UPROPERTY()
    int32 GridSizeY;

    UPROPERTY()
    FMatrix2D XGrid;

    UPROPERTY()
    FMatrix2D YGrid;

    /** The bounds for wave parameters and the scale for deltas */
    UPROPERTY()
    FAgentWaveBounds WaveBounds;

    /** Helper function to clamp a param to min..max */
    float ClampParam(float Value, float MinVal, float MaxVal) const;

    /**
     * Applies a relative delta in [-1..1] scaled by DeltaScale * range,
     * then clamps to min..max.
     */
    float ApplyDelta(float CurrentValue, float DeltaNormalized, float MinVal, float MaxVal) const;
};
