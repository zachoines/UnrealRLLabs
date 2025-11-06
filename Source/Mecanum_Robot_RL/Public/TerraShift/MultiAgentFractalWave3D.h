#pragma once

#include "CoreMinimal.h"
#include <random>
#include "UObject/NoExportTypes.h"
#include "EnvironmentConfig.h"
#include "TerraShift/Matrix2D.h"
#include "MultiAgentFractalWave3D.generated.h"

/**
 * Each agent's action now: 9 degrees of freedom, each in [-1..1].
 *
 * We interpret them as deltas for orientation (pitch, yaw, roll),
 * fractal freq, lacunarity, gain, blend weight,
 * plus sample distance and FOV.
 */
USTRUCT(BlueprintType)
struct FFractalAgentAction
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float dPitch;        // [-1..1], used to build a small rotation delta

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float dYaw;          // [-1..1]

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float dRoll;         // [-1..1]

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float dBaseFreq;     // [-1..1]

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float dLacunarity;   // [-1..1]

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float dGain;         // [-1..1]

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float dBlendWeight;  // [-1..1]

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float dSampleDist;   // [-1..1]

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float dFOV;          // [-1..1]

    FFractalAgentAction()
    {
        dPitch = 0.f;
        dYaw = 0.f;
        dRoll = 0.f;
        dBaseFreq = 0.f;
        dLacunarity = 0.f;
        dGain = 0.f;
        dBlendWeight = 0.f;
        dSampleDist = 0.f;
        dFOV = 0.f;
    }
};


/**
 * Each agent's internal state:
 *  - Orientation: stored as a quaternion (no longer storing separate pitch, yaw, roll)
 *  - FOVDegrees & SampleDist => fractal camera parameters
 *  - BaseFreq, Lacunarity, Gain, BlendWeight => fractal params
 *  - Octaves => fractal octaves
 *  - FractalImage => the rendered fractal for each step
 */
USTRUCT()
struct FFractalAgentState
{
    GENERATED_BODY()

    /** Current orientation of the fractal "camera." */
    UPROPERTY()
    FQuat Orientation;

    /** Camera intrinsics. */
    UPROPERTY()
    float FOVDegrees;

    UPROPERTY()
    float SampleDist;

    /** Fractal parameters. */
    UPROPERTY()
    float BaseFreq;
    UPROPERTY()
    float Lacunarity;
    UPROPERTY()
    float Gain;
    UPROPERTY()
    float BlendWeight;

    UPROPERTY()
    int32 Octaves;
    UPROPERTY()
    int32 ImageSize;

    /** The agent's rendered fractal image (size = ImageSize^2). */
    UPROPERTY()
    TArray<float> FractalImage;

    FFractalAgentState()
    {
        Orientation = FQuat::Identity; // Default looking along +Z
        FOVDegrees = 60.f;
        SampleDist = 10.f;
        BaseFreq = 0.15f;
        Lacunarity = 2.f;
        Gain = 0.6f;
        BlendWeight = 1.f;
        Octaves = 3;
        ImageSize = 50;
    }
};


/**
 * UMultiAgentFractalWave3D:
 *  - Manages multiple agents
 *  - Each agent has a quaternion orientation + fractal params
 *  - Renders fractal images from origin
 *  - Weighted sum => final wave in [-1..1]
 */
UCLASS(BlueprintType)
class UNREALRLLABS_API UMultiAgentFractalWave3D : public UObject
{
    GENERATED_BODY()

public:

    UFUNCTION(BlueprintCallable, Category = "FractalWave")
    void InitializeFromConfig(UEnvironmentConfig* Config);

    UFUNCTION(BlueprintCallable, Category = "FractalWave")
    void Reset(int32 NewNumAgents);

    /**
     * Step environment with multi-agent actions:
     *  - scale & apply deltas (pitch, yaw, roll => small orientation delta)
     *  - fractal params (freq, lac, gain, blend, sampleDist, FOV)
     *  - re-render each agent's fractal
     *  - combine wave
     */
    UFUNCTION(BlueprintCallable, Category = "FractalWave")
    void Step(const TArray<FFractalAgentAction>& Actions, float DeltaTime = 0.1f);

    UFUNCTION(BlueprintCallable, Category = "FractalWave")
    const FMatrix2D& GetWave() const { return FinalWave; }

    UFUNCTION(BlueprintCallable, Category = "FractalWave")
    int32 GetNumAgents() const { return Agents.Num(); }

    UFUNCTION(BlueprintCallable, Category = "FractalWave")
    TArray<float> GetAgentFractalImage(int32 AgentIndex) const;

    /** Returns agent state variables in [-1..1], including fractal parameters and optional orientation deltas. */
    UFUNCTION(BlueprintCallable, Category = "FractalWave")
    TArray<float> GetAgentStateVariables(int32 AgentIndex) const;

public:
    /** Whether to initialize agent parameters uniformly at random
     *  or use midpoints for everything. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    bool bUniformRandomInit = false;

private:
    /** The final NxN wave after combining all agents' fractal images. */
    UPROPERTY()
    FMatrix2D FinalWave;

    /** Per-agent state array. */
    UPROPERTY()
    TArray<FFractalAgentState> Agents;

    UPROPERTY()
    int32 NumAgents;
    UPROPERTY()
    int32 ImageSize;

    UPROPERTY()
    int32 Octaves;

    // Fractal parameter wrap toggles
    bool bWrapFreq = false;
    bool bWrapLacunarity = false;
    bool bWrapGain = false;
    bool bWrapBlendWeight = false;
    bool bWrapSampleDist = false;
    bool bWrapFOV = false;

    // Ranges for fractal params & new camera params
    UPROPERTY()
    FVector2D BaseFreqRange;
    UPROPERTY()
    FVector2D LacunarityRange;
    UPROPERTY()
    FVector2D GainRange;
    UPROPERTY()
    FVector2D BlendWeightRange;
    UPROPERTY()
    FVector2D SampleDistRange;
    UPROPERTY()
    FVector2D FOVRange;

    // Action scaling (delta) ranges for the 9D action
    UPROPERTY()
    FVector2D ActionPitchRange;
    UPROPERTY()
    FVector2D ActionYawRange;
    UPROPERTY()
    FVector2D ActionRollRange;
    UPROPERTY()
    FVector2D ActionBaseFreqRange;
    UPROPERTY()
    FVector2D ActionLacunarityRange;
    UPROPERTY()
    FVector2D ActionGainRange;
    UPROPERTY()
    FVector2D ActionBlendWeightRange;
    UPROPERTY()
    FVector2D ActionSampleDistRange;
    UPROPERTY()
    FVector2D ActionFOVRange;

private:
    void InitializeAgents();
    void RenderFractalForAgent(FFractalAgentState& Agent);

    /** Build a small rotation quaternion from dPitch, dYaw, dRoll. */
    FQuat BuildDeltaOrientation(float dPitch, float dYaw, float dRoll) const;

    /** Multi-octave Perlin. */
    float FractalSample3D(float X, float Y, float Z,
        float BaseFreq, int32 Octs,
        float Lacun, float Gn) const;

    // Random generation
    static std::mt19937& GetGenerator();
    float UniformInRange(const FVector2D& Range);

    // Helpers
    float ActionScaled(float InputN11, const FVector2D& MinMax) const;
    float WrapValue(float val, float MinVal, float MaxVal) const;
    float ClampInRange(float val, const FVector2D& range) const;
    float NormalizeValue(float val, const FVector2D& range) const;
};
