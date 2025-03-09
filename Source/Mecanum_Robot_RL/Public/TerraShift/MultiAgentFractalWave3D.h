#pragma once

#include "CoreMinimal.h"
#include <random>
#include "UObject/NoExportTypes.h"
#include "EnvironmentConfig.h"
#include "TerraShift/Matrix2D.h"
#include "MultiAgentFractalWave3D.generated.h"

/**
 * Each agent's action: 7 degrees of freedom, each in [-1..1].
 *
 * We interpret them as deltas for pitch, yaw, roll, fractal freq,
 * lacunarity, gain, and blend weight.
 */
USTRUCT(BlueprintType)
struct FFractalAgentAction
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float dPitch;        // [-1..1]

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

    FFractalAgentAction()
    {
        dPitch = 0.f;
        dYaw = 0.f;
        dRoll = 0.f;
        dBaseFreq = 0.f;
        dLacunarity = 0.f;
        dGain = 0.f;
        dBlendWeight = 0.f;
    }
};

/**
 * Each agent's internal state:
 *  - (Pitch, Yaw, Roll) => orientation angles in radians
 *  - FOVDegrees & SampleDist => constants (not changed by actions)
 *  - (BaseFreq, Lacunarity, Gain, BlendWeight) => fractal params
 *  - Octaves => fractal octaves
 *  - FractalImage => the rendered fractal for each step
 */
USTRUCT()
struct FFractalAgentState
{
    GENERATED_BODY()

    UPROPERTY()
    float Pitch;
    UPROPERTY()
    float Yaw;
    UPROPERTY()
    float Roll;

    UPROPERTY()
    float FOVDegrees;

    UPROPERTY()
    float BaseFreq;
    UPROPERTY()
    float Lacunarity;
    UPROPERTY()
    float Gain;
    UPROPERTY()
    int32 Octaves;
    UPROPERTY()
    float BlendWeight;

    UPROPERTY()
    int32 ImageSize;
    UPROPERTY()
    float SampleDist;

    UPROPERTY()
    TArray<float> FractalImage;

    FFractalAgentState()
    {
        Pitch = 0.f;
        Yaw = 0.f;
        Roll = 0.f;

        FOVDegrees = 60.f;
        BaseFreq = 0.15f;
        Lacunarity = 2.f;
        Gain = 0.6f;
        Octaves = 3;
        BlendWeight = 1.f;

        ImageSize = 50;
        SampleDist = 10.f;
    }
};

/**
 * UMultiAgentFractalWave3D:
 *  - Manages multiple agents
 *  - Each agent has rotation + fractal parameters
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
     *  - scale & apply each delta (pitch, yaw, roll, fractal params)
     *  - re-render each agent's fractal
     *  - combine wave
     */
    UFUNCTION(BlueprintCallable, Category = "FractalWave")
    void Step(const TArray<FFractalAgentAction>& Actions, float DeltaTime = 0.1f);

    UFUNCTION(BlueprintCallable, Category = "FractalWave")
    const FMatrix2D& GetWave() const;

    UFUNCTION(BlueprintCallable, Category = "FractalWave")
    int32 GetNumAgents() const { return Agents.Num(); }

    UFUNCTION(BlueprintCallable, Category = "FractalWave")
    TArray<float> GetAgentFractalImage(int32 AgentIndex) const;

    /**
     * Returns agent's state variables in [-1..1].
     *
     * pitchNorm, yawNorm, rollNorm,
     * freqNorm, lacNorm, gainNorm, blendNorm
     */
    UFUNCTION(BlueprintCallable, Category = "FractalWave")
    TArray<float> GetAgentStateVariables(int32 AgentIndex) const;

    // -------------------
    //   Wrap Toggles
    // -------------------
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    bool bWrapPitch = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    bool bWrapYaw = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    bool bWrapRoll = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    bool bWrapFreq = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    bool bWrapLacunarity = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    bool bWrapGain = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    bool bWrapBlendWeight = true;

private:

    UPROPERTY()
    FMatrix2D FinalWave;     // NxN final image

    UPROPERTY()
    TArray<FFractalAgentState> Agents;

    UPROPERTY()
    int32 NumAgents;
    UPROPERTY()
    int32 ImageSize;

    // -------------------
    //  Initialization Ranges
    // -------------------
    UPROPERTY()
    FVector2D PitchRange;
    UPROPERTY()
    FVector2D YawRange;
    UPROPERTY()
    FVector2D RollRange;

    UPROPERTY()
    float DefaultFOVDeg;
    UPROPERTY()
    float DefaultSampleDist;

    UPROPERTY()
    FVector2D BaseFreqRange;
    UPROPERTY()
    FVector2D LacunarityRange;
    UPROPERTY()
    FVector2D GainRange;
    UPROPERTY()
    FVector2D BlendWeightRange;
    UPROPERTY()
    int32 Octaves;

    // -------------------
    //  Action Ranges => Delta
    // -------------------
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

    // -------------------
    //  State Normalization Ranges
    // -------------------
    UPROPERTY()
    FVector2D StatePitchRange;
    UPROPERTY()
    FVector2D StateYawRange;
    UPROPERTY()
    FVector2D StateRollRange;
    UPROPERTY()
    FVector2D StateBaseFreqRange;
    UPROPERTY()
    FVector2D StateLacunarityRange;
    UPROPERTY()
    FVector2D StateGainRange;
    UPROPERTY()
    FVector2D StateBlendWeightRange;

private:
    void InitializeAgents();
    void RenderFractalForAgent(FFractalAgentState& Agent);
    float FractalSample3D(float X, float Y, float Z,
        float BaseFreq, int32 Octs,
        float Lacun, float Gn) const;

    // random generation
    static std::mt19937& GetNormalGenerator();
    float SampleNormalInRange(const FVector2D& Range);

    // mapping
    float Map(float x, float in_min, float in_max, float out_min, float out_max) const;
    float ActionScaled(float InputN11, float MinVal, float MaxVal) const;
    float NormalizeValue(float Value, float MinVal, float MaxVal) const;

    // modular wrap
    float WrapValue(float val, float MinVal, float MaxVal) const;
};
