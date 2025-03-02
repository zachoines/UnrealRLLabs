#pragma once

#include "CoreMinimal.h"
#include <random>
#include "UObject/NoExportTypes.h"
#include "EnvironmentConfig.h"
#include "TerraShift/Matrix2D.h"
#include "MultiAgentFractalWave3D.generated.h"

USTRUCT(BlueprintType)
struct FFractalAgentAction
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FVector dPos;          // in [-1..1]
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float dPitch;          // in [-1..1]
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float dYaw;            // in [-1..1]
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float dBaseFreq;       // in [-1..1]
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float dLacunarity;     // in [-1..1]
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float dGain;           // in [-1..1]
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float dBlendWeight;    // in [-1..1]

    FFractalAgentAction()
    {
        dPos = FVector::ZeroVector;
        dPitch = 0.f;
        dYaw = 0.f;
        dBaseFreq = 0.f;
        dLacunarity = 0.f;
        dGain = 0.f;
        dBlendWeight = 0.f;
    }
};

USTRUCT()
struct FFractalAgentState
{
    GENERATED_BODY()

    UPROPERTY()
    FVector Pos3D;

    UPROPERTY()
    float Pitch;
    UPROPERTY()
    float Yaw;

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
        Pos3D = FVector::ZeroVector;
        Pitch = 0.f;
        Yaw = 0.f;
        FOVDegrees = 60.f;

        BaseFreq = 0.15f;
        Octaves = 4;
        Lacunarity = 2.5f;
        Gain = 0.6f;
        BlendWeight = 1.f;

        ImageSize = 50;
        SampleDist = 10.f;
    }
};

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
     * Step environment with the given multi-agent actions.
     *
     * If bWrapPos (etc.) is true => wrap in [StatePosRange.X..StatePosRange.Y],
     * else clamp.
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
     * Return an array of normalized state variables in [-1..1],
     * based on user-defined 'state_ranges' from config.
     */
    UFUNCTION(BlueprintCallable, Category = "FractalWave")
    TArray<float> GetAgentStateVariables(int32 AgentIndex) const;

    // -------------------
    //   WRAP TOGGLES
    // -------------------
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    bool bWrapPos = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    bool bWrapPitch = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    bool bWrapYaw = true; // we had yaw_wrap originally set to true

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    bool bWrapFreq = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    bool bWrapLacunarity = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    bool bWrapGain = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    bool bWrapBlendWeight = false;

private:

    UPROPERTY()
    FMatrix2D FinalWave;

    UPROPERTY()
    TArray<FFractalAgentState> Agents;

    UPROPERTY()
    int32 NumAgents;
    UPROPERTY()
    int32 ImageSize;

    // Removed pitch_limit. We'll rely on pitch_range for bounding or wrapping.

    // -------------------
    //  Initialization Ranges
    // -------------------
    UPROPERTY()
    FVector2D PosRange;
    UPROPERTY()
    FVector2D PitchRange;
    UPROPERTY()
    FVector2D YawRange;
    UPROPERTY()
    float DefaultFOVDeg;
    UPROPERTY()
    float DefaultSampleDist;

    // fractal param ranges for random init
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
    FVector2D ActionPosRange;
    UPROPERTY()
    FVector2D ActionPitchRange;
    UPROPERTY()
    FVector2D ActionYawRange;
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
    FVector2D StatePosRange;
    UPROPERTY()
    FVector2D StatePitchRange;
    UPROPERTY()
    FVector2D StateYawRange;
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

    // interpret [-1..1] => [MinVal..MaxVal]
    float ActionScaled(float InputN11, float MinVal, float MaxVal) const;

    // interpret [Value.. in MinMax] => [-1..1]
    float NormalizeValue(float Value, float MinVal, float MaxVal) const;

    // modular wrap a value into [MinVal..MaxVal)
    float WrapValue(float val, float MinVal, float MaxVal) const;
};
