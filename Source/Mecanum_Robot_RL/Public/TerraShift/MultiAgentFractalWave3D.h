#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "EnvironmentConfig.h"
#include "TerraShift/Matrix2D.h"
#include "MultiAgentFractalWave3D.generated.h"

/**
 * The agent "action" deltas for controlling camera & fractal.
 */
USTRUCT(BlueprintType)
struct FFractalAgentAction
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FVector dPos;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float dPitch;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float dYaw;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float dBaseFreq;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float dLacunarity;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float dGain;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float dBlendWeight;

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

/**
 * Agent camera/fractal state + flattened NxN fractal image.
 */
USTRUCT()
struct FFractalAgentState
{
    GENERATED_BODY()

    // Camera transforms
    UPROPERTY()
    FVector Pos3D;

    UPROPERTY()
    float Pitch;
    UPROPERTY()
    float Yaw;

    UPROPERTY()
    float FOVDegrees;

    // Fractal noise params
    UPROPERTY()
    float BaseFreq;
    UPROPERTY()
    int32 Octaves;
    UPROPERTY()
    float Lacunarity;
    UPROPERTY()
    float Gain;

    UPROPERTY()
    float BlendWeight;

    UPROPERTY()
    int32 ImageSize;

    UPROPERTY()
    float SampleDist;

    // Flattened NxN fractal image in [-1..1]
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

    UFUNCTION(BlueprintCallable, Category = "FractalWave")
    void Step(const TArray<FFractalAgentAction>& Actions, float DeltaTime = 0.1f);

    /** Final NxN wave in [-1..1]. */
    UFUNCTION(BlueprintCallable, Category = "FractalWave")
    const FMatrix2D& GetWave() const;

    UFUNCTION(BlueprintCallable, Category = "FractalWave")
    int32 GetNumAgents() const { return Agents.Num(); }

    /** Return the NxN fractal image for a single agent. */
    UFUNCTION(BlueprintCallable, Category = "FractalWave")
    TArray<float> GetAgentFractalImage(int32 AgentIndex) const;

    /**
     * Return a TArray<float> containing other agent state variables:
     * e.g. [posX, posY, posZ, pitch, yaw, baseFreq, octaves, lacunarity, gain, blendWeight, fovDeg, sampleDist].
     */
    UFUNCTION(BlueprintCallable, Category = "FractalWave")
    TArray<float> GetAgentStateVariables(int32 AgentIndex) const;

private:

    UPROPERTY()
    FMatrix2D FinalWave;

    UPROPERTY()
    TArray<FFractalAgentState> Agents;

    // config-based
    UPROPERTY()
    int32 NumAgents;
    UPROPERTY()
    int32 ImageSize;

    UPROPERTY()
    float PitchLimit;
    UPROPERTY()
    bool bYawWrap;

    // Ranges for random init
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

    // fractal param ranges
    UPROPERTY()
    FVector2D BaseFreqRange;
    UPROPERTY()
    FIntPoint  OctavesRange;
    UPROPERTY()
    FVector2D LacunarityRange;
    UPROPERTY()
    FVector2D GainRange;
    UPROPERTY()
    FVector2D BlendWeightRange;

private:
    void InitializeAgents();
    void RenderFractalForAgent(FFractalAgentState& Agent);
    float FractalSample3D(float X, float Y, float Z, float BaseFreq, int32 Octs, float Lacun, float Gn) const;
};
