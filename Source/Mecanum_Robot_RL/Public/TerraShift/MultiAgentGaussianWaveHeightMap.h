#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "EnvironmentConfig.h"
#include "TerraShift/Matrix2D.h" // Assuming this path is correct
#include "MultiAgentGaussianWaveHeightMap.generated.h"

/**
 * Each agent's Gaussian wave parameters
 */
USTRUCT(BlueprintType)
struct FGaussianWaveAgent
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FVector2D Position;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Orientation;    // Radians

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float Amplitude;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float SigmaX;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float SigmaY;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	FVector2D Velocity;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float AngularVelocity;

	FGaussianWaveAgent()
		: Position(FVector2D::ZeroVector)
		, Orientation(0.f)
		, Amplitude(0.f)
		, SigmaX(1.f)
		, SigmaY(1.f)
		, Velocity(FVector2D::ZeroVector)
		, AngularVelocity(0.f)
	{}
};

/**
 * Multi-agent wave class that sums multiple 2D Gaussian wave packets into NxN.
 * Agents can interpret actions as either "absolute" or "delta" based on config.
 * The final wave is clamped in [min_height, max_height].
 */
UCLASS(BlueprintType)
class UNREALRLLABS_API UMultiAgentGaussianWaveHeightMap : public UObject
{
	GENERATED_BODY()

public:
	/**
	 * Initialize from environment/params/MultiAgentGaussianWaveHeightMap config block.
	 */
	UFUNCTION(BlueprintCallable)
	void InitializeFromConfig(UEnvironmentConfig* EnvConfig);

	/** Re-init with new number of agents but same config. */
	UFUNCTION(BlueprintCallable)
	void Reset(int32 NewNumAgents);

	/**
	 * Step with a single TArray of floats in [-1..1].
	 * Size must match (NumAgents * ValuesPerAgent).
	 * If bUseActionDelta => apply deltas scaled by delta_xxx parameters. Possibly scale by DeltaTime.
	 */
	UFUNCTION(BlueprintCallable)
	void Step(const TArray<float>& Actions, float DeltaTime = 0.1f);

	/** Returns the final NxN wave matrix. */
	UFUNCTION(BlueprintCallable)
	const FMatrix2D& GetHeightMap() const; // Return const reference

	/** Number of wave agents. */
	UFUNCTION(BlueprintCallable)
	int32 GetNumAgents() const { return Agents.Num(); }

	/** Returns agent's wave parameters as a 9-float array in [-1..1]. */
	UFUNCTION(BlueprintCallable)
	TArray<float> GetAgentState(int32 AgentIndex) const;

protected:
	/** Build NxN wave from each agent, sum them into FinalWave, then clip. */
	void ComputeFinalWave();

	/** Creates an NxN matrix for the given agent's Gaussian wave via matrix ops. */
	[[nodiscard]] FMatrix2D ComputeAgentWave(const FGaussianWaveAgent& Agent) const; // Mark nodiscard

	/** Initialize default agent positions, orientations, etc. */
	void InitializeAgents();

	// Utility for normalizing in agent state
	float MapRange(float x, float inMin, float inMax, float outMin, float outMax) const;
	float MapToN11(float x, float mn, float mx) const;

private:

	// from config
	UPROPERTY()
	int32 NumAgents;

	UPROPERTY()
	int32 GridSize;

	UPROPERTY()
	float MinHeight;
	UPROPERTY()
	float MaxHeight;

	UPROPERTY()
	float MinVel;
	UPROPERTY()
	float MaxVel;

	UPROPERTY()
	float MinAmp;
	UPROPERTY()
	float MaxAmp;

	UPROPERTY()
	float MinSigma;
	UPROPERTY()
	float MaxSigma;

	UPROPERTY()
	float MinAngVel;
	UPROPERTY()
	float MaxAngVel;

	/** # floats per agent action. e.g. 6 => [ vx, vy, angVel, amplitude, sigmaX, sigmaY ] */
	UPROPERTY()
	int32 ValuesPerAgent;

	/** If true => interpret actions as deltas. If false => interpret as absolutes. */
	UPROPERTY()
	bool bUseActionDelta;

	UPROPERTY()
	bool bAccumulatedWave;

	UPROPERTY()
	float AccumulatedWaveFadeGamma;

	/** Scales for each delta if bUseActionDelta is true. */
	UPROPERTY()
	float DeltaVelScale;
	UPROPERTY()
	float DeltaAmpScale;
	UPROPERTY()
	float DeltaSigmaScale;
	UPROPERTY()
	float DeltaAngVelScale;

	/** The NxN wave matrix. */
	UPROPERTY()
	FMatrix2D FinalWave;

	/** The array of wave agents. */
	UPROPERTY()
	TArray<FGaussianWaveAgent> Agents;
};
