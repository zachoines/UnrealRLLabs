// NOTICE: This file includes modifications generated with the assistance of generative AI.
// Original code structure and logic by the project author.

#include "TerraShift/MultiAgentGaussianWaveHeightMap.h"
#include "Math/UnrealMathUtility.h"

// Namespace for configuration helper functions
namespace ConfigHelpers
{
	// Safely gets a float from config, logs warning and returns default if path is missing.
	static float GetOrDefaultNumber(UEnvironmentConfig* Cfg, const FString& Path, float DefaultValue)
	{
		if (!Cfg || !Cfg->IsValid() || !Cfg->HasPath(*Path))
		{
			if (Cfg && Cfg->IsValid()) UE_LOG(LogTemp, Warning, TEXT("Config path '%s' not found, using default: %f"), *Path, DefaultValue);
			return DefaultValue;
		}
		return Cfg->GetOrDefaultNumber(Path, DefaultValue);
	}

	// Safely gets an int32 from config, logs warning and returns default if path is missing.
	static int32 GetOrDefaultInt(UEnvironmentConfig* Cfg, const FString& Path, int32 DefaultValue)
	{
		if (!Cfg || !Cfg->IsValid() || !Cfg->HasPath(*Path))
		{
			if (Cfg && Cfg->IsValid()) UE_LOG(LogTemp, Warning, TEXT("Config path '%s' not found, using default: %d"), *Path, DefaultValue);
			return DefaultValue;
		}
		return Cfg->GetOrDefaultInt(Path, DefaultValue);
	}

	// Safely gets a bool from config, logs warning and returns default if path is missing.
	static bool GetOrDefaultBool(UEnvironmentConfig* Cfg, const FString& Path, bool DefaultValue)
	{
		if (!Cfg || !Cfg->IsValid() || !Cfg->HasPath(*Path))
		{
			if (Cfg && Cfg->IsValid()) UE_LOG(LogTemp, Warning, TEXT("Config path '%s' not found, using default: %s"), *Path, DefaultValue ? TEXT("true") : TEXT("false"));
			return DefaultValue;
		}
		return Cfg->GetOrDefaultBool(Path, DefaultValue);
	}

	// Safely gets an FVector2D from config, logs warning and returns default if path is missing or array size is wrong.
	static FVector2D GetVector2DOrDefault(UEnvironmentConfig* Cfg, const FString& Path, const FVector2D& DefaultValue)
	{
		if (!Cfg || !Cfg->IsValid() || !Cfg->HasPath(*Path))
		{
			if (Cfg && Cfg->IsValid()) UE_LOG(LogTemp, Warning, TEXT("Config path '%s' not found, using default: (%f, %f)"), *Path, DefaultValue.X, DefaultValue.Y);
			return DefaultValue;
		}
		// Use the config class's built-in default getter which handles array parsing.
		return Cfg->GetVector2DOrDefault(Path, DefaultValue);
	}

	// Safely gets a TArray<float> from config, logs warning and returns default if path is missing.
	static TArray<float> GetArrayOrDefault(UEnvironmentConfig* Cfg, const FString& Path, const TArray<float>& DefaultValue)
	{
		if (!Cfg || !Cfg->IsValid() || !Cfg->HasPath(*Path))
		{
			if (Cfg && Cfg->IsValid()) UE_LOG(LogTemp, Warning, TEXT("Config path '%s' not found, using default array."), *Path);
			return DefaultValue;
		}
		// Use the config class's built-in default getter which handles array parsing.
		return Cfg->GetArrayOrDefault(Path, DefaultValue);
	}
} // namespace ConfigHelpers


void UMultiAgentGaussianWaveHeightMap::InitializeFromConfig(UEnvironmentConfig* EnvConfig)
{
	if (!EnvConfig || !EnvConfig->IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("UMultiAgentGaussianWaveHeightMap::InitializeFromConfig - Null or invalid config provided!"));
		return;
	}

	// Read core simulation parameters from config
	NumAgents = ConfigHelpers::GetOrDefaultInt(EnvConfig, TEXT("NumAgents"), 5);
	GridSize = ConfigHelpers::GetOrDefaultInt(EnvConfig, TEXT("GridSize"), 50);
	MinHeight = ConfigHelpers::GetOrDefaultNumber(EnvConfig, TEXT("MinHeight"), -2.f);
	MaxHeight = ConfigHelpers::GetOrDefaultNumber(EnvConfig, TEXT("MaxHeight"), 2.f);

	// Read parameter ranges
	FVector2D VelRange = ConfigHelpers::GetVector2DOrDefault(EnvConfig, TEXT("VelMinMax"), FVector2D(-1.f, 1.f));
	MinVel = VelRange.X;
	MaxVel = VelRange.Y;
	FVector2D AmpRange = ConfigHelpers::GetVector2DOrDefault(EnvConfig, TEXT("AmpMinMax"), FVector2D(0.f, 5.f));
	MinAmp = AmpRange.X;
	MaxAmp = AmpRange.Y;
	FVector2D SigRange = ConfigHelpers::GetVector2DOrDefault(EnvConfig, TEXT("SigMinMax"), FVector2D(0.2f, 5.f));
	MinSigma = SigRange.X;
	MaxSigma = SigRange.Y;
	FVector2D AngVelRange = ConfigHelpers::GetVector2DOrDefault(EnvConfig, TEXT("AngVelRange"), FVector2D(-0.5f, 0.5f));
	MinAngVel = AngVelRange.X;
	MaxAngVel = AngVelRange.Y;

	// Read action interpretation parameters
	ValuesPerAgent = ConfigHelpers::GetOrDefaultInt(EnvConfig, TEXT("NumActions"), 6);
	bUseActionDelta = ConfigHelpers::GetOrDefaultBool(EnvConfig, TEXT("bUseActionDelta"), true);

	// Read wave accumulation parameters
	bAccumulatedWave = ConfigHelpers::GetOrDefaultBool(EnvConfig, TEXT("bAccumulatedWave"), false);
	AccumulatedWaveFadeGamma = ConfigHelpers::GetOrDefaultNumber(EnvConfig, TEXT("AccumulatedWaveFadeGamma"), 0.99f);

	// Read delta scaling factors (used only if bUseActionDelta is true)
	DeltaVelScale = ConfigHelpers::GetOrDefaultNumber(EnvConfig, TEXT("DeltaVelScale"), 0.5f);
	DeltaAmpScale = ConfigHelpers::GetOrDefaultNumber(EnvConfig, TEXT("DeltaAmpScale"), 0.2f);
	DeltaSigmaScale = ConfigHelpers::GetOrDefaultNumber(EnvConfig, TEXT("DeltaSigmaScale"), 0.05f);
	DeltaAngVelScale = ConfigHelpers::GetOrDefaultNumber(EnvConfig, TEXT("DeltaAngVelScale"), 0.3f);

	// Initialize the final wave matrix (constructor performs invariant checks).
	FinalWave = FMatrix2D(GridSize, GridSize, 0.f);

	// Initialize agent states.
	InitializeAgents();
}

void UMultiAgentGaussianWaveHeightMap::Reset(int32 NewNumAgents)
{
	NumAgents = NewNumAgents;

	// Reset the wave matrix (constructor performs invariant checks).
	FinalWave = FMatrix2D(GridSize, GridSize, 0.f);

	InitializeAgents();
}

void UMultiAgentGaussianWaveHeightMap::InitializeAgents()
{
	Agents.Reset(NumAgents); // Optimize allocation
	Agents.SetNum(NumAgents);

	// Initialize each agent with random parameters within configured ranges.
	for (int32 i = 0; i < NumAgents; i++)
	{
		FGaussianWaveAgent& A = Agents[i];
		A.Position.X = FMath::RandRange(0.f, static_cast<float>(GridSize - 1));
		A.Position.Y = FMath::RandRange(0.f, static_cast<float>(GridSize - 1));
		A.Orientation = FMath::RandRange(0.f, 2.f * PI);
		A.Amplitude = FMath::FRandRange(MinAmp, MaxAmp);
		A.SigmaX = FMath::FRandRange(MinSigma, MaxSigma);
		A.SigmaY = FMath::FRandRange(MinSigma, MaxSigma);
		A.Velocity.X = FMath::FRandRange(MinVel, MaxVel);
		A.Velocity.Y = FMath::FRandRange(MinVel, MaxVel);
		A.AngularVelocity = FMath::FRandRange(MinAngVel, MaxAngVel);
	}

	// Compute the initial combined wave based on initialized agents.
	ComputeFinalWave();
}

void UMultiAgentGaussianWaveHeightMap::Step(const TArray<float>& Actions, float DeltaTime)
{
	const int32 ExpectedActionCount = NumAgents * ValuesPerAgent;
	if (Actions.Num() != ExpectedActionCount)
	{
		UE_LOG(LogTemp, Error,
			TEXT("UMultiAgentGaussianWaveHeightMap::Step => Action array size mismatch (received=%d, needed=%d)"),
			Actions.Num(), ExpectedActionCount);
		return; // Avoid processing invalid actions
	}

	// 1. Apply actions to update agent parameters based on config (delta or absolute).
	for (int32 i = 0; i < NumAgents; i++)
	{
		const int32 BaseActionIndex = i * ValuesPerAgent;
		FGaussianWaveAgent& AgentState = Agents[i];

		// Extract actions for clarity (assuming order: vx, vy, angVel, amp, sx, sy)
		float ActionVx = Actions[BaseActionIndex + 0];
		float ActionVy = Actions[BaseActionIndex + 1];
		float ActionAngVel = Actions[BaseActionIndex + 2];
		float ActionAmplitude = Actions[BaseActionIndex + 3];
		float ActionSigmaX = Actions[BaseActionIndex + 4];
		float ActionSigmaY = Actions[BaseActionIndex + 5];

		if (bUseActionDelta)
		{
			// Interpret actions as scaled deltas applied over DeltaTime.
			AgentState.Velocity.X += ActionVx * DeltaVelScale * DeltaTime;
			AgentState.Velocity.Y += ActionVy * DeltaVelScale * DeltaTime;
			AgentState.AngularVelocity += ActionAngVel * DeltaAngVelScale * DeltaTime;
			AgentState.Amplitude += ActionAmplitude * DeltaAmpScale * DeltaTime;
			AgentState.SigmaX += ActionSigmaX * DeltaSigmaScale * DeltaTime;
			AgentState.SigmaY += ActionSigmaY * DeltaSigmaScale * DeltaTime;
		}
		else
		{
			// Interpret actions as absolute target values mapped from [-1, 1] to parameter ranges.
			auto MapN11ToRange = [&](float ValN11, float Min, float Max) {
				float t = FMath::Clamp(0.5f * (ValN11 + 1.f), 0.f, 1.f); // Map [-1,1] to [0,1]
				return FMath::Lerp(Min, Max, t);
				};

			AgentState.Velocity.X = MapN11ToRange(ActionVx, MinVel, MaxVel);
			AgentState.Velocity.Y = MapN11ToRange(ActionVy, MinVel, MaxVel);
			AgentState.AngularVelocity = MapN11ToRange(ActionAngVel, MinAngVel, MaxAngVel);
			AgentState.Amplitude = MapN11ToRange(ActionAmplitude, MinAmp, MaxAmp);
			AgentState.SigmaX = MapN11ToRange(ActionSigmaX, MinSigma, MaxSigma);
			AgentState.SigmaY = MapN11ToRange(ActionSigmaY, MinSigma, MaxSigma);
		}
	}

	// 2. Integrate physics (update position/orientation) and clamp parameters.
	for (FGaussianWaveAgent& AgentState : Agents)
	{
		AgentState.Position += AgentState.Velocity * DeltaTime;
		AgentState.Orientation += AgentState.AngularVelocity * DeltaTime;
		// Consider wrapping AgentState.Orientation if necessary: FMath::Fmod(AgentState.Orientation, 2.f * PI);

		// Clamp position to grid bounds [0, GridSize-1].
		AgentState.Position.X = FMath::Clamp(AgentState.Position.X, 0.f, static_cast<float>(GridSize - 1));
		AgentState.Position.Y = FMath::Clamp(AgentState.Position.Y, 0.f, static_cast<float>(GridSize - 1));

		// Clamp parameters to their defined min/max ranges.
		AgentState.Amplitude = FMath::Clamp(AgentState.Amplitude, MinAmp, MaxAmp);
		AgentState.SigmaX = FMath::Clamp(AgentState.SigmaX, MinSigma, MaxSigma);
		AgentState.SigmaY = FMath::Clamp(AgentState.SigmaY, MinSigma, MaxSigma);
		AgentState.Velocity.X = FMath::Clamp(AgentState.Velocity.X, MinVel, MaxVel);
		AgentState.Velocity.Y = FMath::Clamp(AgentState.Velocity.Y, MinVel, MaxVel);
		AgentState.AngularVelocity = FMath::Clamp(AgentState.AngularVelocity, MinAngVel, MaxAngVel);
	}

	// 3. Recompute the final combined wave based on the updated agent states.
	ComputeFinalWave();
}

void UMultiAgentGaussianWaveHeightMap::ComputeFinalWave()
{
	// Initialize the wave map: either start from zero or apply decay to the previous wave.
	if (bAccumulatedWave) {
		FinalWave *= AccumulatedWaveFadeGamma; // Apply decay factor
	}
	else {
		FinalWave.Init(0.f); // Reset to zero (Init includes invariant check)
	}

	// Add each agent's individual wave contribution to the final wave.
	for (const FGaussianWaveAgent& Agent : Agents)
	{
		FMatrix2D AgentWave = ComputeAgentWave(Agent); // Compute individual wave
		// The += operator in FMatrix2D now performs internal invariant checks.
		FinalWave += AgentWave;
	}

	// Clip the final combined wave to the configured height limits.
	FinalWave.Clip(MinHeight, MaxHeight);
}

FMatrix2D UMultiAgentGaussianWaveHeightMap::ComputeAgentWave(const FGaussianWaveAgent& Agent) const
{
	// Create matrices representing grid row and column indices. Constructors check invariants.
	FMatrix2D RowIndices(GridSize, GridSize);
	FMatrix2D ColIndices(GridSize, GridSize);

	// Fill matrices: RowIndices[r][c] = r, ColIndices[r][c] = c.
	for (int32 r = 0; r < GridSize; ++r)
	{
		for (int32 c = 0; c < GridSize; ++c)
		{
			RowIndices[r][c] = static_cast<float>(r);
			ColIndices[r][c] = static_cast<float>(c);
		}
	}

	// Calculate coordinates relative to the agent's position.
	RowIndices -= Agent.Position.X;
	ColIndices -= Agent.Position.Y;

	// Rotate coordinates by the negative of the agent's orientation.
	float AngleRad = -Agent.Orientation;
	float CosTheta = FMath::Cos(AngleRad);
	float SinTheta = FMath::Sin(AngleRad);

	FMatrix2D RotatedX = (RowIndices * CosTheta) - (ColIndices * SinTheta);
	FMatrix2D RotatedY = (RowIndices * SinTheta) + (ColIndices * CosTheta);

	// Calculate the exponent term of the Gaussian function.
	checkf(!FMath::IsNearlyZero(Agent.SigmaX), TEXT("Agent SigmaX is near zero during wave computation!"));
	checkf(!FMath::IsNearlyZero(Agent.SigmaY), TEXT("Agent SigmaY is near zero during wave computation!"));

	FMatrix2D ExponentTerm = ((RotatedX * RotatedX) / (Agent.SigmaX * Agent.SigmaX)) +
		((RotatedY * RotatedY) / (Agent.SigmaY * Agent.SigmaY));
	ExponentTerm *= -0.5f;

	// Calculate the final wave matrix for this agent.
	FMatrix2D AgentWave = ExponentTerm.Exp(); // Element-wise exponentiation
	AgentWave *= Agent.Amplitude;             // Scale by amplitude

	return AgentWave;
}

//--------------------------------------------------------------------------
// Utility Mappers
//--------------------------------------------------------------------------

// Maps value x from [inMin, inMax] to [outMin, outMax], clamping result.
float UMultiAgentGaussianWaveHeightMap::MapRange(
	float x, float inMin, float inMax, float outMin, float outMax) const
{
	if (FMath::IsNearlyZero(inMax - inMin))
	{
		return outMin; // Avoid division by zero; return lower bound of output range.
	}
	// Clamp input to ensure t is within [0, 1].
	float t = FMath::Clamp((x - inMin) / (inMax - inMin), 0.f, 1.f);
	return FMath::Lerp(outMin, outMax, t);
}

// Maps value x from range [mn, mx] to [-1, 1].
float UMultiAgentGaussianWaveHeightMap::MapToN11(float x, float mn, float mx) const
{
	// Reuse MapRange for consistency and safety checks.
	return MapRange(x, mn, mx, -1.f, 1.f);
}

//--------------------------------------------------------------------------
// State Representation
//--------------------------------------------------------------------------

// Returns the normalized state vector for a given agent.
TArray<float> UMultiAgentGaussianWaveHeightMap::GetAgentState(int32 AgentIndex) const
{
	TArray<float> StateVector;
	if (!Agents.IsValidIndex(AgentIndex))
	{
		UE_LOG(LogTemp, Warning, TEXT("GetAgentState: Invalid AgentIndex %d requested."), AgentIndex);
		// Return empty or default state? Returning empty for now.
		return StateVector;
	}

	const FGaussianWaveAgent& AgentState = Agents[AgentIndex];

	// Normalize parameters to [-1, 1] range using configured min/max values.
	float PosX_N11 = MapToN11(AgentState.Position.X, 0.f, static_cast<float>(GridSize - 1));
	float PosY_N11 = MapToN11(AgentState.Position.Y, 0.f, static_cast<float>(GridSize - 1));

	// Normalize orientation [0, 2*PI) to [-1, 1].
	float WrappedOrientation = FMath::Fmod(AgentState.Orientation, 2.f * PI);
	if (WrappedOrientation < 0.f) { WrappedOrientation += 2.f * PI; }
	float Ori_N11 = MapToN11(WrappedOrientation, 0.f, 2.f * PI);

	float Amp_N11 = MapToN11(AgentState.Amplitude, MinAmp, MaxAmp);
	float SigmaX_N11 = MapToN11(AgentState.SigmaX, MinSigma, MaxSigma);
	float SigmaY_N11 = MapToN11(AgentState.SigmaY, MinSigma, MaxSigma);
	float VelX_N11 = MapToN11(AgentState.Velocity.X, MinVel, MaxVel);
	float VelY_N11 = MapToN11(AgentState.Velocity.Y, MinVel, MaxVel);
	float AngVel_N11 = MapToN11(AgentState.AngularVelocity, MinAngVel, MaxAngVel);

	// Reserve space and add normalized values to the state vector.
	StateVector.Reserve(9);
	StateVector.Add(PosX_N11);
	StateVector.Add(PosY_N11);
	StateVector.Add(Ori_N11);
	StateVector.Add(Amp_N11);
	StateVector.Add(SigmaX_N11);
	StateVector.Add(SigmaY_N11);
	StateVector.Add(VelX_N11);
	StateVector.Add(VelY_N11);
	StateVector.Add(AngVel_N11);

	return StateVector;
}

//--------------------------------------------------------------------------
// Accessors
//--------------------------------------------------------------------------

// Returns a const reference to the final height map.
const FMatrix2D& UMultiAgentGaussianWaveHeightMap::GetHeightMap() const
{
	return FinalWave;
}
