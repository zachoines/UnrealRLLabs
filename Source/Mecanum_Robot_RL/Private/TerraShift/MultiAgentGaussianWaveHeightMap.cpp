// NOTICE: This file includes modifications generated with the assistance of generative AI.
// Original code structure and logic by the project author.

#include "TerraShift/MultiAgentGaussianWaveHeightMap.h"
#include "Math/UnrealMathUtility.h"

// Namespace for configuration helper functions (as it was in your original file)
namespace ConfigHelpers
{
	static float GetOrDefaultNumber(UEnvironmentConfig* Cfg, const FString& Path, float DefaultValue)
	{
		if (!Cfg || !Cfg->IsValid() || !Cfg->HasPath(Path)) // Removed TEXT macro around Path
		{
			if (Cfg && Cfg->IsValid()) UE_LOG(LogTemp, Warning, TEXT("Config path '%s' not found, using default: %f"), *Path, DefaultValue);
			return DefaultValue;
		}
		return Cfg->GetOrDefaultNumber(Path, DefaultValue);
	}

	static int32 GetOrDefaultInt(UEnvironmentConfig* Cfg, const FString& Path, int32 DefaultValue)
	{
		if (!Cfg || !Cfg->IsValid() || !Cfg->HasPath(Path))
		{
			if (Cfg && Cfg->IsValid()) UE_LOG(LogTemp, Warning, TEXT("Config path '%s' not found, using default: %d"), *Path, DefaultValue);
			return DefaultValue;
		}
		return Cfg->GetOrDefaultInt(Path, DefaultValue);
	}

	static bool GetOrDefaultBool(UEnvironmentConfig* Cfg, const FString& Path, bool DefaultValue)
	{
		if (!Cfg || !Cfg->IsValid() || !Cfg->HasPath(Path))
		{
			if (Cfg && Cfg->IsValid()) UE_LOG(LogTemp, Warning, TEXT("Config path '%s' not found, using default: %s"), *Path, DefaultValue ? TEXT("true") : TEXT("false"));
			return DefaultValue;
		}
		return Cfg->GetOrDefaultBool(Path, DefaultValue);
	}

	static FVector2D GetVector2DOrDefault(UEnvironmentConfig* Cfg, const FString& Path, const FVector2D& DefaultValue)
	{
		if (!Cfg || !Cfg->IsValid() || !Cfg->HasPath(Path))
		{
			if (Cfg && Cfg->IsValid()) UE_LOG(LogTemp, Warning, TEXT("Config path '%s' not found, using default: (%f, %f)"), *Path, DefaultValue.X, DefaultValue.Y);
			return DefaultValue;
		}
		return Cfg->GetVector2DOrDefault(Path, DefaultValue);
	}

	static TArray<float> GetArrayOrDefault(UEnvironmentConfig* Cfg, const FString& Path, const TArray<float>& DefaultValue)
	{
		if (!Cfg || !Cfg->IsValid() || !Cfg->HasPath(Path))
		{
			if (Cfg && Cfg->IsValid()) UE_LOG(LogTemp, Warning, TEXT("Config path '%s' not found, using default array."), *Path);
			return DefaultValue;
		}
		return Cfg->GetArrayOrDefault(Path, DefaultValue);
	}
} // namespace ConfigHelpers


UMultiAgentGaussianWaveHeightMap::UMultiAgentGaussianWaveHeightMap()
	: NumAgents(5)
	, GridSize(50)
	, MinHeight(-2.f)
	, MaxHeight(2.f)
	, MinVel(-1.f)
	, MaxVel(1.f)
	, MinAmp(0.f)
	, MaxAmp(5.f)
	, MinSigma(0.2f)
	, MaxSigma(5.f)
	, MinAngVel(-0.5f)
	, MaxAngVel(0.5f)
	, ValuesPerAgent(6)
	, bUseActionDelta(true)
	, bAccumulatedWave(false)
	, AccumulatedWaveFadeGamma(0.99f)
	, bWrapAroundEdges(true)
	, DeltaVelScale(0.5f)
	, DeltaAmpScale(0.2f)
	, DeltaSigmaScale(0.05f)
	, DeltaAngVelScale(0.3f)
{
	// FinalWave will be initialized by FMatrix2D default constructor (0x0)
	// and then properly sized in InitializeFromConfig or Reset.
}

void UMultiAgentGaussianWaveHeightMap::InitializeFromConfig(UEnvironmentConfig* EnvConfig)
{
	if (!EnvConfig || !EnvConfig->IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("UMultiAgentGaussianWaveHeightMap::InitializeFromConfig - Null or invalid config provided!"));
		// Keep default values if config is bad
		FinalWave = FMatrix2D(GridSize, GridSize, 0.f); // Ensure FinalWave is sized with defaults
		InitializeAgents();
		return;
	}

	NumAgents = ConfigHelpers::GetOrDefaultInt(EnvConfig, TEXT("NumAgents"), NumAgents);
	GridSize = ConfigHelpers::GetOrDefaultInt(EnvConfig, TEXT("GridSize"), GridSize);
	MinHeight = ConfigHelpers::GetOrDefaultNumber(EnvConfig, TEXT("MinHeight"), MinHeight);
	MaxHeight = ConfigHelpers::GetOrDefaultNumber(EnvConfig, TEXT("MaxHeight"), MaxHeight);

	FVector2D VelRange = ConfigHelpers::GetVector2DOrDefault(EnvConfig, TEXT("VelMinMax"), FVector2D(MinVel, MaxVel));
	MinVel = VelRange.X; MaxVel = VelRange.Y;
	FVector2D AmpRange = ConfigHelpers::GetVector2DOrDefault(EnvConfig, TEXT("AmpMinMax"), FVector2D(MinAmp, MaxAmp));
	MinAmp = AmpRange.X; MaxAmp = AmpRange.Y;
	FVector2D SigRange = ConfigHelpers::GetVector2DOrDefault(EnvConfig, TEXT("SigMinMax"), FVector2D(MinSigma, MaxSigma));
	MinSigma = SigRange.X; MaxSigma = SigRange.Y;
	FVector2D AngVelRange = ConfigHelpers::GetVector2DOrDefault(EnvConfig, TEXT("AngVelRange"), FVector2D(MinAngVel, MaxAngVel));
	MinAngVel = AngVelRange.X; MaxAngVel = AngVelRange.Y;

	ValuesPerAgent = ConfigHelpers::GetOrDefaultInt(EnvConfig, TEXT("NumActions"), ValuesPerAgent);
	bUseActionDelta = ConfigHelpers::GetOrDefaultBool(EnvConfig, TEXT("bUseActionDelta"), bUseActionDelta);
	bAccumulatedWave = ConfigHelpers::GetOrDefaultBool(EnvConfig, TEXT("bAccumulatedWave"), bAccumulatedWave);
	AccumulatedWaveFadeGamma = ConfigHelpers::GetOrDefaultNumber(EnvConfig, TEXT("AccumulatedWaveFadeGamma"), AccumulatedWaveFadeGamma);

	bWrapAroundEdges = ConfigHelpers::GetOrDefaultBool(EnvConfig, TEXT("bWrapAroundEdges"), bWrapAroundEdges);

	DeltaVelScale = ConfigHelpers::GetOrDefaultNumber(EnvConfig, TEXT("DeltaVelScale"), DeltaVelScale);
	DeltaAmpScale = ConfigHelpers::GetOrDefaultNumber(EnvConfig, TEXT("DeltaAmpScale"), DeltaAmpScale);
	DeltaSigmaScale = ConfigHelpers::GetOrDefaultNumber(EnvConfig, TEXT("DeltaSigmaScale"), DeltaSigmaScale);
	DeltaAngVelScale = ConfigHelpers::GetOrDefaultNumber(EnvConfig, TEXT("DeltaAngVelScale"), DeltaAngVelScale);

	FinalWave = FMatrix2D(GridSize, GridSize, 0.f);
	InitializeAgents();
}

void UMultiAgentGaussianWaveHeightMap::Reset(int32 NewNumAgents)
{
	NumAgents = NewNumAgents > 0 ? NewNumAgents : NumAgents; // Ensure NewNumAgents is positive
	if (GridSize <= 0) GridSize = 1; // Ensure GridSize is positive before matrix creation

	FinalWave = FMatrix2D(GridSize, GridSize, 0.f);
	InitializeAgents();
}

void UMultiAgentGaussianWaveHeightMap::InitializeAgents()
{
	if (GridSize <= 0) {
		UE_LOG(LogTemp, Error, TEXT("UMultiAgentGaussianWaveHeightMap::InitializeAgents - GridSize is not positive (%d). Cannot initialize agents."), GridSize);
		Agents.Empty();
		return;
	}
	Agents.Reset(NumAgents);
	Agents.SetNum(NumAgents);

	for (int32 i = 0; i < NumAgents; i++)
	{
		FGaussianWaveAgent& A = Agents[i];
		A.Position.X = FMath::FRandRange(0.f, static_cast<float>(GridSize - 1)); // Ensure GridSize-1 is valid
		A.Position.Y = FMath::FRandRange(0.f, static_cast<float>(GridSize - 1));
		A.Orientation = FMath::FRandRange(0.f, 2.f * PI);
		A.Amplitude = FMath::FRandRange(MinAmp, MaxAmp);
		A.SigmaX = FMath::FRandRange(MinSigma, MaxSigma);
		A.SigmaY = FMath::FRandRange(MinSigma, MaxSigma);
		A.Velocity.X = FMath::FRandRange(MinVel, MaxVel);
		A.Velocity.Y = FMath::FRandRange(MinVel, MaxVel);
		A.AngularVelocity = FMath::FRandRange(MinAngVel, MaxAngVel);
	}
	ComputeFinalWave();
}

void UMultiAgentGaussianWaveHeightMap::Step(const TArray<float>& Actions, float DeltaTime)
{
	const int32 ExpectedActionCount = NumAgents * ValuesPerAgent;
	if (Actions.Num() != ExpectedActionCount)
	{
		UE_LOG(LogTemp, Error, TEXT("UMultiAgentGaussianWaveHeightMap::Step => Action array size mismatch (received=%d, needed=%d) for %d agents and %d valuesPerAgent"), Actions.Num(), ExpectedActionCount, NumAgents, ValuesPerAgent);
		return;
	}
	if (GridSize <= 0) {
		UE_LOG(LogTemp, Error, TEXT("UMultiAgentGaussianWaveHeightMap::Step - GridSize is not positive (%d). Cannot step."), GridSize);
		return;
	}


	for (int32 i = 0; i < NumAgents; i++)
	{
		const int32 BaseActionIndex = i * ValuesPerAgent;
		FGaussianWaveAgent& AgentState = Agents[i];

		float ActionVx = Actions[BaseActionIndex + 0];
		float ActionVy = Actions[BaseActionIndex + 1];
		float ActionAngVel = Actions[BaseActionIndex + 2];
		float ActionAmplitude = Actions[BaseActionIndex + 3];
		float ActionSigmaX = Actions[BaseActionIndex + 4];
		float ActionSigmaY = Actions[BaseActionIndex + 5];

		if (bUseActionDelta)
		{
			AgentState.Velocity.X += ActionVx * DeltaVelScale * DeltaTime;
			AgentState.Velocity.Y += ActionVy * DeltaVelScale * DeltaTime;
			AgentState.AngularVelocity += ActionAngVel * DeltaAngVelScale * DeltaTime;
			AgentState.Amplitude += ActionAmplitude * DeltaAmpScale * DeltaTime;
			AgentState.SigmaX += ActionSigmaX * DeltaSigmaScale * DeltaTime;
			AgentState.SigmaY += ActionSigmaY * DeltaSigmaScale * DeltaTime;
		}
		else
		{
			auto MapN11ToRange = [&](float ValN11, float Min, float Max) {
				float t = FMath::Clamp(0.5f * (ValN11 + 1.f), 0.f, 1.f);
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

	for (FGaussianWaveAgent& AgentState : Agents)
	{
		AgentState.Position += AgentState.Velocity * DeltaTime;
                AgentState.Orientation += AgentState.AngularVelocity * DeltaTime;
                // Keep orientation within [0, 2π) to avoid drift of large values
                AgentState.Orientation = FMath::Fmod(AgentState.Orientation, 2.f * PI);
                if (AgentState.Orientation < 0.f)
                {
                        AgentState.Orientation += 2.f * PI;
                }

		const float GridSizeF = static_cast<float>(GridSize);
		if (GridSizeF <= 0.f) continue; // Should not happen if GridSize check passed above

		if (bWrapAroundEdges)
		{
			AgentState.Position.X = FMath::Fmod(AgentState.Position.X, GridSizeF);
			if (AgentState.Position.X < 0.f) AgentState.Position.X += GridSizeF;

			AgentState.Position.Y = FMath::Fmod(AgentState.Position.Y, GridSizeF);
			if (AgentState.Position.Y < 0.f) AgentState.Position.Y += GridSizeF;
		}
		else
		{
			AgentState.Position.X = FMath::Clamp(AgentState.Position.X, 0.f, GridSizeF - 1.f);
			AgentState.Position.Y = FMath::Clamp(AgentState.Position.Y, 0.f, GridSizeF - 1.f);
		}

		AgentState.Amplitude = FMath::Clamp(AgentState.Amplitude, MinAmp, MaxAmp);
		AgentState.SigmaX = FMath::Clamp(AgentState.SigmaX, MinSigma, MaxSigma);
		AgentState.SigmaY = FMath::Clamp(AgentState.SigmaY, MinSigma, MaxSigma);
		AgentState.Velocity.X = FMath::Clamp(AgentState.Velocity.X, MinVel, MaxVel);
		AgentState.Velocity.Y = FMath::Clamp(AgentState.Velocity.Y, MinVel, MaxVel);
		AgentState.AngularVelocity = FMath::Clamp(AgentState.AngularVelocity, MinAngVel, MaxAngVel);
	}
	ComputeFinalWave();
}

void UMultiAgentGaussianWaveHeightMap::ComputeFinalWave()
{
	if (GridSize <= 0) { // Ensure FinalWave is valid
		if (FinalWave.GetNumRows() > 0 || FinalWave.GetNumColumns() > 0) FinalWave.Resize(0, 0); // Make it empty
		return;
	}
	if (FinalWave.GetNumRows() != GridSize || FinalWave.GetNumColumns() != GridSize) { // Ensure correctly sized
		FinalWave.Resize(GridSize, GridSize);
	}

	if (bAccumulatedWave) {
		FinalWave *= AccumulatedWaveFadeGamma;
	}
	else {
		FinalWave.Init(0.f);
	}

	for (const FGaussianWaveAgent& Agent : Agents)
	{
		FMatrix2D AgentWave = ComputeAgentWave(Agent);
		FinalWave += AgentWave;
	}
	FinalWave.Clip(MinHeight, MaxHeight);
}

FMatrix2D UMultiAgentGaussianWaveHeightMap::ComputeAgentWave(const FGaussianWaveAgent& Agent) const
{
	if (GridSize <= 0) return FMatrix2D(0, 0); // Return empty if GridSize is invalid

	FMatrix2D RowIndicesMat(GridSize, GridSize);
	FMatrix2D ColIndicesMat(GridSize, GridSize);

	for (int32 r = 0; r < GridSize; ++r)
	{
		for (int32 c = 0; c < GridSize; ++c)
		{
			RowIndicesMat[r][c] = static_cast<float>(r);
			ColIndicesMat[r][c] = static_cast<float>(c);
		}
	}

	FMatrix2D DeltaR = RowIndicesMat;
	DeltaR -= Agent.Position.X;
	FMatrix2D DeltaC = ColIndicesMat;
	DeltaC -= Agent.Position.Y;

	if (bWrapAroundEdges)
	{
		const float GridSizeF = static_cast<float>(GridSize);
		const float HalfGridSizeF = GridSizeF / 2.0f;
		if (GridSizeF > 0) { // Avoid division by zero or issues if GridSize is non-positive
			for (int32 r_idx = 0; r_idx < GridSize; ++r_idx)
			{
				for (int32 c_idx = 0; c_idx < GridSize; ++c_idx)
				{
					float dr_val = DeltaR[r_idx][c_idx];
					while (dr_val > HalfGridSizeF) dr_val -= GridSizeF;   // More robust fmod style wrapping
					while (dr_val < -HalfGridSizeF) dr_val += GridSizeF;
					DeltaR[r_idx][c_idx] = dr_val;

					float dc_val = DeltaC[r_idx][c_idx];
					while (dc_val > HalfGridSizeF) dc_val -= GridSizeF;
					while (dc_val < -HalfGridSizeF) dc_val += GridSizeF;
					DeltaC[r_idx][c_idx] = dc_val;
				}
			}
		}
	}

	float AngleRad = -Agent.Orientation;
	float CosTheta = FMath::Cos(AngleRad);
	float SinTheta = FMath::Sin(AngleRad);

	FMatrix2D RotatedX = (DeltaR * CosTheta) - (DeltaC * SinTheta);
	FMatrix2D RotatedY = (DeltaR * SinTheta) + (DeltaC * CosTheta);

	float DenomX = Agent.SigmaX * Agent.SigmaX;
	float DenomY = Agent.SigmaY * Agent.SigmaY;
	if (FMath::IsNearlyZero(DenomX)) DenomX = KINDA_SMALL_NUMBER; // Avoid division by zero
	if (FMath::IsNearlyZero(DenomY)) DenomY = KINDA_SMALL_NUMBER;

	FMatrix2D ExponentTerm = ((RotatedX * RotatedX) / DenomX) +
		((RotatedY * RotatedY) / DenomY);
	ExponentTerm *= -0.5f;

	FMatrix2D AgentWave = ExponentTerm.Exp();
	AgentWave *= Agent.Amplitude;

	return AgentWave;
}

float UMultiAgentGaussianWaveHeightMap::MapRange(float x, float inMin, float inMax, float outMin, float outMax) const
{
	if (FMath::IsNearlyZero(inMax - inMin)) return outMin;
	float t = FMath::Clamp((x - inMin) / (inMax - inMin), 0.f, 1.f);
	return FMath::Lerp(outMin, outMax, t);
}

float UMultiAgentGaussianWaveHeightMap::MapToN11(float x, float mn, float mx) const
{
	return MapRange(x, mn, mx, -1.f, 1.f);
}

TArray<float> UMultiAgentGaussianWaveHeightMap::GetAgentState(int32 AgentIndex) const
{
	TArray<float> StateVector;
	if (!Agents.IsValidIndex(AgentIndex) || GridSize <= 0)
	{
		// Return empty or default state for invalid index or GridSize
		return StateVector;
	}

	const FGaussianWaveAgent& AgentState = Agents[AgentIndex];
	const float GridSizeF = static_cast<float>(GridSize); // Use for normalization if GridSize > 0

	float PosX_N11 = GridSizeF > 1.f ? MapToN11(AgentState.Position.X, 0.f, GridSizeF - 1.f) : 0.f;
	float PosY_N11 = GridSizeF > 1.f ? MapToN11(AgentState.Position.Y, 0.f, GridSizeF - 1.f) : 0.f;

	float WrappedOrientation = FMath::Fmod(AgentState.Orientation, 2.f * PI);
	if (WrappedOrientation < 0.f) { WrappedOrientation += 2.f * PI; }
	float Ori_N11 = MapToN11(WrappedOrientation, 0.f, 2.f * PI);

	float Amp_N11 = MapToN11(AgentState.Amplitude, MinAmp, MaxAmp);
	float SigmaX_N11 = MapToN11(AgentState.SigmaX, MinSigma, MaxSigma);
	float SigmaY_N11 = MapToN11(AgentState.SigmaY, MinSigma, MaxSigma);
	float VelX_N11 = MapToN11(AgentState.Velocity.X, MinVel, MaxVel);
	float VelY_N11 = MapToN11(AgentState.Velocity.Y, MinVel, MaxVel);
	float AngVel_N11 = MapToN11(AgentState.AngularVelocity, MinAngVel, MaxAngVel);

	StateVector.Reserve(9);
	StateVector.Add(PosX_N11); StateVector.Add(PosY_N11); StateVector.Add(Ori_N11);
	StateVector.Add(Amp_N11); StateVector.Add(SigmaX_N11); StateVector.Add(SigmaY_N11);
	StateVector.Add(VelX_N11); StateVector.Add(VelY_N11); StateVector.Add(AngVel_N11);

	return StateVector;
}

const FMatrix2D& UMultiAgentGaussianWaveHeightMap::GetHeightMap() const
{
	return FinalWave;
}