#include "TerraShift/MultiAgentGaussianWaveHeightMap.h"
#include "Math/UnrealMathUtility.h"

/** Helper macros to read config or log error. */
static float GAUSS_GetOrErrorNumber(UEnvironmentConfig* Cfg, const FString& Path)
{
    if (!Cfg->HasPath(*Path))
    {
        UE_LOG(LogTemp, Error, TEXT("Missing config path: %s"), *Path);
        return 0.f;
    }
    return Cfg->Get(*Path)->AsNumber();
}
static int32 GAUSS_GetOrErrorInt(UEnvironmentConfig* Cfg, const FString& Path)
{
    if (!Cfg->HasPath(*Path))
    {
        UE_LOG(LogTemp, Error, TEXT("Missing config path: %s"), *Path);
        return 0;
    }
    return Cfg->Get(*Path)->AsInt();
}

void UMultiAgentGaussianWaveHeightMap::InitializeFromConfig(UEnvironmentConfig* EnvConfig)
{
    if (!EnvConfig || !EnvConfig->IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("UMultiAgentGaussianWaveHeightMap => null or invalid config!"));
        return;
    }

    UEnvironmentConfig* WaveCfg = EnvConfig;

    // read core params
    NumAgents = WaveCfg->GetOrDefaultInt(TEXT("num_agents"), 5);
    GridSize = WaveCfg->GetOrDefaultInt(TEXT("grid_size"), 50);

    MinHeight = WaveCfg->GetOrDefaultNumber(TEXT("min_height"), -2.f);
    MaxHeight = WaveCfg->GetOrDefaultNumber(TEXT("max_height"), 2.f);

    // velocity
    TArray<float> VelMinMax = WaveCfg->GetArrayOrDefault(TEXT("velocity_minmax"), { -1.f, 1.f });
    if (VelMinMax.Num() >= 2)
    {
        MinVel = VelMinMax[0];
        MaxVel = VelMinMax[1];
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("velocity_minmax must have 2 floats, defaulting to [-1,1]!"));
        MinVel = -1.f; MaxVel = 1.f;
    }

    // amplitude
    TArray<float> AmpMinMax = WaveCfg->GetArrayOrDefault(TEXT("amplitude_minmax"), { 0.f,5.f });
    if (AmpMinMax.Num() >= 2)
    {
        MinAmp = AmpMinMax[0];
        MaxAmp = AmpMinMax[1];
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("amplitude_minmax must have 2 floats, default => [0..5]!"));
        MinAmp = 0; MaxAmp = 5;
    }

    // sigma
    TArray<float> SigMinMax = WaveCfg->GetArrayOrDefault(TEXT("sigma_minmax"), { 0.2f,5.f });
    if (SigMinMax.Num() >= 2)
    {
        MinSigma = SigMinMax[0];
        MaxSigma = SigMinMax[1];
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("sigma_minmax must have 2 floats => default= [0.2..5]"));
        MinSigma = 0.2f; MaxSigma = 5.f;
    }

    // angular velocity
    TArray<float> AngVelRange = WaveCfg->GetArrayOrDefault(TEXT("angular_vel_minmax"), { -0.5f, 0.5f });
    if (AngVelRange.Num() >= 2)
    {
        MinAngVel = AngVelRange[0];
        MaxAngVel = AngVelRange[1];
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("angular_vel_minmax must have 2 floats => default= [-0.5..0.5]"));
        MinAngVel = -0.5f; MaxAngVel = 0.5f;
    }

    ValuesPerAgent = WaveCfg->GetOrDefaultInt(TEXT("action_dim"), 6);

    // read whether we interpret actions as deltas
    bUseActionDelta = WaveCfg->GetOrDefaultBool(TEXT("bUseActionDelta"), true);

    // read delta scales
    DeltaVelScale = WaveCfg->GetOrDefaultNumber(TEXT("delta_velocity_scale"), 0.5f);
    DeltaAmpScale = WaveCfg->GetOrDefaultNumber(TEXT("delta_amp_scale"), 0.2f);
    DeltaSigmaScale = WaveCfg->GetOrDefaultNumber(TEXT("delta_sigma_scale"), 0.05f);
    DeltaAngVelScale = WaveCfg->GetOrDefaultNumber(TEXT("delta_angvel_scale"), 0.3f);

    // build NxN wave
    FinalWave = FMatrix2D(GridSize, GridSize, 0.f);
    InitializeAgents();
}

void UMultiAgentGaussianWaveHeightMap::Reset(int32 NewNumAgents)
{
    NumAgents = NewNumAgents;
    FinalWave = FMatrix2D(GridSize, GridSize, 0.f);
    InitializeAgents();
}

void UMultiAgentGaussianWaveHeightMap::InitializeAgents()
{
    Agents.SetNum(NumAgents);

    for (int32 i = 0; i < NumAgents; i++)
    {
        FGaussianWaveAgent& A = Agents[i];
        A.Position.X = FMath::RandRange(0.f, (float)GridSize - 1.f);
        A.Position.Y = FMath::RandRange(0.f, (float)GridSize - 1.f);
        A.Orientation = FMath::FRandRange(0.f, 2.f * PI);
        A.Amplitude = 0.f;
        A.SigmaX = 1.f;
        A.SigmaY = 1.f;
        A.Velocity = FVector2D::ZeroVector;
        A.AngularVelocity = 0.f;
    }
    ComputeFinalWave();
}

void UMultiAgentGaussianWaveHeightMap::Step(const TArray<float>& Actions, float DeltaTime)
{
    int32 needed = NumAgents * ValuesPerAgent;
    if (Actions.Num() != needed)
    {
        UE_LOG(LogTemp, Error, TEXT("UMultiAgentGaussianWaveHeightMap::Step => mismatch (#=%d, needed=%d)"),
            Actions.Num(), needed);
        return;
    }

    // parse
    for (int32 i = 0; i < NumAgents; i++)
    {
        const int32 bIdx = i * ValuesPerAgent;
        FGaussianWaveAgent& A = Agents[i];

        float inVx = Actions[bIdx + 0];
        float inVy = Actions[bIdx + 1];
        float inAng = Actions[bIdx + 2];
        float inAmp = Actions[bIdx + 3];
        float inSx = Actions[bIdx + 4];
        float inSy = Actions[bIdx + 5];

        if (bUseActionDelta)
        {
            // interpret [inVx..inSx] as deltas in [-1..1], scale, then * DeltaTime if desired
            float dvx = inVx * DeltaVelScale;
            float dvy = inVy * DeltaVelScale;
            float dang = inAng * DeltaAngVelScale;
            float damp = inAmp * DeltaAmpScale;
            float dsx = inSx * DeltaSigmaScale;
            float dsy = inSy * DeltaSigmaScale;

            // apply
            A.Velocity.X += dvx * DeltaTime;
            A.Velocity.Y += dvy * DeltaTime;
            A.AngularVelocity += dang * DeltaTime;
            A.Amplitude += damp * DeltaTime;
            A.SigmaX += dsx * DeltaTime;
            A.SigmaY += dsy * DeltaTime;
        }
        else
        {
            // interpret as "absolutes" in [-1..1], then map them into [MinVel..MaxVel], etc.
            auto lerp01 = [&](float v) {return (v + 1.f) * 0.5f; };

            float vxFrac = lerp01(inVx);
            float vyFrac = lerp01(inVy);
            float angFrac = lerp01(inAng);
            float ampFrac = lerp01(inAmp);
            float sxFrac = lerp01(inSx);
            float syFrac = lerp01(inSy);

            A.Velocity.X = FMath::Lerp(MinVel, MaxVel, vxFrac);
            A.Velocity.Y = FMath::Lerp(MinVel, MaxVel, vyFrac);
            A.AngularVelocity = FMath::Lerp(MinAngVel, MaxAngVel, angFrac);
            A.Amplitude = FMath::Lerp(MinAmp, MaxAmp, ampFrac);
            A.SigmaX = FMath::Lerp(MinSigma, MaxSigma, sxFrac);
            A.SigmaY = FMath::Lerp(MinSigma, MaxSigma, syFrac);
        }
    }

    // integrate
    for (FGaussianWaveAgent& A : Agents)
    {
        A.Position += A.Velocity * DeltaTime;
        A.Orientation += A.AngularVelocity * DeltaTime;

        // wrap
        A.Position.X = FMath::Fmod(A.Position.X, (float)GridSize);
        if (A.Position.X < 0) A.Position.X += GridSize;
        A.Position.Y = FMath::Fmod(A.Position.Y, (float)GridSize);
        if (A.Position.Y < 0) A.Position.Y += GridSize;

        // clamp amplitude, sigma
        A.Amplitude = FMath::Clamp(A.Amplitude, MinAmp, MaxAmp);
        A.SigmaX = FMath::Clamp(A.SigmaX, MinSigma, MaxSigma);
        A.SigmaY = FMath::Clamp(A.SigmaY, MinSigma, MaxSigma);

        // clamp velocity
        A.Velocity.X = FMath::Clamp(A.Velocity.X, MinVel, MaxVel);
        A.Velocity.Y = FMath::Clamp(A.Velocity.Y, MinVel, MaxVel);
        A.AngularVelocity = FMath::Clamp(A.AngularVelocity, MinAngVel, MaxAngVel);
    }

    ComputeFinalWave();
}

void UMultiAgentGaussianWaveHeightMap::ComputeFinalWave()
{
    FinalWave.Init(0.f);

    for (const FGaussianWaveAgent& A : Agents)
    {
        for (int32 r = 0; r < GridSize; r++)
        {
            for (int32 c = 0; c < GridSize; c++)
            {
                float oldVal = FinalWave[r][c];
                float gVal = Gauss2D(r, c, A);
                FinalWave[r][c] = oldVal + gVal;
            }
        }
    }

    FinalWave.Clip(MinHeight, MaxHeight);
}

float UMultiAgentGaussianWaveHeightMap::Gauss2D(float row, float col, const FGaussianWaveAgent& A) const
{
    float dx = row - A.Position.X;
    float dy = col - A.Position.Y;

    float cosT = FMath::Cos(-A.Orientation);
    float sinT = FMath::Sin(-A.Orientation);

    float rx = dx * cosT - dy * sinT;
    float ry = dx * sinT + dy * cosT;

    float exponent = -0.5f * ((rx * rx) / (A.SigmaX * A.SigmaX) + (ry * ry) / (A.SigmaY * A.SigmaY));
    return A.Amplitude * FMath::Exp(exponent);
}

// -------------- UTILITY MAPPERS --------------

float UMultiAgentGaussianWaveHeightMap::MapRange(
    float x, float inMin, float inMax, float outMin, float outMax) const
{
    if (FMath::IsNearlyZero(inMax - inMin))
    {
        return outMin;
    }
    float t = (x - inMin) / (inMax - inMin);
    return FMath::Lerp(outMin, outMax, FMath::Clamp(t, 0.f, 1.f));
}

float UMultiAgentGaussianWaveHeightMap::MapToN11(float x, float mn, float mx) const
{
    // map [mn..mx] => [-1..1]
    if (mx <= mn + 1e-6f) return 0.f;
    float ratio = (x - mn) / (mx - mn);
    ratio = FMath::Clamp(ratio, 0.f, 1.f);
    return ratio * 2.f - 1.f;
}

// -------------- GET AGENT STATE --------------

TArray<float> UMultiAgentGaussianWaveHeightMap::GetAgentState(int32 AgentIndex) const
{
    TArray<float> outArr;
    if (!Agents.IsValidIndex(AgentIndex)) return outArr;

    const FGaussianWaveAgent& A = Agents[AgentIndex];

    // position => [0..GridSize] => => [-1..1]
    float posx = MapToN11(A.Position.X, 0.f, (float)GridSize);
    float posy = MapToN11(A.Position.Y, 0.f, (float)GridSize);

    // orientation => [0..2PI] => => [-1..1]
    float ori = FMath::Fmod(A.Orientation, 2.f * PI);
    if (ori < 0) ori += 2.f * PI;
    float or01 = ori / (2.f * PI);
    float orN11 = or01 * 2.f - 1.f;

    // amplitude
    float ampN = MapToN11(A.Amplitude, MinAmp, MaxAmp);
    // sigma
    float sxN = MapToN11(A.SigmaX, MinSigma, MaxSigma);
    float syN = MapToN11(A.SigmaY, MinSigma, MaxSigma);

    // velocity => [MinVel..MaxVel]
    float vxN = MapToN11(A.Velocity.X, MinVel, MaxVel);
    float vyN = MapToN11(A.Velocity.Y, MinVel, MaxVel);

    // angular vel => [MinAngVel..MaxAngVel]
    float wN = MapToN11(A.AngularVelocity, MinAngVel, MaxAngVel);

    outArr.Add(posx);
    outArr.Add(posy);
    outArr.Add(orN11);
    outArr.Add(ampN);
    outArr.Add(sxN);
    outArr.Add(syN);
    outArr.Add(vxN);
    outArr.Add(vyN);
    outArr.Add(wN);

    return outArr;
}