#include "TerraShift/MultiAgentGaussianWaveHeightMap.h"
#include "Math/UnrealMathUtility.h"

// Helper macros for config reading
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

    // read core params
    NumAgents = EnvConfig->GetOrDefaultInt(TEXT("NumAgents"), 5);
    GridSize = EnvConfig->GetOrDefaultInt(TEXT("GridSize"), 50);

    MinHeight = EnvConfig->GetOrDefaultNumber(TEXT("MinHeight"), -2.f);
    MaxHeight = EnvConfig->GetOrDefaultNumber(TEXT("MaxHeight"), 2.f);

    // velocity range
    TArray<float> VelMinMax = EnvConfig->GetArrayOrDefault(TEXT("VelMinMax"), { -1.f, 1.f });
    if (VelMinMax.Num() >= 2)
    {
        MinVel = VelMinMax[0];
        MaxVel = VelMinMax[1];
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("VelMinMax must have at least 2 floats => defaulting to [-1..1]."));
        MinVel = -1.f;
        MaxVel = 1.f;
    }

    // amplitude range
    TArray<float> AmpMinMax = EnvConfig->GetArrayOrDefault(TEXT("AmpMinMax"), { 0.f, 5.f });
    if (AmpMinMax.Num() >= 2)
    {
        MinAmp = AmpMinMax[0];
        MaxAmp = AmpMinMax[1];
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("AmpMinMax must have 2 floats => defaulting to [0..5]."));
        MinAmp = 0.f;
        MaxAmp = 5.f;
    }

    // sigma range
    TArray<float> SigMinMax = EnvConfig->GetArrayOrDefault(TEXT("SigMinMax"), { 0.2f, 5.f });
    if (SigMinMax.Num() >= 2)
    {
        MinSigma = SigMinMax[0];
        MaxSigma = SigMinMax[1];
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("SigMinMax must have 2 floats => defaulting to [0.2..5]."));
        MinSigma = 0.2f;
        MaxSigma = 5.f;
    }

    // angular velocity range
    TArray<float> AngVelRange = EnvConfig->GetArrayOrDefault(TEXT("AngVelRange"), { -0.5f, 0.5f });
    if (AngVelRange.Num() >= 2)
    {
        MinAngVel = AngVelRange[0];
        MaxAngVel = AngVelRange[1];
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("AngVelRange must have 2 floats => defaulting to [-0.5..0.5]."));
        MinAngVel = -0.5f;
        MaxAngVel = 0.5f;
    }

    ValuesPerAgent = EnvConfig->GetOrDefaultInt(TEXT("NumActions"), 6);
    bUseActionDelta = EnvConfig->GetOrDefaultBool(TEXT("bUseActionDelta"), true);

    bAccumulatedWave = EnvConfig->GetOrDefaultBool(TEXT("bAccumulatedWave"), false);
    AccumulatedWaveFadeGamma = EnvConfig->GetOrDefaultNumber(TEXT("AccumulatedWaveFadeGamma"), 0.99f);

    DeltaVelScale = EnvConfig->GetOrDefaultNumber(TEXT("DeltaVelScale"), 0.5f);
    DeltaAmpScale = EnvConfig->GetOrDefaultNumber(TEXT("DeltaAmpScale"), 0.2f);
    DeltaSigmaScale = EnvConfig->GetOrDefaultNumber(TEXT("DeltaSigmaScale"), 0.05f);
    DeltaAngVelScale = EnvConfig->GetOrDefaultNumber(TEXT("DeltaAngVelScale"), 0.3f);

    // Initialize the NxN final wave matrix to zero
    FinalWave = FMatrix2D(GridSize, GridSize, 0.f);

    // Build agents
    InitializeAgents();
}

void UMultiAgentGaussianWaveHeightMap::Reset(int32 NewNumAgents)
{
    NumAgents = NewNumAgents;
    // Clear the wave
    FinalWave = FMatrix2D(GridSize, GridSize, 0.f);

    InitializeAgents();
}

void UMultiAgentGaussianWaveHeightMap::InitializeAgents()
{
    Agents.SetNum(NumAgents);

    for (int32 i = 0; i < NumAgents; i++)
    {
        FGaussianWaveAgent& A = Agents[i];
        // Random position in [0..GridSize - 1]
        A.Position.X = FMath::RandRange(0.f, (float)GridSize - 1.f);
        A.Position.Y = FMath::RandRange(0.f, (float)GridSize - 1.f);
        A.Orientation = FMath::RandRange(0.f, 2.f * PI);

        // random amplitude in [MinAmp..MaxAmp]
        A.Amplitude = FMath::FRandRange(MinAmp, MaxAmp);
        // random sigmas
        A.SigmaX = FMath::FRandRange(MinSigma, MaxSigma);
        A.SigmaY = FMath::FRandRange(MinSigma, MaxSigma);

        // random velocity
        A.Velocity.X = FMath::FRandRange(MinVel, MaxVel);
        A.Velocity.Y = FMath::FRandRange(MinVel, MaxVel);
        // random angular velocity
        A.AngularVelocity = FMath::FRandRange(MinAngVel, MaxAngVel);
    }

    // Build the wave from scratch
    ComputeFinalWave();
}

void UMultiAgentGaussianWaveHeightMap::Step(const TArray<float>& Actions, float DeltaTime)
{
    const int32 needed = NumAgents * ValuesPerAgent;
    if (Actions.Num() != needed)
    {
        UE_LOG(LogTemp, Error,
            TEXT("UMultiAgentGaussianWaveHeightMap::Step => mismatch (#=%d, needed=%d)"),
            Actions.Num(), needed);
        return;
    }

    // 1) Apply new actions to each agent
    for (int32 i = 0; i < NumAgents; i++)
    {
        const int32 baseIdx = i * ValuesPerAgent;
        FGaussianWaveAgent& A = Agents[i];

        float inVx = Actions[baseIdx + 0];
        float inVy = Actions[baseIdx + 1];
        float inAng = Actions[baseIdx + 2];
        float inAmp = Actions[baseIdx + 3];
        float inSx = Actions[baseIdx + 4];
        float inSy = Actions[baseIdx + 5];

        if (bUseActionDelta)
        {
            // interpret them as deltas in [-1..1] => scale => * DeltaTime
            float dvx = inVx * DeltaVelScale * DeltaTime;
            float dvy = inVy * DeltaVelScale * DeltaTime;
            float dang = inAng * DeltaAngVelScale * DeltaTime;
            float damp = inAmp * DeltaAmpScale * DeltaTime;
            float dsx = inSx * DeltaSigmaScale * DeltaTime;
            float dsy = inSy * DeltaSigmaScale * DeltaTime;

            // apply increments
            A.Velocity.X += dvx;
            A.Velocity.Y += dvy;
            A.AngularVelocity += dang;
            A.Amplitude += damp;
            A.SigmaX += dsx;
            A.SigmaY += dsy;
        }
        else
        {
            // interpret as "absolutes" => map [-1..1] -> config ranges
            auto lerp01 = [&](float v) { return 0.5f * (v + 1.f); };

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

    // 2) Integrate positions, clamp edges
    for (FGaussianWaveAgent& A : Agents)
    {
        // update position/orientation
        A.Position += A.Velocity * DeltaTime;
        A.Orientation += A.AngularVelocity * DeltaTime;

        // clamp (no wrapping):
        A.Position.X = FMath::Clamp(A.Position.X, 0.f, (float)GridSize - 1.f);
        A.Position.Y = FMath::Clamp(A.Position.Y, 0.f, (float)GridSize - 1.f);

        // clamp amplitude, sigma
        A.Amplitude = FMath::Clamp(A.Amplitude, MinAmp, MaxAmp);
        A.SigmaX = FMath::Clamp(A.SigmaX, MinSigma, MaxSigma);
        A.SigmaY = FMath::Clamp(A.SigmaY, MinSigma, MaxSigma);

        // clamp velocity & angular velocity
        A.Velocity.X = FMath::Clamp(A.Velocity.X, MinVel, MaxVel);
        A.Velocity.Y = FMath::Clamp(A.Velocity.Y, MinVel, MaxVel);
        A.AngularVelocity = FMath::Clamp(A.AngularVelocity, MinAngVel, MaxAngVel);
    }

    // 3) Recompute wave
    ComputeFinalWave();
}

void UMultiAgentGaussianWaveHeightMap::ComputeFinalWave()
{
    // Start from zero or build on existing wave map
    if (bAccumulatedWave) {
        FinalWave *= AccumulatedWaveFadeGamma;
    }
    else {
        FinalWave.Init(0.f);
    }
    
    // Add each agent's wave
    for (const FGaussianWaveAgent& Agent : Agents)
    {
        FMatrix2D waveMat = ComputeAgentWave(Agent);
        FinalWave += waveMat;
    }

    // Clip final wave to [MinHeight..MaxHeight]
    FinalWave.Clip(MinHeight, MaxHeight);
}

FMatrix2D UMultiAgentGaussianWaveHeightMap::ComputeAgentWave(const FGaussianWaveAgent& Agent) const
{
    // Create NxN "row" and "col" index matrices
    FMatrix2D rowMat(GridSize, GridSize, 0.f);
    FMatrix2D colMat(GridSize, GridSize, 0.f);

    // Fill rowMat[r][c] = r, colMat[r][c] = c
    for (int32 r = 0; r < GridSize; r++)
    {
        for (int32 c = 0; c < GridSize; c++)
        {
            rowMat[r][c] = static_cast<float>(r);
            colMat[r][c] = static_cast<float>(c);
        }
    }

    // Shift by agent's position => (dx, dy)
    rowMat -= Agent.Position.X; // rowMat = rowMat - Agent.Position.X
    colMat -= Agent.Position.Y; // colMat = colMat - Agent.Position.Y

    // Rotate by -Orientation
    float cosT = FMath::Cos(-Agent.Orientation);
    float sinT = FMath::Sin(-Agent.Orientation);

    // rx = rowMat*cosT - colMat*sinT
    // ry = rowMat*sinT + colMat*cosT
    FMatrix2D rx = (rowMat * cosT) - (colMat * sinT);
    FMatrix2D ry = (rowMat * sinT) + (colMat * cosT);

    // exponent = -0.5 * ((rx^2 / SigmaX^2) + (ry^2 / SigmaY^2))
    FMatrix2D rx2 = rx * rx; // elementwise square
    rx2 /= (Agent.SigmaX * Agent.SigmaX);

    FMatrix2D ry2 = ry * ry; // elementwise square
    ry2 /= (Agent.SigmaY * Agent.SigmaY);

    FMatrix2D exponent = rx2 + ry2;
    exponent *= -0.5f;

    // waveVal = Agent.Amplitude * exp(exponent)
    FMatrix2D waveMat = exponent.Exp();
    waveMat *= Agent.Amplitude;

    return waveMat;
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
    if (mx <= mn + 1e-6f)
    {
        return 0.f;
    }
    float ratio = (x - mn) / (mx - mn);
    ratio = FMath::Clamp(ratio, 0.f, 1.f);
    return ratio * 2.f - 1.f;
}

// -------------- GET AGENT STATE --------------

TArray<float> UMultiAgentGaussianWaveHeightMap::GetAgentState(int32 AgentIndex) const
{
    TArray<float> outArr;
    if (!Agents.IsValidIndex(AgentIndex))
    {
        return outArr;
    }

    const FGaussianWaveAgent& A = Agents[AgentIndex];

    // position => [0..GridSize] => => [-1..1]
    float posx = MapToN11(A.Position.X, 0.f, (float)GridSize);
    float posy = MapToN11(A.Position.Y, 0.f, (float)GridSize);

    // orientation => [0..2PI] => => [-1..1]
    float ori = FMath::Fmod(A.Orientation, 2.f * PI);
    if (ori < 0)
    {
        ori += 2.f * PI;
    }
    float orFrac = ori / (2.f * PI);
    float orN11 = orFrac * 2.f - 1.f;

    // amplitude => [MinAmp..MaxAmp] => [-1..1]
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