#include "TerraShift/MultiAgentFractalWave3D.h"
#include "EnvironmentConfig.h"
#include "TerraShift/Matrix2D.h"
#include "Math/UnrealMathUtility.h"
#include <random>

// -------------------------------------
//  Initialization from Config
// -------------------------------------
void UMultiAgentFractalWave3D::InitializeFromConfig(UEnvironmentConfig* Config)
{
    if (!Config || !Config->IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("UMultiAgentFractalWave3D::InitializeFromConfig - Invalid config!"));
        return;
    }

    // Basic parameters
    NumAgents = Config->GetOrDefaultInt(TEXT("num_agents"), 5);
    ImageSize = Config->GetOrDefaultInt(TEXT("image_size"), 50);
    Octaves = Config->GetOrDefaultInt(TEXT("octaves"), 3);

    // Whether we do uniform-random init or midpoint
    // (default false => midpoint)
    bUniformRandomInit = Config->GetOrDefaultBool(TEXT("random_init"), false);

    // Wrap toggles for fractal params / camera
    bWrapFreq = Config->GetOrDefaultBool(TEXT("wrap_freq"), false);
    bWrapLacunarity = Config->GetOrDefaultBool(TEXT("wrap_lacunarity"), false);
    bWrapGain = Config->GetOrDefaultBool(TEXT("wrap_gain"), false);
    bWrapBlendWeight = Config->GetOrDefaultBool(TEXT("wrap_blend_weight"), false);
    bWrapSampleDist = Config->GetOrDefaultBool(TEXT("wrap_sampledist"), false);
    bWrapFOV = Config->GetOrDefaultBool(TEXT("wrap_fov"), false);

    // --- Ranges for fractal / camera init ---
    UEnvironmentConfig* FInit = Config->Get(TEXT("fractal_init"));
    if (FInit && FInit->IsValid())
    {
        auto bfRange = FInit->GetArrayOrDefault(TEXT("base_freq_range"), { 0.01, 2.0 });
        BaseFreqRange = FVector2D(bfRange[0], bfRange[1]);

        auto lacRange = FInit->GetArrayOrDefault(TEXT("lacunarity_range"), { 1.0, 2.0 });
        LacunarityRange = FVector2D(lacRange[0], lacRange[1]);

        auto gRange = FInit->GetArrayOrDefault(TEXT("gain_range"), { 0.0, 1.0 });
        GainRange = FVector2D(gRange[0], gRange[1]);

        auto bwRange = FInit->GetArrayOrDefault(TEXT("blend_weight_range"), { 0.0, 5.0 });
        BlendWeightRange = FVector2D(bwRange[0], bwRange[1]);

        // SampleDist and FOV can also be placed in "fractal_init" or
        // a separate block, up to you:
        auto sdRange = FInit->GetArrayOrDefault(TEXT("sample_dist_range"), { 1.0, 20.0 });
        SampleDistRange = FVector2D(sdRange[0], sdRange[1]);

        auto fovRange = FInit->GetArrayOrDefault(TEXT("fov_range"), { 30.0, 120.0 });
        FOVRange = FVector2D(fovRange[0], fovRange[1]);
    }
    else
    {
        // Fallback
        BaseFreqRange = FVector2D(0.01f, 2.0f);
        LacunarityRange = FVector2D(1.f, 2.f);
        GainRange = FVector2D(0.f, 1.f);
        BlendWeightRange = FVector2D(0.f, 5.f);
        SampleDistRange = FVector2D(1.f, 20.f);
        FOVRange = FVector2D(30.f, 120.f);
    }

    // --- Action Ranges for the 9D Action ---
    UEnvironmentConfig* ActCfg = Config->Get(TEXT("action_ranges"));
    if (ActCfg && ActCfg->IsValid())
    {
        ActionPitchRange = ActCfg->GetVector2DOrDefault("pitch_minmax", { -0.5f, 0.5f });
        ActionYawRange = ActCfg->GetVector2DOrDefault("yaw_minmax", { -0.5f, 0.5f });
        ActionRollRange = ActCfg->GetVector2DOrDefault("roll_minmax", { -0.5f, 0.5f });
        ActionBaseFreqRange = ActCfg->GetVector2DOrDefault("base_freq_minmax", { -0.02f, 0.02f });
        ActionLacunarityRange = ActCfg->GetVector2DOrDefault("lacunarity_minmax", { -0.02f, 0.02f });
        ActionGainRange = ActCfg->GetVector2DOrDefault("gain_minmax", { -0.15f, 0.15f });
        ActionBlendWeightRange = ActCfg->GetVector2DOrDefault("blend_weight_minmax", { -0.15f, 0.15f });
        ActionSampleDistRange = ActCfg->GetVector2DOrDefault("sampledist_minmax", { -1.0f, 1.0f });
        ActionFOVRange = ActCfg->GetVector2DOrDefault("fov_minmax", { -5.f, 5.f });
    }
    else
    {
        // Fallback action deltas
        ActionPitchRange = FVector2D(-0.5f, 0.5f);
        ActionYawRange = FVector2D(-0.5f, 0.5f);
        ActionRollRange = FVector2D(-0.5f, 0.5f);
        ActionBaseFreqRange = FVector2D(-0.02f, 0.02f);
        ActionLacunarityRange = FVector2D(-0.02f, 0.02f);
        ActionGainRange = FVector2D(-0.15f, 0.15f);
        ActionBlendWeightRange = FVector2D(-0.15f, 0.15f);
        ActionSampleDistRange = FVector2D(-1.f, 1.f);
        ActionFOVRange = FVector2D(-5.f, 5.f);
    }

    // Initialize final wave
    FinalWave = FMatrix2D(ImageSize, ImageSize, 0.f);

    // Create agent states
    InitializeAgents();
}

// -------------------------------------
//  Reset
// -------------------------------------
void UMultiAgentFractalWave3D::Reset(int32 NewNumAgents)
{
    NumAgents = NewNumAgents;
    FinalWave = FMatrix2D(ImageSize, ImageSize, 0.f);
    InitializeAgents();
}

// -------------------------------------
//  InitializeAgents
// -------------------------------------
void UMultiAgentFractalWave3D::InitializeAgents()
{
    Agents.Reset();
    Agents.SetNum(NumAgents);

    for (int32 i = 0; i < NumAgents; i++)
    {
        FFractalAgentState& State = Agents[i];
        State.ImageSize = ImageSize;
        State.Octaves = Octaves;

        if (bUniformRandomInit)
        {
            // Uniform random within each range
            State.Orientation = []() {
                // For a random orientation, pick random yaw/pitch/roll
                // then convert to quaternion:
                float randYaw = FMath::FRandRange(0.f, 2.f * PI);
                float randPitch = FMath::FRandRange(-PI / 2.f, PI / 2.f);
                float randRoll = FMath::FRandRange(0.f, 2.f * PI);
                FRotator rads = FRotator(FMath::RadiansToDegrees(randPitch),
                    FMath::RadiansToDegrees(randYaw),
                    FMath::RadiansToDegrees(randRoll));
                return rads.Quaternion();
                }();

            State.BaseFreq = UniformInRange(BaseFreqRange);
            State.Lacunarity = UniformInRange(LacunarityRange);
            State.Gain = UniformInRange(GainRange);
            State.BlendWeight = UniformInRange(BlendWeightRange);
            State.SampleDist = UniformInRange(SampleDistRange);
            State.FOVDegrees = UniformInRange(FOVRange);
        }
        else
        {
            // Midpoints of each range
            State.Orientation = FQuat::Identity; // "midpoint" orientation

            auto MidOf = [](const FVector2D& r) { return 0.5f * (r.X + r.Y); };
            State.BaseFreq = MidOf(BaseFreqRange);
            State.Lacunarity = MidOf(LacunarityRange);
            State.Gain = MidOf(GainRange);
            State.BlendWeight = MidOf(BlendWeightRange);
            State.SampleDist = MidOf(SampleDistRange);
            State.FOVDegrees = MidOf(FOVRange);
        }

        State.FractalImage.SetNumZeroed(ImageSize * ImageSize);
    }
}

// -------------------------------------
//  Step
// -------------------------------------
void UMultiAgentFractalWave3D::Step(const TArray<FFractalAgentAction>& Actions, float DeltaTime)
{
    // Apply each agent's actions to the state
    for (int32 i = 0; i < Actions.Num() && i < Agents.Num(); i++)
    {
        FFractalAgentState& State = Agents[i];
        const FFractalAgentAction& Act = Actions[i];

        // 1) Orientation update via small Euler-angle deltas => quaternion
        float dPitch = ActionScaled(Act.dPitch, ActionPitchRange) * DeltaTime;
        float dYaw = ActionScaled(Act.dYaw, ActionYawRange) * DeltaTime;
        float dRoll = ActionScaled(Act.dRoll, ActionRollRange) * DeltaTime;

        FQuat DeltaQ = BuildDeltaOrientation(dPitch, dYaw, dRoll);
        State.Orientation = (DeltaQ * State.Orientation).GetNormalized();

        // 2) BaseFreq
        float dfreq = ActionScaled(Act.dBaseFreq, ActionBaseFreqRange) * DeltaTime;
        float newFreq = State.BaseFreq + dfreq;
        if (bWrapFreq)
            newFreq = WrapValue(newFreq, BaseFreqRange.X, BaseFreqRange.Y);
        else
            newFreq = ClampInRange(newFreq, BaseFreqRange);
        State.BaseFreq = newFreq;

        // 3) Lacunarity
        float dlac = ActionScaled(Act.dLacunarity, ActionLacunarityRange) * DeltaTime;
        float newLac = State.Lacunarity + dlac;
        if (bWrapLacunarity)
            newLac = WrapValue(newLac, LacunarityRange.X, LacunarityRange.Y);
        else
            newLac = ClampInRange(newLac, LacunarityRange);
        State.Lacunarity = newLac;

        // 4) Gain
        float dgain = ActionScaled(Act.dGain, ActionGainRange) * DeltaTime;
        float newGain = State.Gain + dgain;
        if (bWrapGain)
            newGain = WrapValue(newGain, GainRange.X, GainRange.Y);
        else
            newGain = ClampInRange(newGain, GainRange);
        State.Gain = newGain;

        // 5) BlendWeight
        float dbw = ActionScaled(Act.dBlendWeight, ActionBlendWeightRange) * DeltaTime;
        float newBW = State.BlendWeight + dbw;
        if (bWrapBlendWeight)
            newBW = WrapValue(newBW, BlendWeightRange.X, BlendWeightRange.Y);
        else
            newBW = ClampInRange(newBW, BlendWeightRange);
        State.BlendWeight = newBW;

        // 6) SampleDist
        float dsample = ActionScaled(Act.dSampleDist, ActionSampleDistRange) * DeltaTime;
        float newSD = State.SampleDist + dsample;
        if (bWrapSampleDist)
            newSD = WrapValue(newSD, SampleDistRange.X, SampleDistRange.Y);
        else
            newSD = ClampInRange(newSD, SampleDistRange);
        State.SampleDist = newSD;

        // 7) FOV
        float dFov = ActionScaled(Act.dFOV, ActionFOVRange) * DeltaTime;
        float newFov = State.FOVDegrees + dFov;
        if (bWrapFOV)
            newFov = WrapValue(newFov, FOVRange.X, FOVRange.Y);
        else
            newFov = ClampInRange(newFov, FOVRange);
        State.FOVDegrees = newFov;
    }

    // Render each agent, combine final wave
    const int32 N = ImageSize;
    TArray<float> WaveSums;   WaveSums.SetNumZeroed(N * N);
    TArray<float> WeightSums; WeightSums.SetNumZeroed(N * N);

    for (int32 i = 0; i < Agents.Num(); i++)
    {
        RenderFractalForAgent(Agents[i]);

        float w = Agents[i].BlendWeight;
        const TArray<float>& img = Agents[i].FractalImage;

        for (int32 idx = 0; idx < N * N; idx++)
        {
            WaveSums[idx] += (img[idx] * w);
            WeightSums[idx] += w;
        }
    }

    // Build final wave
    FinalWave = FMatrix2D(N, N, 0.f);
    for (int32 idx = 0; idx < N * N; idx++)
    {
        float w = (WeightSums[idx] == 0.f) ? 1e-6f : WeightSums[idx];
        float val = WaveSums[idx] / w;
        val = FMath::Clamp(val, -1.f, 1.f);
        int32 row = idx / N;
        int32 col = idx % N;
        FinalWave[row][col] = val;
    }
}

// -------------------------------------
//  GetAgentFractalImage
// -------------------------------------
TArray<float> UMultiAgentFractalWave3D::GetAgentFractalImage(int32 AgentIndex) const
{
    if (!Agents.IsValidIndex(AgentIndex))
        return TArray<float>();
    return Agents[AgentIndex].FractalImage;
}

TArray<float> UMultiAgentFractalWave3D::GetAgentStateVariables(int32 AgentIndex) const
{
    TArray<float> Result;
    if (!Agents.IsValidIndex(AgentIndex))
        return Result;

    const FFractalAgentState& S = Agents[AgentIndex];

    // 1) Fractal & camera parameters, normalized
    float freqNorm = NormalizeValue(S.BaseFreq, BaseFreqRange);
    float lacNorm = NormalizeValue(S.Lacunarity, LacunarityRange);
    float gainNorm = NormalizeValue(S.Gain, GainRange);
    float bwNorm = NormalizeValue(S.BlendWeight, BlendWeightRange);
    float distNorm = NormalizeValue(S.SampleDist, SampleDistRange);
    float fovNorm = NormalizeValue(S.FOVDegrees, FOVRange);

    // 2) Add fractal/camera params to result
    //    e.g. { freqNorm, lacNorm, gainNorm, bwNorm, distNorm, fovNorm }
    Result.Add(freqNorm);
    // Result.Add(lacNorm);
    // Result.Add(gainNorm);
    Result.Add(bwNorm);
    Result.Add(distNorm);
    Result.Add(fovNorm);

    // 3) Orientation as quaternion components (each in [-1..1] for a unit quaternion)
    const FQuat& Q = S.Orientation;
    Result.Add(Q.X);
    Result.Add(Q.Y);
    Result.Add(Q.Z);
    Result.Add(Q.W);

    return Result;
}

// -------------------------------------
//  RenderFractalForAgent
// -------------------------------------
void UMultiAgentFractalWave3D::RenderFractalForAgent(FFractalAgentState& Agent)
{
    const int32 N = Agent.ImageSize;
    Agent.FractalImage.SetNumUninitialized(N * N);

    float halfFovRad = FMath::DegreesToRadians(Agent.FOVDegrees * 0.5f);

    for (int32 v = 0; v < N; v++)
    {
        float ndc_y = (float(v) / (N - 1)) * 2.f - 1.f;
        for (int32 u = 0; u < N; u++)
        {
            float ndc_x = (float(u) / (N - 1)) * 2.f - 1.f;

            // Base forward ray in local camera space
            float cx = ndc_x * FMath::Tan(halfFovRad);
            float cy = -ndc_y * FMath::Tan(halfFovRad);
            float cz = 1.f;

            // Normalize
            float length = FMath::Sqrt(cx * cx + cy * cy + cz * cz);
            cx /= length;
            cy /= length;
            cz /= length;

            // Rotate by the agent's quaternion orientation
            FVector localDir(cx, cy, cz);
            FVector worldDir = Agent.Orientation.RotateVector(localDir);

            // Sample fractal space
            float sx = worldDir.X * Agent.SampleDist;
            float sy = worldDir.Y * Agent.SampleDist;
            float sz = worldDir.Z * Agent.SampleDist;

            float val = FractalSample3D(
                sx, sy, sz,
                Agent.BaseFreq,
                Agent.Octaves,
                Agent.Lacunarity,
                Agent.Gain
            );

            Agent.FractalImage[v * N + u] = val;
        }
    }
}

// -------------------------------------
//  BuildDeltaOrientation
// -------------------------------------
FQuat UMultiAgentFractalWave3D::BuildDeltaOrientation(float dPitch, float dYaw, float dRoll) const
{
    // Each delta is in radians. If your action was intended in degrees,
    // you'd convert, but we assume radians here.
    // Create a small Rotator => then convert to quaternion.
    FRotator RotDelta(
        FMath::RadiansToDegrees(dPitch),
        FMath::RadiansToDegrees(dYaw),
        FMath::RadiansToDegrees(dRoll)
    );
    return RotDelta.Quaternion();
}

// -------------------------------------
//  FractalSample3D
// -------------------------------------
float UMultiAgentFractalWave3D::FractalSample3D(
    float X, float Y, float Z,
    float BaseFreq, int32 Octs,
    float Lacun, float Gn
) const
{
    /*float total = 0.f;
    float amplitude = 1.f;
    float freq = BaseFreq;
    for (int32 i = 0; i < Octs; i++)
    {
        float noiseVal = FMath::PerlinNoise3D(FVector(X * freq, Y * freq, Z * freq));
        total += amplitude * noiseVal;
        freq *= Lacun;
        amplitude *= Gn;
    }*/
    return FMath::Clamp(
        FMath::PerlinNoise3D(FVector(X * BaseFreq, Y * BaseFreq, Z * BaseFreq))
    , -1.f, 1.f);
}

// -------------------------------------
//  RNG Helpers
// -------------------------------------
std::mt19937& UMultiAgentFractalWave3D::GetGenerator()
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return gen;
}

float UMultiAgentFractalWave3D::UniformInRange(const FVector2D& Range)
{
    std::uniform_real_distribution<float> dist(Range.X, Range.Y);
    return dist(GetGenerator());
}

// -------------------------------------
//  Helper: scale [-1..1] action into some minmax range
// -------------------------------------
float UMultiAgentFractalWave3D::ActionScaled(float InputN11, const FVector2D& MinMax) const
{
    float t = FMath::Clamp(InputN11, -1.f, 1.f);
    // Map [-1..1] => [MinMax.X..MinMax.Y]
    float inSpan = 2.f;
    float outSpan = (MinMax.Y - MinMax.X);
    return MinMax.X + ((t + 1.f) / inSpan) * outSpan;
}

// -------------------------------------
//  Helper: wrap value in [MinVal, MaxVal]
// -------------------------------------
float UMultiAgentFractalWave3D::WrapValue(float val, float MinVal, float MaxVal) const
{
    float range = MaxVal - MinVal;
    if (range <= 0.f) return val;
    val -= MinVal;
    val = FMath::Fmod(val, range);
    if (val < 0.f) val += range;
    return val + MinVal;
}

// -------------------------------------
//  Helper: clamp value in [range.X..range.Y]
// -------------------------------------
float UMultiAgentFractalWave3D::ClampInRange(float val, const FVector2D& range) const
{
    return FMath::Clamp(val, range.X, range.Y);
}

// -------------------------------------
//  Helper: normalize value to [-1..1] given the range
// -------------------------------------
float UMultiAgentFractalWave3D::NormalizeValue(float val, const FVector2D& range) const
{
    float mn = range.X;
    float mx = range.Y;
    if (mn >= mx) return 0.f;

    float clipped = FMath::Clamp(val, mn, mx);
    float ratio = (clipped - mn) / (mx - mn);
    float mapped = (ratio * 2.f) - 1.f; // => [-1..1]
    return FMath::Clamp(mapped, -1.f, 1.f);
}