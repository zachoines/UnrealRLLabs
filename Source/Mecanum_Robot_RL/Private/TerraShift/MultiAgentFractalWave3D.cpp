#include "TerraShift/MultiAgentFractalWave3D.h"
#include "EnvironmentConfig.h"
#include "TerraShift/Matrix2D.h"
#include "Math/UnrealMathUtility.h"
#include <random>

void UMultiAgentFractalWave3D::InitializeFromConfig(UEnvironmentConfig* Config)
{
    if (!Config || !Config->IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("UMultiAgentFractalWave3D::InitializeFromConfig - Invalid config!"));
        return;
    }

    NumAgents = Config->HasPath("num_agents")
        ? Config->Get("num_agents")->AsInt()
        : 3;
    ImageSize = Config->HasPath("image_size")
        ? Config->Get("image_size")->AsInt()
        : 50;

    // octaves
    Octaves = Config->HasPath("octaves")
        ? Config->Get("octaves")->AsInt()
        : 4;

    // We no longer read pitch_limit, ignoring that.

    // If config has "yaw_wrap", we can read that:
    if (Config->HasPath("yaw_wrap"))
    {
        bWrapYaw = Config->Get("yaw_wrap")->AsBool();
    }

    // agent_init
    UEnvironmentConfig* AInit = Config->Get("agent_init");
    if (AInit && AInit->IsValid())
    {
        if (AInit->HasPath("pos_range"))
        {
            auto pr = AInit->Get("pos_range")->AsArrayOfNumbers();
            if (pr.Num() == 2) PosRange = FVector2D(pr[0], pr[1]);
        }
        if (AInit->HasPath("pitch_range"))
        {
            auto rr = AInit->Get("pitch_range")->AsArrayOfNumbers();
            if (rr.Num() == 2) PitchRange = FVector2D(rr[0], rr[1]);
        }
        if (AInit->HasPath("yaw_range"))
        {
            auto yr = AInit->Get("yaw_range")->AsArrayOfNumbers();
            if (yr.Num() == 2) YawRange = FVector2D(yr[0], yr[1]);
        }
        DefaultFOVDeg = AInit->HasPath("fov_deg")
            ? AInit->Get("fov_deg")->AsNumber()
            : 60.f;
        DefaultSampleDist = AInit->HasPath("sample_dist")
            ? AInit->Get("sample_dist")->AsNumber()
            : 10.f;
    }

    // fractal_init
    UEnvironmentConfig* FInit = Config->Get("fractal_init");
    if (FInit && FInit->IsValid())
    {
        if (FInit->HasPath("base_freq_range"))
        {
            auto bf = FInit->Get("base_freq_range")->AsArrayOfNumbers();
            if (bf.Num() == 2) BaseFreqRange = FVector2D(bf[0], bf[1]);
        }
        if (FInit->HasPath("lacunarity_range"))
        {
            auto lr = FInit->Get("lacunarity_range")->AsArrayOfNumbers();
            if (lr.Num() == 2) LacunarityRange = FVector2D(lr[0], lr[1]);
        }
        if (FInit->HasPath("gain_range"))
        {
            auto gr = FInit->Get("gain_range")->AsArrayOfNumbers();
            if (gr.Num() == 2) GainRange = FVector2D(gr[0], gr[1]);
        }
        if (FInit->HasPath("blend_weight_range"))
        {
            auto bw = FInit->Get("blend_weight_range")->AsArrayOfNumbers();
            if (bw.Num() == 2) BlendWeightRange = FVector2D(bw[0], bw[1]);
        }
    }

    // action_ranges
    UEnvironmentConfig* ActCfg = Config->Get("action_ranges");
    if (ActCfg && ActCfg->IsValid())
    {
        if (ActCfg->HasPath("pos_minmax"))
        {
            auto pm = ActCfg->Get("pos_minmax")->AsArrayOfNumbers();
            if (pm.Num() == 2) ActionPosRange = FVector2D(pm[0], pm[1]);
        }
        if (ActCfg->HasPath("pitch_minmax"))
        {
            auto pm = ActCfg->Get("pitch_minmax")->AsArrayOfNumbers();
            if (pm.Num() == 2) ActionPitchRange = FVector2D(pm[0], pm[1]);
        }
        if (ActCfg->HasPath("yaw_minmax"))
        {
            auto ym = ActCfg->Get("yaw_minmax")->AsArrayOfNumbers();
            if (ym.Num() == 2) ActionYawRange = FVector2D(ym[0], ym[1]);
        }
        if (ActCfg->HasPath("base_freq_minmax"))
        {
            auto bf = ActCfg->Get("base_freq_minmax")->AsArrayOfNumbers();
            if (bf.Num() == 2) ActionBaseFreqRange = FVector2D(bf[0], bf[1]);
        }
        if (ActCfg->HasPath("lacunarity_minmax"))
        {
            auto lr = ActCfg->Get("lacunarity_minmax")->AsArrayOfNumbers();
            if (lr.Num() == 2) ActionLacunarityRange = FVector2D(lr[0], lr[1]);
        }
        if (ActCfg->HasPath("gain_minmax"))
        {
            auto gr = ActCfg->Get("gain_minmax")->AsArrayOfNumbers();
            if (gr.Num() == 2) ActionGainRange = FVector2D(gr[0], gr[1]);
        }
        if (ActCfg->HasPath("blend_weight_minmax"))
        {
            auto bw = ActCfg->Get("blend_weight_minmax")->AsArrayOfNumbers();
            if (bw.Num() == 2) ActionBlendWeightRange = FVector2D(bw[0], bw[1]);
        }
    }

    // state_ranges
    UEnvironmentConfig* SR = Config->Get("state_ranges");
    if (SR && SR->IsValid())
    {
        if (SR->HasPath("pos_range"))
        {
            auto pr = SR->Get("pos_range")->AsArrayOfNumbers();
            if (pr.Num() == 2) StatePosRange = FVector2D(pr[0], pr[1]);
        }
        if (SR->HasPath("pitch_range"))
        {
            auto rr = SR->Get("pitch_range")->AsArrayOfNumbers();
            if (rr.Num() == 2) StatePitchRange = FVector2D(rr[0], rr[1]);
        }
        if (SR->HasPath("yaw_range"))
        {
            auto yr = SR->Get("yaw_range")->AsArrayOfNumbers();
            if (yr.Num() == 2) StateYawRange = FVector2D(yr[0], yr[1]);
        }
        if (SR->HasPath("base_freq_range"))
        {
            auto bf = SR->Get("base_freq_range")->AsArrayOfNumbers();
            if (bf.Num() == 2) StateBaseFreqRange = FVector2D(bf[0], bf[1]);
        }
        if (SR->HasPath("lacunarity_range"))
        {
            auto lr = SR->Get("lacunarity_range")->AsArrayOfNumbers();
            if (lr.Num() == 2) StateLacunarityRange = FVector2D(lr[0], lr[1]);
        }
        if (SR->HasPath("gain_range"))
        {
            auto gr = SR->Get("gain_range")->AsArrayOfNumbers();
            if (gr.Num() == 2) StateGainRange = FVector2D(gr[0], gr[1]);
        }
        if (SR->HasPath("blend_weight_range"))
        {
            auto bw = SR->Get("blend_weight_range")->AsArrayOfNumbers();
            if (bw.Num() == 2) StateBlendWeightRange = FVector2D(bw[0], bw[1]);
        }
    }

    // We'll also read wrap toggles from config if you want. Example:
    // if (Config->HasPath("wrap_pos")) { bWrapPos = Config->Get("wrap_pos")->AsBool(); }
    // etc.

    // create NxN wave
    FinalWave = FMatrix2D(ImageSize, ImageSize, 0.f);
    InitializeAgents();
}

void UMultiAgentFractalWave3D::Reset(int32 NewNumAgents)
{
    NumAgents = NewNumAgents;
    FinalWave = FMatrix2D(ImageSize, ImageSize, 0.f);
    InitializeAgents();
}

void UMultiAgentFractalWave3D::InitializeAgents()
{
    Agents.Reset();
    Agents.SetNum(NumAgents);

    for (int32 i = 0; i < NumAgents; i++)
    {
        FFractalAgentState& S = Agents[i];

        // random init from posRange etc.
        float px = SampleNormalInRange(PosRange);
        float py = SampleNormalInRange(PosRange);
        float pz = SampleNormalInRange(PosRange);
        S.Pos3D = FVector(px, py, pz);

        S.Pitch = SampleNormalInRange(PitchRange);
        S.Yaw = SampleNormalInRange(YawRange);

        S.FOVDegrees = DefaultFOVDeg;
        S.ImageSize = ImageSize;
        S.SampleDist = DefaultSampleDist;

        S.BaseFreq = SampleNormalInRange(BaseFreqRange);
        S.Lacunarity = SampleNormalInRange(LacunarityRange);
        S.Gain = SampleNormalInRange(GainRange);
        S.BlendWeight = SampleNormalInRange(BlendWeightRange);
        S.Octaves = Octaves;

        S.FractalImage.SetNumZeroed(ImageSize * ImageSize);
    }
}

void UMultiAgentFractalWave3D::Step(const TArray<FFractalAgentAction>& Actions, float DeltaTime)
{
    for (int32 i = 0; i < Actions.Num() && i < Agents.Num(); i++)
    {
        FFractalAgentState& A = Agents[i];
        const FFractalAgentAction& Act = Actions[i];

        // Position
        float dx = ActionScaled(Act.dPos.X, ActionPosRange.X, ActionPosRange.Y) * DeltaTime;
        float dy = ActionScaled(Act.dPos.Y, ActionPosRange.X, ActionPosRange.Y) * DeltaTime;
        float dz = ActionScaled(Act.dPos.Z, ActionPosRange.X, ActionPosRange.Y) * DeltaTime;

        float newPosX = A.Pos3D.X + dx;
        float newPosY = A.Pos3D.Y + dy;
        float newPosZ = A.Pos3D.Z + dz;
        if (bWrapPos)
        {
            newPosX = WrapValue(newPosX, StatePosRange.X, StatePosRange.Y);
            newPosY = WrapValue(newPosY, StatePosRange.X, StatePosRange.Y);
            newPosZ = WrapValue(newPosZ, StatePosRange.X, StatePosRange.Y);
        }
        else
        {
            newPosX = FMath::Clamp(newPosX, StatePosRange.X, StatePosRange.Y);
            newPosY = FMath::Clamp(newPosY, StatePosRange.X, StatePosRange.Y);
            newPosZ = FMath::Clamp(newPosZ, StatePosRange.X, StatePosRange.Y);
        }
        A.Pos3D = FVector(newPosX, newPosY, newPosZ);

        // Pitch
        float dpitch = ActionScaled(Act.dPitch, ActionPitchRange.X, ActionPitchRange.Y) * DeltaTime;
        float newPitch = A.Pitch + dpitch;
        if (bWrapPitch)
        {
            newPitch = WrapValue(newPitch, StatePitchRange.X, StatePitchRange.Y);
        }
        else
        {
            newPitch = FMath::Clamp(newPitch, StatePitchRange.X, StatePitchRange.Y);
        }
        A.Pitch = newPitch;

        // Yaw
        float dyaw = ActionScaled(Act.dYaw, ActionYawRange.X, ActionYawRange.Y) * DeltaTime;
        float newYaw = A.Yaw + dyaw;
        if (bWrapYaw)
        {
            // We do 0..2pi wrapping in the old approach:
            // but if you want to unify with StateYawRange, use:
            // newYaw = WrapValue(newYaw, StateYawRange.X, StateYawRange.Y);
            while (newYaw < 0.f)        newYaw += 2.f * PI;
            while (newYaw >= 2.f * PI) newYaw -= 2.f * PI;
        }
        else
        {
            newYaw = FMath::Clamp(newYaw, StateYawRange.X, StateYawRange.Y);
        }
        A.Yaw = newYaw;

        // Base Frequency
        float dfreq = ActionScaled(Act.dBaseFreq, ActionBaseFreqRange.X, ActionBaseFreqRange.Y) * DeltaTime;
        float newFreq = A.BaseFreq + dfreq;
        if (bWrapFreq)
        {
            newFreq = WrapValue(newFreq, StateBaseFreqRange.X, StateBaseFreqRange.Y);
        }
        else
        {
            newFreq = FMath::Clamp(newFreq, StateBaseFreqRange.X, StateBaseFreqRange.Y);
        }
        A.BaseFreq = newFreq;

        // Lacunarity
        float dlac = ActionScaled(Act.dLacunarity, ActionLacunarityRange.X, ActionLacunarityRange.Y) * DeltaTime;
        float newLac = A.Lacunarity + dlac;
        if (bWrapLacunarity)
        {
            newLac = WrapValue(newLac, StateLacunarityRange.X, StateLacunarityRange.Y);
        }
        else
        {
            newLac = FMath::Clamp(newLac, StateLacunarityRange.X, StateLacunarityRange.Y);
        }
        A.Lacunarity = newLac;

        // Gain
        float dgain = ActionScaled(Act.dGain, ActionGainRange.X, ActionGainRange.Y) * DeltaTime;
        float newGain = A.Gain + dgain;
        if (bWrapGain)
        {
            newGain = WrapValue(newGain, StateGainRange.X, StateGainRange.Y);
        }
        else
        {
            newGain = FMath::Clamp(newGain, StateGainRange.X, StateGainRange.Y);
        }
        A.Gain = newGain;

        // Blend Weight
        float dbw = ActionScaled(Act.dBlendWeight, ActionBlendWeightRange.X, ActionBlendWeightRange.Y) * DeltaTime;
        float newBW = A.BlendWeight + dbw;
        if (bWrapBlendWeight)
        {
            newBW = WrapValue(newBW, StateBlendWeightRange.X, StateBlendWeightRange.Y);
        }
        else
        {
            newBW = FMath::Clamp(newBW, StateBlendWeightRange.X, StateBlendWeightRange.Y);
        }
        A.BlendWeight = newBW;
    }

    // Re-render each fractal => combine wave
    int32 N = ImageSize;
    TArray<float> WaveSums; WaveSums.SetNumZeroed(N * N);
    TArray<float> WeightSums; WeightSums.SetNumZeroed(N * N);

    for (int32 i = 0; i < Agents.Num(); i++)
    {
        RenderFractalForAgent(Agents[i]);
        float w = Agents[i].BlendWeight;
        const TArray<float>& Img = Agents[i].FractalImage;
        for (int32 idx = 0; idx < N * N; idx++)
        {
            WaveSums[idx] += (Img[idx] * w);
            WeightSums[idx] += w;
        }
    }

    FinalWave = FMatrix2D(N, N, 0.f);
    for (int32 idx = 0; idx < N * N; idx++)
    {
        float w = (WeightSums[idx] == 0.f) ? 1e-6f : WeightSums[idx];
        float val = WaveSums[idx] / w; // in [-1..1]
        int32 row = idx / N;
        int32 col = idx % N;
        FinalWave[row][col] = val;
    }
}

const FMatrix2D& UMultiAgentFractalWave3D::GetWave() const
{
    return FinalWave;
}

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

    const FFractalAgentState& A = Agents[AgentIndex];

    // position => normalized
    float px = NormalizeValue(A.Pos3D.X, StatePosRange.X, StatePosRange.Y);
    float py = NormalizeValue(A.Pos3D.Y, StatePosRange.X, StatePosRange.Y);
    float pz = NormalizeValue(A.Pos3D.Z, StatePosRange.X, StatePosRange.Y);

    // pitch => normalized
    float pitchNorm = NormalizeValue(A.Pitch, StatePitchRange.X, StatePitchRange.Y);

    // yaw => normalize
    float yawNorm = NormalizeValue(A.Yaw, StateYawRange.X, StateYawRange.Y);

    // fractal params => normalize
    float freqNorm = NormalizeValue(A.BaseFreq, StateBaseFreqRange.X, StateBaseFreqRange.Y);
    float lacNorm = NormalizeValue(A.Lacunarity, StateLacunarityRange.X, StateLacunarityRange.Y);
    float gainNorm = NormalizeValue(A.Gain, StateGainRange.X, StateGainRange.Y);
    float blendNorm = NormalizeValue(A.BlendWeight, StateBlendWeightRange.X, StateBlendWeightRange.Y);

    Result.Add(px);
    Result.Add(py);
    Result.Add(pz);
    Result.Add(pitchNorm);
    Result.Add(yawNorm);
    Result.Add(freqNorm);
    Result.Add(lacNorm);
    Result.Add(gainNorm);
    Result.Add(blendNorm);

    return Result;
}

void UMultiAgentFractalWave3D::RenderFractalForAgent(FFractalAgentState& Agent)
{
    int32 N = Agent.ImageSize;
    Agent.FractalImage.SetNumUninitialized(N * N);

    float halfFov = FMath::DegreesToRadians(Agent.FOVDegrees * 0.5f);

    for (int32 v = 0; v < N; v++)
    {
        float ndc_y = (float(v) / (N - 1)) * 2.f - 1.f;
        for (int32 u = 0; u < N; u++)
        {
            float ndc_x = (float(u) / (N - 1)) * 2.f - 1.f;
            float cx = ndc_x * FMath::Tan(halfFov);
            float cy = -ndc_y * FMath::Tan(halfFov);
            float cz = 1.f;
            float length = FMath::Sqrt(cx * cx + cy * cy + cz * cz);
            cx /= length;
            cy /= length;
            cz /= length;

            // rotate by agent yaw
            float sy = FMath::Sin(Agent.Yaw);
            float cyw = FMath::Cos(Agent.Yaw);
            float rx = cx * cyw + cz * sy;
            float rz = -cx * sy + cz * cyw;
            float ry = cy;

            // rotate by agent pitch
            float sp = FMath::Sin(Agent.Pitch);
            float cp = FMath::Cos(Agent.Pitch);
            float final_x = rx;
            float final_y = ry * cp - rz * sp;
            float final_z = ry * sp + rz * cp;

            float sx = Agent.Pos3D.X + Agent.SampleDist * final_x;
            float sy_ = Agent.Pos3D.Y + Agent.SampleDist * final_y;
            float sz = Agent.Pos3D.Z + Agent.SampleDist * final_z;

            float val = FractalSample3D(
                sx, sy_, sz,
                Agent.BaseFreq,
                Agent.Octaves,
                Agent.Lacunarity,
                Agent.Gain
            );

            Agent.FractalImage[v * N + u] = val;
        }
    }
}

float UMultiAgentFractalWave3D::FractalSample3D(
    float X, float Y, float Z,
    float BaseFreq, int32 Octs, float Lacun, float Gn
) const
{
    float total = 0.f;
    float amplitude = 1.f;
    float freq = BaseFreq;
    for (int32 i = 0; i < Octs; i++)
    {
        float noiseVal = FMath::PerlinNoise3D(FVector(X * freq, Y * freq, Z * freq));
        total += amplitude * noiseVal;
        freq *= Lacun;
        amplitude *= Gn;
    }
    return FMath::Clamp(total, -1.f, 1.f);
}

std::mt19937& UMultiAgentFractalWave3D::GetNormalGenerator()
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return gen;
}

float UMultiAgentFractalWave3D::SampleNormalInRange(const FVector2D& Range)
{
    float mid = 0.5f * (Range.X + Range.Y);
    float span = (Range.Y - Range.X);
    float stdev = span / 6.f;
    std::normal_distribution<float> dist(mid, stdev);
    float val = dist(GetNormalGenerator());
    val = FMath::Clamp(val, Range.X, Range.Y);
    return val;
}

/** interpret a [-1..1] input => [MinVal..MaxVal]. */
float UMultiAgentFractalWave3D::ActionScaled(float InputN11, float MinVal, float MaxVal) const
{
    float t = FMath::Clamp(InputN11, -1.f, 1.f);
    return Map(t, -1.f, 1.f, MinVal, MaxVal);
}

/** interpret a value in [MinVal..MaxVal] => [-1..1]. */
float UMultiAgentFractalWave3D::NormalizeValue(float Value, float MinVal, float MaxVal) const
{
    if (MinVal >= MaxVal) return 0.f;
    float clipped = FMath::Clamp(Value, MinVal, MaxVal);
    float norm = ((clipped - MinVal) / (MaxVal - MinVal)) * 2.f - 1.f;
    return FMath::Clamp(norm, -1.f, 1.f);
}

float UMultiAgentFractalWave3D::Map(float x, float in_min, float in_max, float out_min, float out_max) const
{
    if (FMath::IsNearlyZero(in_max - in_min))
    {
        UE_LOG(LogTemp, Warning, TEXT("Map() - division by zero"));
        return out_min;
    }
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

/** Wrap 'val' into the domain [MinVal..MaxVal) using modular arithmetic. */
float UMultiAgentFractalWave3D::WrapValue(float val, float MinVal, float MaxVal) const
{
    float range = MaxVal - MinVal;
    if (range <= 0.f) return val; // no wrap possible
    val -= MinVal;
    while (val < 0.f)    val += range;
    while (val >= range) val -= range;
    val += MinVal;
    return val;
}
