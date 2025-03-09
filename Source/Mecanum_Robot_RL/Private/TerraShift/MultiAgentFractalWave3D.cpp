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

    // -- Basic parameters --
    NumAgents = Config->HasPath("num_agents")
        ? Config->Get("num_agents")->AsInt()
        : 5;
    ImageSize = Config->HasPath("image_size")
        ? Config->Get("image_size")->AsInt()
        : 50;
    Octaves = Config->HasPath("octaves")
        ? Config->Get("octaves")->AsInt()
        : 3;

    // Wrap toggles
    if (Config->HasPath("wrap_pitch"))
        bWrapPitch = Config->Get("wrap_pitch")->AsBool();
    if (Config->HasPath("wrap_yaw"))
        bWrapYaw = Config->Get("wrap_yaw")->AsBool();
    if (Config->HasPath("wrap_roll"))
        bWrapRoll = Config->Get("wrap_roll")->AsBool();
    if (Config->HasPath("wrap_freq"))
        bWrapFreq = Config->Get("wrap_freq")->AsBool();
    if (Config->HasPath("wrap_lacunarity"))
        bWrapLacunarity = Config->Get("wrap_lacunarity")->AsBool();
    if (Config->HasPath("wrap_gain"))
        bWrapGain = Config->Get("wrap_gain")->AsBool();
    if (Config->HasPath("wrap_blend_weight"))
        bWrapBlendWeight = Config->Get("wrap_blend_weight")->AsBool();

    // agent_init
    UEnvironmentConfig* AInit = Config->Get("agent_init");
    if (AInit && AInit->IsValid())
    {
        if (AInit->HasPath("pitch_range"))
        {
            auto rr = AInit->Get("pitch_range")->AsArrayOfNumbers();
            if (rr.Num() == 2) PitchRange = FVector2D(rr[0], rr[1]);
        }
        else
            PitchRange = FVector2D(-1.57f, 1.57f);

        if (AInit->HasPath("yaw_range"))
        {
            auto yr = AInit->Get("yaw_range")->AsArrayOfNumbers();
            if (yr.Num() == 2) YawRange = FVector2D(yr[0], yr[1]);
        }
        else
            YawRange = FVector2D(0.f, 6.28318f);

        if (AInit->HasPath("roll_range"))
        {
            auto ro = AInit->Get("roll_range")->AsArrayOfNumbers();
            if (ro.Num() == 2) RollRange = FVector2D(ro[0], ro[1]);
        }
        else
            RollRange = FVector2D(0.f, 6.28318f);

        DefaultFOVDeg = AInit->HasPath("fov_deg")
            ? AInit->Get("fov_deg")->AsNumber()
            : 60.f;
        DefaultSampleDist = AInit->HasPath("sample_dist")
            ? AInit->Get("sample_dist")->AsNumber()
            : 10.f;
    }
    else
    {
        // fallback
        PitchRange = FVector2D(-1.57f, 1.57f);
        YawRange = FVector2D(0.f, 6.28318f);
        RollRange = FVector2D(0.f, 6.28318f);
        DefaultFOVDeg = 60.f;
        DefaultSampleDist = 10.f;
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
        else
            BaseFreqRange = FVector2D(0.01f, 2.0f);

        if (FInit->HasPath("lacunarity_range"))
        {
            auto lr = FInit->Get("lacunarity_range")->AsArrayOfNumbers();
            if (lr.Num() == 2) LacunarityRange = FVector2D(lr[0], lr[1]);
        }
        else
            LacunarityRange = FVector2D(1.f, 2.f);

        if (FInit->HasPath("gain_range"))
        {
            auto gr = FInit->Get("gain_range")->AsArrayOfNumbers();
            if (gr.Num() == 2) GainRange = FVector2D(gr[0], gr[1]);
        }
        else
            GainRange = FVector2D(0.f, 1.f);

        if (FInit->HasPath("blend_weight_range"))
        {
            auto bw = FInit->Get("blend_weight_range")->AsArrayOfNumbers();
            if (bw.Num() == 2) BlendWeightRange = FVector2D(bw[0], bw[1]);
        }
        else
            BlendWeightRange = FVector2D(0.f, 5.f);
    }
    else
    {
        // fallback
        BaseFreqRange = FVector2D(0.01f, 2.0f);
        LacunarityRange = FVector2D(1.f, 2.f);
        GainRange = FVector2D(0.f, 1.f);
        BlendWeightRange = FVector2D(0.f, 5.f);
    }

    // action_ranges
    UEnvironmentConfig* ActCfg = Config->Get("action_ranges");
    if (ActCfg && ActCfg->IsValid())
    {
        if (ActCfg->HasPath("pitch_minmax"))
        {
            auto pm = ActCfg->Get("pitch_minmax")->AsArrayOfNumbers();
            if (pm.Num() == 2) ActionPitchRange = FVector2D(pm[0], pm[1]);
        }
        else
            ActionPitchRange = FVector2D(-0.39f, 0.39f);

        if (ActCfg->HasPath("yaw_minmax"))
        {
            auto ym = ActCfg->Get("yaw_minmax")->AsArrayOfNumbers();
            if (ym.Num() == 2) ActionYawRange = FVector2D(ym[0], ym[1]);
        }
        else
            ActionYawRange = FVector2D(-0.39f, 0.39f);

        if (ActCfg->HasPath("roll_minmax"))
        {
            auto rm = ActCfg->Get("roll_minmax")->AsArrayOfNumbers();
            if (rm.Num() == 2) ActionRollRange = FVector2D(rm[0], rm[1]);
        }
        else
            ActionRollRange = FVector2D(-0.39f, 0.39f);

        if (ActCfg->HasPath("base_freq_minmax"))
        {
            auto bf = ActCfg->Get("base_freq_minmax")->AsArrayOfNumbers();
            if (bf.Num() == 2) ActionBaseFreqRange = FVector2D(bf[0], bf[1]);
        }
        else
            ActionBaseFreqRange = FVector2D(-0.02f, 0.02f);

        if (ActCfg->HasPath("lacunarity_minmax"))
        {
            auto lr = ActCfg->Get("lacunarity_minmax")->AsArrayOfNumbers();
            if (lr.Num() == 2) ActionLacunarityRange = FVector2D(lr[0], lr[1]);
        }
        else
            ActionLacunarityRange = FVector2D(-0.02f, 0.02f);

        if (ActCfg->HasPath("gain_minmax"))
        {
            auto gr = ActCfg->Get("gain_minmax")->AsArrayOfNumbers();
            if (gr.Num() == 2) ActionGainRange = FVector2D(gr[0], gr[1]);
        }
        else
            ActionGainRange = FVector2D(-0.15f, 0.15f);

        if (ActCfg->HasPath("blend_weight_minmax"))
        {
            auto bw = ActCfg->Get("blend_weight_minmax")->AsArrayOfNumbers();
            if (bw.Num() == 2) ActionBlendWeightRange = FVector2D(bw[0], bw[1]);
        }
        else
            ActionBlendWeightRange = FVector2D(-0.15f, 0.15f);
    }
    else
    {
        // fallback
        ActionPitchRange = FVector2D(-0.39f, 0.39f);
        ActionYawRange = FVector2D(-0.39f, 0.39f);
        ActionRollRange = FVector2D(-0.39f, 0.39f);
        ActionBaseFreqRange = FVector2D(-0.02f, 0.02f);
        ActionLacunarityRange = FVector2D(-0.02f, 0.02f);
        ActionGainRange = FVector2D(-0.15f, 0.15f);
        ActionBlendWeightRange = FVector2D(-0.15f, 0.15f);
    }

    // state_ranges
    UEnvironmentConfig* SR = Config->Get("state_ranges");
    if (SR && SR->IsValid())
    {
        if (SR->HasPath("pitch_range"))
        {
            auto rr = SR->Get("pitch_range")->AsArrayOfNumbers();
            if (rr.Num() == 2) StatePitchRange = FVector2D(rr[0], rr[1]);
        }
        else
            StatePitchRange = FVector2D(-1.57f, 1.57f);

        if (SR->HasPath("yaw_range"))
        {
            auto yr = SR->Get("yaw_range")->AsArrayOfNumbers();
            if (yr.Num() == 2) StateYawRange = FVector2D(yr[0], yr[1]);
        }
        else
            StateYawRange = FVector2D(0.f, 6.28318f);

        if (SR->HasPath("roll_range"))
        {
            auto ro = SR->Get("roll_range")->AsArrayOfNumbers();
            if (ro.Num() == 2) StateRollRange = FVector2D(ro[0], ro[1]);
        }
        else
            StateRollRange = FVector2D(0.f, 6.28318f);

        if (SR->HasPath("base_freq_range"))
        {
            auto bf = SR->Get("base_freq_range")->AsArrayOfNumbers();
            if (bf.Num() == 2) StateBaseFreqRange = FVector2D(bf[0], bf[1]);
        }
        else
            StateBaseFreqRange = FVector2D(0.01f, 2.0f);

        if (SR->HasPath("lacunarity_range"))
        {
            auto lr = SR->Get("lacunarity_range")->AsArrayOfNumbers();
            if (lr.Num() == 2) StateLacunarityRange = FVector2D(lr[0], lr[1]);
        }
        else
            StateLacunarityRange = FVector2D(1.f, 2.f);

        if (SR->HasPath("gain_range"))
        {
            auto gr = SR->Get("gain_range")->AsArrayOfNumbers();
            if (gr.Num() == 2) StateGainRange = FVector2D(gr[0], gr[1]);
        }
        else
            StateGainRange = FVector2D(0.f, 1.f);

        if (SR->HasPath("blend_weight_range"))
        {
            auto bw = SR->Get("blend_weight_range")->AsArrayOfNumbers();
            if (bw.Num() == 2) StateBlendWeightRange = FVector2D(bw[0], bw[1]);
        }
        else
            StateBlendWeightRange = FVector2D(0.f, 5.f);
    }
    else
    {
        // fallback for state ranges
        StatePitchRange = FVector2D(-1.57f, 1.57f);
        StateYawRange = FVector2D(0.f, 6.28318f);
        StateRollRange = FVector2D(0.f, 6.28318f);
        StateBaseFreqRange = FVector2D(0.01f, 2.0f);
        StateLacunarityRange = FVector2D(1.f, 2.f);
        StateGainRange = FVector2D(0.f, 1.f);
        StateBlendWeightRange = FVector2D(0.f, 5.f);
    }

    // Initialize final wave
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

        // Random init from configured ranges
        S.Pitch = SampleNormalInRange(PitchRange);
        S.Yaw = SampleNormalInRange(YawRange);
        S.Roll = SampleNormalInRange(RollRange);

        S.FOVDegrees = DefaultFOVDeg;
        S.SampleDist = DefaultSampleDist;
        S.ImageSize = ImageSize;
        S.Octaves = Octaves;

        S.BaseFreq = SampleNormalInRange(BaseFreqRange);
        S.Lacunarity = SampleNormalInRange(LacunarityRange);
        S.Gain = SampleNormalInRange(GainRange);
        S.BlendWeight = SampleNormalInRange(BlendWeightRange);

        S.FractalImage.SetNumZeroed(ImageSize * ImageSize);
    }
}

void UMultiAgentFractalWave3D::Step(const TArray<FFractalAgentAction>& Actions, float DeltaTime)
{
    for (int32 i = 0; i < Actions.Num() && i < Agents.Num(); i++)
    {
        FFractalAgentState& A = Agents[i];
        const FFractalAgentAction& Act = Actions[i];

        //
        // PITCH
        //
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

        //
        // YAW
        //
        float dyaw = ActionScaled(Act.dYaw, ActionYawRange.X, ActionYawRange.Y) * DeltaTime;
        float newYaw = A.Yaw + dyaw;
        if (bWrapYaw)
        {
            newYaw = WrapValue(newYaw, StateYawRange.X, StateYawRange.Y);
        }
        else
        {
            newYaw = FMath::Clamp(newYaw, StateYawRange.X, StateYawRange.Y);
        }
        A.Yaw = newYaw;

        //
        // ROLL
        //
        float droll = ActionScaled(Act.dRoll, ActionRollRange.X, ActionRollRange.Y) * DeltaTime;
        float newRoll = A.Roll + droll;
        if (bWrapRoll)
        {
            newRoll = WrapValue(newRoll, StateRollRange.X, StateRollRange.Y);
        }
        else
        {
            newRoll = FMath::Clamp(newRoll, StateRollRange.X, StateRollRange.Y);
        }
        A.Roll = newRoll;

        //
        // BaseFreq
        //
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

        //
        // Lacunarity
        //
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

        //
        // Gain
        //
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

        //
        // Blend Weight
        //
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

    // -- Re-render each fractal => combine wave
    int32 N = ImageSize;
    TArray<float> WaveSums;   WaveSums.SetNumZeroed(N * N);
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
        float val = WaveSums[idx] / w;
        val = FMath::Clamp(val, -1.f, 1.f); // just in case
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

    // pitch, yaw, roll => normalized
    float pitchNorm = NormalizeValue(A.Pitch, StatePitchRange.X, StatePitchRange.Y);
    float yawNorm = NormalizeValue(A.Yaw, StateYawRange.X, StateYawRange.Y);
    float rollNorm = NormalizeValue(A.Roll, StateRollRange.X, StateRollRange.Y);

    // fractal params => normalized
    float freqNorm = NormalizeValue(A.BaseFreq, StateBaseFreqRange.X, StateBaseFreqRange.Y);
    float lacNorm = NormalizeValue(A.Lacunarity, StateLacunarityRange.X, StateLacunarityRange.Y);
    float gainNorm = NormalizeValue(A.Gain, StateGainRange.X, StateGainRange.Y);
    float blendNorm = NormalizeValue(A.BlendWeight, StateBlendWeightRange.X, StateBlendWeightRange.Y);

    // e.g. {pitchNorm, yawNorm, rollNorm, freqNorm, lacNorm, gainNorm, blendNorm}
    Result.Add(pitchNorm);
    Result.Add(yawNorm);
    Result.Add(rollNorm);
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

    // The camera is at origin [0,0,0], we rotate each direction by pitch,yaw,roll
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

            // Yaw->Pitch->Roll rotation
            float sinY = FMath::Sin(Agent.Yaw);
            float cosY = FMath::Cos(Agent.Yaw);
            float rx = cx * cosY - cy * sinY;
            float ry = cx * sinY + cy * cosY;
            float rz = cz;

            float sinP = FMath::Sin(Agent.Pitch);
            float cosP = FMath::Cos(Agent.Pitch);
            float r2x = rx;
            float r2y = ry * cosP - rz * sinP;
            float r2z = ry * sinP + rz * cosP;

            float sinR = FMath::Sin(Agent.Roll);
            float cosR = FMath::Cos(Agent.Roll);
            float final_x = r2x * cosR + r2z * sinR;
            float final_y = r2y;
            float final_z = -r2x * sinR + r2z * cosR;

            // sample position in fractal space
            float sx = final_x * Agent.SampleDist;
            float sy = final_y * Agent.SampleDist;
            float sz = final_z * Agent.SampleDist;

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

float UMultiAgentFractalWave3D::FractalSample3D(
    float X, float Y, float Z,
    float BaseFreq, int32 Octs,
    float Lacun, float Gn
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

float UMultiAgentFractalWave3D::ActionScaled(float InputN11, float MinVal, float MaxVal) const
{
    float t = FMath::Clamp(InputN11, -1.f, 1.f);
    return Map(t, -1.f, 1.f, MinVal, MaxVal);
}

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

float UMultiAgentFractalWave3D::WrapValue(float val, float MinVal, float MaxVal) const
{
    float range = MaxVal - MinVal;
    if (range <= 0.f) return val;
    val -= MinVal;
    while (val < 0.f)    val += range;
    while (val >= range) val -= range;
    val += MinVal;
    return val;
}
