#include "TerraShift/MultiAgentFractalWave3D.h"
#include "EnvironmentConfig.h"
#include "TerraShift/Matrix2D.h"
#include "Math/UnrealMathUtility.h"

void UMultiAgentFractalWave3D::InitializeFromConfig(UEnvironmentConfig* Config)
{
    if (!Config)
    {
        UE_LOG(LogTemp, Error, TEXT("UMultiAgentFractalWave3D::InitializeFromConfig - Null config!"));
        return;
    }

    
    if (!Config || !Config->IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("Could not find environment/MultiAgentFractalWave in config!"));
        return;
    }

    // read num_agents, image_size
    NumAgents = Config->HasPath("num_agents") ? Config->Get("num_agents")->AsInt() : 3;
    ImageSize = Config->HasPath("image_size") ? Config->Get("image_size")->AsInt() : 50;

    // agent_init
    UEnvironmentConfig* AgentInitCfg = Config->Get("agent_init");
    if (AgentInitCfg && AgentInitCfg->IsValid())
    {
        if (AgentInitCfg->HasPath("pos_range"))
        {
            TArray<float> pr = AgentInitCfg->Get("pos_range")->AsArrayOfNumbers();
            if (pr.Num() == 2)
                PosRange = FVector2D(pr[0], pr[1]);
        }
        if (AgentInitCfg->HasPath("pitch_range"))
        {
            TArray<float> rr = AgentInitCfg->Get("pitch_range")->AsArrayOfNumbers();
            if (rr.Num() == 2)
                PitchRange = FVector2D(rr[0], rr[1]);
        }
        if (AgentInitCfg->HasPath("yaw_range"))
        {
            TArray<float> yr = AgentInitCfg->Get("yaw_range")->AsArrayOfNumbers();
            if (yr.Num() == 2)
                YawRange = FVector2D(yr[0], yr[1]);
        }
        DefaultFOVDeg = AgentInitCfg->HasPath("fov_deg")
            ? AgentInitCfg->Get("fov_deg")->AsNumber()
            : 60.f;

        DefaultSampleDist = AgentInitCfg->HasPath("sample_dist")
            ? AgentInitCfg->Get("sample_dist")->AsNumber()
            : 10.f;
    }

    // fractal_init
    UEnvironmentConfig* FracCfg = Config->Get("fractal_init");
    if (FracCfg && FracCfg->IsValid())
    {
        if (FracCfg->HasPath("base_freq_range"))
        {
            TArray<float> bf = FracCfg->Get("base_freq_range")->AsArrayOfNumbers();
            if (bf.Num() == 2) BaseFreqRange = FVector2D(bf[0], bf[1]);
        }
        if (FracCfg->HasPath("lacunarity_range"))
        {
            TArray<float> lr = FracCfg->Get("lacunarity_range")->AsArrayOfNumbers();
            if (lr.Num() == 2)
                LacunarityRange = FVector2D(lr[0], lr[1]);
        }
        if (FracCfg->HasPath("gain_range"))
        {
            TArray<float> gr = FracCfg->Get("gain_range")->AsArrayOfNumbers();
            if (gr.Num() == 2)
                GainRange = FVector2D(gr[0], gr[1]);
        }
        if (FracCfg->HasPath("blend_weight_range"))
        {
            TArray<float> bw = FracCfg->Get("blend_weight_range")->AsArrayOfNumbers();
            if (bw.Num() == 2)
                BlendWeightRange = FVector2D(bw[0], bw[1]);
        }
    }

    PitchLimit = Config->HasPath("pitch_limit")
        ? Config->Get("pitch_limit")->AsNumber()
        : (PI / 2.f);

    bYawWrap = Config->HasPath("yaw_wrap")
        ? Config->Get("yaw_wrap")->AsBool()
        : true;

    Octaves = Config->HasPath("octaves")
        ? Config->Get("octaves")->AsNumber()
        : 3;

    // create NxN wave
    FinalWave = FMatrix2D(ImageSize, ImageSize, 0.f);

    // randomize
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

        // positions => normal sampling 
        float px = SampleNormalInRange(PosRange);
        float py = SampleNormalInRange(PosRange);
        float pz = SampleNormalInRange(PosRange);
        S.Pos3D = FVector(px, py, pz);

        // pitch, yaw => normal sampling if you want them centered in [PitchRange..YawRange]
        S.Pitch = SampleNormalInRange(PitchRange);
        S.Yaw = SampleNormalInRange(YawRange);

        // The rest as is (or also normal if you prefer)
        S.FOVDegrees = DefaultFOVDeg;
        S.ImageSize = ImageSize;
        S.SampleDist = DefaultSampleDist;

        // fractal params => normal sampling 
        S.BaseFreq = SampleNormalInRange(BaseFreqRange);
        S.Lacunarity = SampleNormalInRange(LacunarityRange);
        S.Gain = SampleNormalInRange(GainRange);
        S.BlendWeight = SampleNormalInRange(BlendWeightRange);

        // Set as a constant
        S.Octaves = Octaves;

        // NxN fractal image
        S.FractalImage.SetNumZeroed(ImageSize * ImageSize);
    }
}

void UMultiAgentFractalWave3D::Step(const TArray<FFractalAgentAction>& Actions, float DeltaTime)
{
    // 1) apply
    for (int32 i = 0; i < Actions.Num() && i < Agents.Num(); i++)
    {
        FFractalAgentState& A = Agents[i];
        const FFractalAgentAction& Act = Actions[i];

        A.Pos3D += (Act.dPos * DeltaTime);

        A.Pitch += (Act.dPitch * DeltaTime);
        float limit = PitchLimit;
        A.Pitch = FMath::Clamp(A.Pitch, -limit, limit);

        A.Yaw += (Act.dYaw * DeltaTime);
        if (bYawWrap)
        {
            while (A.Yaw < 0.f) { A.Yaw += 2.f * PI; }
            while (A.Yaw >= 2.f * PI) { A.Yaw -= 2.f * PI; }
        }

        A.BaseFreq = FMath::Max(0.01f, A.BaseFreq + Act.dBaseFreq * DeltaTime);
        A.Lacunarity = FMath::Clamp(A.Lacunarity + Act.dLacunarity * DeltaTime, 1.f, 5.f);
        A.Gain = FMath::Clamp(A.Gain + Act.dGain * DeltaTime, 0.f, 1.f);
        A.BlendWeight = FMath::Clamp(A.BlendWeight + Act.dBlendWeight * DeltaTime, 0.f, 5.f);
    }

    // 2) re-render => combine
    int32 N = ImageSize;
    TArray<float> WaveSums;
    WaveSums.SetNumZeroed(N * N);
    TArray<float> WeightSums;
    WeightSums.SetNumZeroed(N * N);

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

    for (int32 idx = 0; idx < N * N; idx++)
    {
        float w = (WeightSums[idx] == 0.f) ? 1e-6f : WeightSums[idx];
        float val = WaveSums[idx] / w; // in [-1..1]
        FinalWave[idx / N][idx % N] = val;
    }
}

const FMatrix2D& UMultiAgentFractalWave3D::GetWave() const
{
    return FinalWave;
}

TArray<float> UMultiAgentFractalWave3D::GetAgentFractalImage(int32 AgentIndex) const
{
    TArray<float> Empty;
    if (!Agents.IsValidIndex(AgentIndex)) return Empty;
    return Agents[AgentIndex].FractalImage;
}

TArray<float> UMultiAgentFractalWave3D::GetAgentStateVariables(int32 AgentIndex) const
{
    TArray<float> Result;
    if (!Agents.IsValidIndex(AgentIndex))
    {
        return Result; // empty
    }
    const FFractalAgentState& A = Agents[AgentIndex];
    // Collect a standard set of state variables:
    // [posX, posY, posZ, pitch, yaw, baseFreq, lacunarity, gain, blendWeight, fovDeg, sampleDist]
    Result.Add(A.Pos3D.X);
    Result.Add(A.Pos3D.Y);
    Result.Add(A.Pos3D.Z);
    Result.Add(A.Pitch);
    Result.Add(A.Yaw);
    Result.Add(A.BaseFreq);
    Result.Add(A.Lacunarity);
    Result.Add(A.Gain);
    Result.Add(A.BlendWeight);
    Result.Add(A.FOVDegrees);
    Result.Add(A.SampleDist);
    return Result;
}

void UMultiAgentFractalWave3D::RenderFractalForAgent(FFractalAgentState& Agent)
{
    int32 N = Agent.ImageSize;
    Agent.FractalImage.SetNumUninitialized(N * N);

    float halfFov = FMath::DegreesToRadians(Agent.FOVDegrees * 0.5f);

    for (int32 v = 0; v < N; v++)
    {
        float ndc_y = ((float)v / (N - 1)) * 2.f - 1.f;
        for (int32 u = 0; u < N; u++)
        {
            float ndc_x = ((float)u / (N - 1)) * 2.f - 1.f;

            // pinhole dir
            float cx = ndc_x * FMath::Tan(halfFov);
            float cy = -ndc_y * FMath::Tan(halfFov);
            float cz = 1.f;
            float length = FMath::Sqrt(cx * cx + cy * cy + cz * cz);
            cx /= length; cy /= length; cz /= length;

            // yaw
            float sy = FMath::Sin(Agent.Yaw);
            float cyw = FMath::Cos(Agent.Yaw);
            float rx = cx * cyw + cz * sy;
            float rz = -cx * sy + cz * cyw;
            float ry = cy;

            // pitch
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
                Agent.BaseFreq, Agent.Octaves,
                Agent.Lacunarity, Agent.Gain
            );
            int32 idx = v * N + u;
            Agent.FractalImage[idx] = val;
        }
    }
}

float UMultiAgentFractalWave3D::FractalSample3D(
    float X, float Y, float Z,
    float BaseFreq, int32 Octs, float Lacun, float Gn
) const
{
    // fBm with FMath::PerlinNoise3D
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
    // Midpoint and the range’s span
    float mid = 0.5f * (Range.X + Range.Y);
    float span = (Range.Y - Range.X);

    // For a typical normal distribution, ±3 std dev covers ~99.7% of samples.
    // So we define stdev = span/6 => most samples stay within [X..Y].
    float stdev = span / 6.f;

    // Setup distribution with mean=mid, stddev=stdev
    std::normal_distribution<float> dist(mid, stdev);

    // Sample
    float val = dist(GetNormalGenerator());

    // Clamp to the [X..Y] range
    val = FMath::Clamp(val, Range.X, Range.Y);
    return val;
}