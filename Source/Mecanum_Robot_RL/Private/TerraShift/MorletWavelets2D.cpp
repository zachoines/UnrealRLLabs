#include "TerraShift/MorletWavelets2D.h"

UMorletWavelets2D::UMorletWavelets2D()
{
}

void UMorletWavelets2D::Initialize(
    int32 InGridSizeX,
    int32 InGridSizeY,
    const FWaveParameterRanges& InRanges,
    float InDeltaScale
)
{
    GridSizeX = InGridSizeX;
    GridSizeY = InGridSizeY;

    WaveBounds.Ranges = InRanges;
    WaveBounds.DeltaScale = InDeltaScale;

    // Build XGrid, YGrid
    XGrid = FMatrix2D(GridSizeY, GridSizeX);
    YGrid = FMatrix2D(GridSizeY, GridSizeX);

    float XStep = 1.0f;
    float YStep = 1.0f;

    for (int32 r = 0; r < GridSizeY; ++r)
    {
        for (int32 c = 0; c < GridSizeX; ++c)
        {
            XGrid[r][c] = c * XStep;
            YGrid[r][c] = r * YStep;
        }
    }

    Heights = FMatrix2D(GridSizeY, GridSizeX, 0.0f);
    DeltaHeights = FMatrix2D(GridSizeY, GridSizeX, 0.0f);
}

void UMorletWavelets2D::Reset(int32 NumAgents)
{
    // Resize arrays for the specified number of agents
    AgentWaveStates.SetNum(NumAgents);
    AgentPositions.SetNum(NumAgents);

    // Use a helper to pick random values in [Min,Max]
    auto RandomInRange = [](const FVector2D& Range)
    {
        return FMath::FRandRange(Range.X, Range.Y);
    };


    const auto& R = WaveBounds.Ranges; // For convenience

    // For each agent, randomize wave parameters within min..max
    for (int32 i = 0; i < NumAgents; ++i)
    {
        AgentPositions[i].X = FMath::FRandRange(0.0f, GridSizeX);
        AgentPositions[i].Y = FMath::FRandRange(0.0f, GridSizeY);

        // Randomly set each parameter
        FAgentWaveState& WS = AgentWaveStates[i];
        WS.Velocity.X = FMath::FRandRange(R.VelocityRange.X, R.VelocityRange.Y);
        WS.Velocity.Y = FMath::FRandRange(R.VelocityRange.X, R.VelocityRange.Y);
        WS.Amplitude = RandomInRange(R.AmplitudeRange);
        WS.WaveOrientation = RandomInRange(R.WaveOrientationRange);
        WS.Wavenumber = RandomInRange(R.WavenumberRange);
        WS.PhaseVelocity = RandomInRange(R.PhaseVelocityRange);
        WS.Phase = RandomInRange(R.PhaseRange);
        WS.Sigma = RandomInRange(R.SigmaRange);
        WS.Time = 0.0f;
    }

    // Clear the existing height maps
    Heights.Init(0.0f);
    DeltaHeights.Init(0.0f);
}

const FMatrix2D& UMorletWavelets2D::Update(const TArray<FAgentDeltaParameters>& DeltaParameters)
{
    FMatrix2D NewHeightsMap(GridSizeY, GridSizeX, 0.0f);

    // For each agent, interpret the incoming deltas as relative param updates
    for (int32 i = 0; i < DeltaParameters.Num(); ++i)
    {
        if (!AgentWaveStates.IsValidIndex(i))
            continue;

        const FAgentDeltaParameters& Deltas = DeltaParameters[i];
        FAgentWaveState& WS = AgentWaveStates[i];
        const auto& R = WaveBounds.Ranges;
        float Scale = WaveBounds.DeltaScale;

        // 1) Update wave parameters by the deltas
        WS.Velocity.X = ApplyDelta(WS.Velocity.X, Deltas.Velocity.X, R.VelocityRange.X, R.VelocityRange.Y);
        WS.Velocity.Y = ApplyDelta(WS.Velocity.Y, Deltas.Velocity.Y, R.VelocityRange.X, R.VelocityRange.Y);

        WS.Amplitude = ApplyDelta(WS.Amplitude, Deltas.Amplitude, R.AmplitudeRange.X, R.AmplitudeRange.Y);
        WS.WaveOrientation = ApplyDelta(WS.WaveOrientation, Deltas.WaveOrientation, R.WaveOrientationRange.X, R.WaveOrientationRange.Y);
        WS.Wavenumber = ApplyDelta(WS.Wavenumber, Deltas.Wavenumber, R.WavenumberRange.X, R.WavenumberRange.Y);
        WS.PhaseVelocity = ApplyDelta(WS.PhaseVelocity, Deltas.PhaseVelocity, R.PhaseVelocityRange.X, R.PhaseVelocityRange.Y);
        WS.Phase = ApplyDelta(WS.Phase, Deltas.Phase, R.PhaseRange.X, R.PhaseRange.Y);
        WS.Sigma = ApplyDelta(WS.Sigma, Deltas.Sigma, R.SigmaRange.X, R.SigmaRange.Y);

        // Time is accumulated
        WS.Time += Deltas.Time;

        // 2) Update agent positions based on velocity + orientation
        float DeltaX = WS.Velocity.X * FMath::Cos(WS.WaveOrientation);
        float DeltaY = WS.Velocity.Y * FMath::Sin(WS.WaveOrientation);

        AgentPositions[i].X = FMath::Fmod(AgentPositions[i].X + DeltaX + GridSizeX, GridSizeX);
        AgentPositions[i].Y = FMath::Fmod(AgentPositions[i].Y + DeltaY + GridSizeY, GridSizeY);

        // 3) Generate wave heights if amplitude & sigma are valid
        if (WS.Amplitude == 0.0f || WS.Sigma < 0.01f)
            continue;

        float x_a = AgentPositions[i].X;
        float y_a = AgentPositions[i].Y;
        float A_a = WS.Amplitude;
        float theta = WS.WaveOrientation;
        float k_a = WS.Wavenumber;
        float omega = k_a * WS.PhaseVelocity;
        float phi = WS.Phase;
        float sigma = FMath::Max(WS.Sigma, 0.01f);
        float time = WS.Time;

        FMatrix2D XShifted = XGrid - x_a;
        FMatrix2D YShifted = YGrid - y_a;

        float cosT = FMath::Cos(theta);
        float sinT = FMath::Sin(theta);

        FMatrix2D XRot = (XShifted * cosT) + (YShifted * sinT);
        FMatrix2D YRot = (XShifted * -sinT) + (YShifted * cosT);

        FMatrix2D Envelope = ((XRot * XRot + YRot * YRot) / (-2.0f * sigma * sigma)).Exp();
        FMatrix2D Phase = (XRot * k_a) - (omega * time) + phi;
        FMatrix2D Wave = Envelope * (Phase.Cos()) * A_a;

        NewHeightsMap += Wave;
    }

    // Compute DeltaHeights
    DeltaHeights = NewHeightsMap - Heights;
    Heights = NewHeightsMap;
    return Heights;
}

const FMatrix2D& UMorletWavelets2D::GetHeights() const
{
    return Heights;
}

const FMatrix2D& UMorletWavelets2D::GetDeltaHeights() const
{
    return DeltaHeights;
}

FVector2f UMorletWavelets2D::GetAgentPosition(int32 AgentIndex) const
{
    return AgentPositions[AgentIndex];
}

/** Helper: clamp a param within [MinVal..MaxVal]. */
float UMorletWavelets2D::ClampParam(float Value, float MinVal, float MaxVal) const
{
    return FMath::Clamp(Value, MinVal, MaxVal);
}

/**
 * DeltaNormalized is in [-1..1]. We interpret it as a fraction (DeltaScale * Range).
 * Then we clamp the resulting new value to [MinVal..MaxVal].
 */
float UMorletWavelets2D::ApplyDelta(float CurrentValue, float DeltaNormalized, float MinVal, float MaxVal) const
{
    float Range = MaxVal - MinVal;
    float DeltaAmount = DeltaNormalized * WaveBounds.DeltaScale * Range;
    float NewValue = CurrentValue + DeltaAmount;
    return ClampParam(NewValue, MinVal, MaxVal);
}

/** Get current wave parameters for agent. */
FAgentWaveState UMorletWavelets2D::GetAgentWaveState(int32 AgentIndex)
{
    return AgentWaveStates[AgentIndex];
}
