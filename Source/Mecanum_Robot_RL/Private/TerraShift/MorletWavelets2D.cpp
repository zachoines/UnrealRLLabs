#include "TerraShift/MorletWavelets2D.h"
#include "Math/UnrealMathUtility.h"

// Constructor
MorletWavelets2D::MorletWavelets2D(int32 InGridSizeX, int32 InGridSizeY)
    : GridSizeX(InGridSizeX), GridSizeY(InGridSizeY)
{
    Initialize();
}

// Initialize grid coordinates and heights
void MorletWavelets2D::Initialize()
{
    XGrid = Matrix2D(GridSizeY, GridSizeX);
    YGrid = Matrix2D(GridSizeY, GridSizeX);

    float XStep = 1.0f;
    float YStep = 1.0f;

    for (int32 i = 0; i < GridSizeY; ++i)
    {
        for (int32 j = 0; j < GridSizeX; ++j)
        {
            XGrid[i][j] = j * XStep;
            YGrid[i][j] = i * YStep;
        }
    }

    Heights = Matrix2D(GridSizeY, GridSizeX, 0.0f);
}

// Reset function to center all agents on the grid
void MorletWavelets2D::Reset(int32 NumAgents)
{
    AgentPositions.Init(FVector2f(GridSizeX / 2.0f, GridSizeY / 2.0f), NumAgents);
    Heights.Init(0.0f);
}

// Update function with position propagation and height clamping
Matrix2D MorletWavelets2D::Update(const TArray<AgentParameters>& UpdatedParameters)
{
    Matrix2D NewHeights(GridSizeY, GridSizeX, 0.0f);

    for (int32 AgentIndex = 0; AgentIndex < UpdatedParameters.Num(); ++AgentIndex)
    {
        const AgentParameters& Agent = UpdatedParameters[AgentIndex];

        // Update agent position based on velocity and wrap within grid bounds
        float DeltaX = Agent.Velocity.X * FMath::Cos(Agent.WaveOrientation);
        float DeltaY = Agent.Velocity.Y * FMath::Sin(Agent.WaveOrientation);

        AgentPositions[AgentIndex].X = FMath::Fmod(AgentPositions[AgentIndex].X + DeltaX + GridSizeX, GridSizeX);
        AgentPositions[AgentIndex].Y = FMath::Fmod(AgentPositions[AgentIndex].Y + DeltaY + GridSizeY, GridSizeY);

        float x_a = AgentPositions[AgentIndex].X;
        float y_a = AgentPositions[AgentIndex].Y;
        float A_a = Agent.Amplitude;
        float theta_a = Agent.WaveOrientation;
        float k_a = Agent.Wavenumber;
        float phase_velocity_a = Agent.PhaseVelocity;
        float omega_a = k_a * phase_velocity_a;
        float phi_a = Agent.Phase;
        float sigma_a = FMath::Max(Agent.Sigma, 0.01f);
        float Time = Agent.Time;

        if (A_a == 0.0f || sigma_a < 0.01f) continue;

        // Shift coordinates relative to agent position
        Matrix2D XShifted = XGrid - x_a;
        Matrix2D YShifted = YGrid - y_a;

        // Rotate coordinates
        float CosTheta = FMath::Cos(theta_a);
        float SinTheta = FMath::Sin(theta_a);
        Matrix2D XRot = (XShifted * CosTheta) + (YShifted * SinTheta);
        Matrix2D YRot = (XShifted * -SinTheta) + (YShifted * CosTheta);

        // Gaussian envelope
        Matrix2D Envelope = ((XRot * XRot + YRot * YRot) / (-2.0f * sigma_a * sigma_a)).Exp();

        // Morlet wavelet calculation with phase
        Matrix2D Phase = XRot * k_a - omega_a * Time + phi_a;
        Matrix2D Wave = Envelope * (Phase.Cos()) * A_a;

        NewHeights += Wave;
    }

    // Smooth height transition with clamping
    for (int32 i = 0; i < GridSizeY; ++i)
    {
        for (int32 j = 0; j < GridSizeX; ++j)
        {
            float DeltaHeight = FMath::Clamp(NewHeights[i][j] - Heights[i][j], -MaxDeltaHeight, MaxDeltaHeight);
            Heights[i][j] += DeltaHeight;
        }
    }

    return Heights;
}

// Get the current height map
const Matrix2D& MorletWavelets2D::GetHeights() const
{
    return Heights;
}

// Get the current position of a specific agent
FVector2f MorletWavelets2D::GetAgentPosition(int32 AgentIndex) const
{
    return AgentPositions.IsValidIndex(AgentIndex) ? AgentPositions[AgentIndex] : FVector2f(0.0f, 0.0f);
}
